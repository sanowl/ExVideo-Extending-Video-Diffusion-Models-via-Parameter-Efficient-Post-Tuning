import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from typing import Dict, Any
from torchvision import datasets, transforms

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing the configuration file: {e}")
        raise
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        raise

def setup_distributed():
    """
    Set up distributed training environment.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            dist.init_process_group(backend="nccl", init_method='env://', world_size=world_size, rank=rank)
        else:
            dist.init_process_group(backend="gloo", init_method='env://', world_size=world_size, rank=rank)
    return rank, world_size

class ExVideoModel(nn.Module):
    """
    PyTorch implementation of the ExVideo model.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['model']['hidden_dim']
        self.num_frames = config['model']['num_frames']
        
        # Simple model layers for CIFAR-10 (3x32x32 images)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class YourDataset(Dataset):
    """
    Wrapper for CIFAR-10 dataset.
    """
    def __init__(self, train=True):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def train_epoch(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(state: Dict[str, Any], is_best: bool, filename: str):
    """
    Save model checkpoint.
    """
    torch.save(state, filename)
    if is_best:
        best_filename = 'best_' + filename
        torch.save(state, best_filename)
        print(f"Saved best model to {best_filename}")

def train(config, args):
    """
    Main training function.
    """
    rank, world_size = setup_distributed()
    is_distributed = world_size > 1

    # Device selection
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}" if is_distributed else "cuda")
    else:
        device = torch.device("cpu")

    # Model initialization
    model = ExVideoModel(config)
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    # Optimizer
    learning_rate = float(config['training']['learning_rate'])  # Ensure learning_rate is a float
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Dataset and DataLoader
    train_dataset = YourDataset(train=True)  # Use CIFAR-10 for training
    val_dataset = YourDataset(train=False)  # Use CIFAR-10 for validation

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], sampler=val_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Early stopping
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'], verbose=True)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['num_epochs']):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_loss = validate(model, val_dataloader, device)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Checkpointing
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, is_best, f"checkpoint_epoch_{epoch+1}.pth")

        # Early stopping check
        early_stopping(val_loss, model, 'early_stopping_checkpoint.pth')
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print("Training completed.")

def main():
    parser = argparse.ArgumentParser(description="ExVideo Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args)

if __name__ == "__main__":
    main()

