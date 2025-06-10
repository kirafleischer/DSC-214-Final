import os
import pandas as pd
import numpy as np
from utils import *
from transformation import *
from topological_model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets import *
from topological_transform import *

def main():
    os.makedirs("topological_output", exist_ok=True)

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------
    dataset_name    = "faces"
    identity_counts = [128] #[4, 8, 16, 32, 64, 128]
    splits          = ["train", "valid", "test"]
    total_epochs    = 20 #240
    epoch_block     = 20  # how many epochs per identity
    num_gpu         = 1
    idx_gpu         = 7   # The index of GPU that this task is about to run on
    num_workers     = 4
    batch_size      = 64
    lr              = 1e-3
    device = torch.device(f"cuda:{idx_gpu}" if torch.cuda.is_available() and torch.cuda.device_count() > idx_gpu else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------------
    # 1) Pre‑load all datasets
    # ------------------------------------------------------------------------
    all_datasets = {
        ident: { split: load_dataset(dataset_name, ident, split)
                for split in splits }
        for ident in identity_counts
    }

    # ------------------------------------------------------------------------
    # 2) Helper to map an epoch → identity
    # ------------------------------------------------------------------------
    def identity_for_epoch(epoch: int) -> int:
        idx = (epoch - 1) // epoch_block
        return identity_counts[idx]

    # ------------------------------------------------------------------------
    # 3) Load Pretrained Model
    # ------------------------------------------------------------------------
    
    # Load LPNet checkpoint
    checkpoint = torch.load("output/model.pt", map_location="cpu") 
    
    # Initialize LPNet and Model instances
    lpnet = LPNet(num_classes=128, pretrained=False)
    model = Model(num_classes=128, pretrained=False)
    
    # Load checkpoint into LPNet
    lpnet.load_state_dict(checkpoint)
    
    # Extract and load the ResNet backbone weights into Model
    resnet_state_dict = {k.replace("model.", ""): v for k, v in checkpoint.items() if k.startswith("model.")}
    model.resnet_model.load_state_dict(resnet_state_dict, strict=False)
    
    # Load fc1 and fc2 weights 
    model.fc1.weight.data = checkpoint['fc1.weight'].clone()
    model.fc1.bias.data = checkpoint['fc1.bias'].clone()
    model.fc2.weight.data = checkpoint['fc2.weight'].clone()
    model.fc2.bias.data = checkpoint['fc2.bias'].clone()
    model.to(device)
    print(f"model fc1: {model.fc1.weight.device}")
    feature_dims = [(64, 45, 45), (128, 23, 23), (256, 12, 12), (512, 6, 6)]
    TDA_losses = [TDA_Loss(feature_dim, num_persistence_features=32).to(device) for feature_dim in feature_dims]

    print(f"Model device: {next(model.parameters()).device}")
    print(f"TDA block 1 device: {next(TDA_losses[0].parameters()).device}")
    
    params = []
    for loss_net in TDA_losses:
        params += list(loss_net.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)

    losses = []

    history_df = pd.DataFrame(columns=[
                "epoch",
                "train_block1", "train_block2", "train_block3", "train_block4",
                "valid_block1", "valid_block2", "valid_block3", "valid_block4",
                "test_block1",  "test_block2",  "test_block3",  "test_block4"
            ])

    for epoch in range(1, total_epochs + 1):
        # 1) figure out which identity we're on
        ident = identity_for_epoch(epoch)

        # 2) re-create loaders for this identity
        train_loader = DataLoader(all_datasets[ident]["train"],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
        valid_loader = DataLoader(all_datasets[ident]["valid"],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)
        test_loader  = DataLoader(all_datasets[ident]["test"],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

        # 3) ----- TRAIN -----
        model.eval()

        for tda in TDA_losses:
            tda.train()

        train_losses = [[] for _ in range(4)]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
        
            with torch.no_grad():
                true_pers_image = compute_persistence_image(inputs).to(device)
                outputs, features = model(inputs) # features is 4-long list of tensors, one for each block, each shaped B,32

            optimizer.zero_grad()
            total_loss = 0
            for idx in range(4):
                block_loss = TDA_losses[idx](features[idx], true_pers_image)
                train_losses[idx].append(block_loss.item())
                total_loss += block_loss
            
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix({f"Block{i+1}": f"{np.mean(train_losses[i]):.4f}" for i in range(4)})
            
        losses.append([np.mean(train_losses[i]) for i in range(4)])

        # 4) ----- VALIDATION -----
        model.eval()
        valid_losses = [[] for _ in range(4)]

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                true_pers_image = compute_persistence_image(inputs).to(device)
                outputs, features = model(inputs)
                for idx in range(4):
                    block_loss = TDA_losses[idx](features[idx], true_pers_image)
                    valid_losses[idx].append(block_loss.item())

        # 5) ----- TEST -----
        test_losses  = [[] for _ in range(4)]

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                true_pers_image = compute_persistence_image(inputs).to(device)
                outputs, features = model(inputs)
                for idx in range(4):
                    block_loss = TDA_losses[idx](features[idx], true_pers_image)
                    test_losses[idx].append(block_loss.item())

        # ------------------------------------------------------------------------
        # Store results
        # ------------------------------------------------------------------------
        train_means = [np.mean(train_losses[i]) for i in range(4)]
        valid_means = [np.mean(valid_losses[i]) for i in range(4)]
        test_means  = [np.mean(test_losses[i])  for i in range(4)]

        print(f"→ Epoch {epoch}/{total_epochs}")
        for i in range(4):
            print(f"  Block {i+1}: "
                  f"Train {train_means[i]:.4f} | "
                  f"Valid {valid_means[i]:.4f} | "
                  f"Test {test_means[i]:.4f}")
        
        row = [epoch] + train_means + valid_means + test_means
        history_df.loc[len(history_df)] = row
        history_df.to_csv("topological_output/losses1.csv", index=False)

if __name__ == "__main__":
    main()