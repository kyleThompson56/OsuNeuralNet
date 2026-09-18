import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from neuralNet import MovementNet
from config import windowLength

# -------------------------------------------------------------------------------
# 1) Define a history‐aware subclass: accepts 5 frames × 6 features = 30 inputs
# -------------------------------------------------------------------------------
class MovementNetHistory(MovementNet):
    def __init__(self):
        super().__init__()
        # replace first Linear layer: was in_features=5, now in_features=30
        self.model[0] = nn.Linear(windowLength * 6, 64)


# -------------------------------------------------------------------------------
# 2) Configuration
# -------------------------------------------------------------------------------
DATA_DIR   = "data"
MODEL_PATH = "models/movement_net.pt"
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR         = 1e-3


# -------------------------------------------------------------------------------
# 3) Find & load all CSVs
# -------------------------------------------------------------------------------
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in data/ – run training_capture.py first")

print(f"Using {len(csv_files)} CSV files for training:")
for path in csv_files:
    print("  ", path)

dfs = [pd.read_csv(path) for path in csv_files]
df  = pd.concat(dfs, ignore_index=True)


# -------------------------------------------------------------------------------
# 4) Compute dx, dy and normalize timestamp
# -------------------------------------------------------------------------------
df["timestamp"] = df["timestamp"].astype(np.float32)
df["dx"]        = df["target_x"] - df["cursor_x"]
df["dy"]        = df["target_y"] - df["cursor_y"]
df["dt"]        = df["timestamp"].diff().fillna(0.0).astype(np.float32)


# -------------------------------------------------------------------------------
# 5) Build sliding windows of 5 frames → flatten to 30‐dim inputs
# -------------------------------------------------------------------------------
feat_cols = [
    "cursor_x", "cursor_y",
    "target_x", "target_y",
    "target_radius",
    "timestamp"
]
targ_cols = ["dx", "dy"]

features = df[feat_cols].values.astype(np.float32)   # shape: [N, 6]
targets  = df[targ_cols].values.astype(np.float32)   # shape: [N, 2]

seq_len = windowLength
windows, labels = [], []

for i in range(seq_len - 1, len(df)):
    # stack frames [i-4, …, i] → shape [5,6]
    win = features[i - (seq_len - 1) : i + 1]
    windows.append(win.reshape(-1))   # flatten → [30]
    labels.append(targets[i])         # current dx, dy

X_np = np.stack(windows)              # shape [M, 30], M = N-4
y_np = np.stack(labels)               # shape [M, 2]

X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)


# -------------------------------------------------------------------------------
# 6) Create DataLoader
# -------------------------------------------------------------------------------
dataset    = TensorDataset(X, y)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)


# -------------------------------------------------------------------------------
# 7) Prepare model, loss, optimizer
# -------------------------------------------------------------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model     = MovementNetHistory().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# -------------------------------------------------------------------------------
# 8) Training loop with logging
# -------------------------------------------------------------------------------
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        preds = model(batch_X)
        loss  = criterion(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} — Avg Loss: {avg_loss:.6f}")


# -------------------------------------------------------------------------------
# 9) Save final model
# -------------------------------------------------------------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"History‐aware model saved to {MODEL_PATH}")
