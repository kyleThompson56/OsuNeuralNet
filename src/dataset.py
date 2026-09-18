import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OsuHistoryDataset(Dataset):
    def __init__(
        self,
        csv_path,
        seq_len: int = 5,
        normalize: bool = True,
        screen_width: int = 1920,
        screen_height: int = 1080
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # check for required columns
        required = [
            "timestamp",
            "cursor_x", "cursor_y",
            "target_x", "target_y",
            "target_radius"
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column in CSV: {col}")

        # normalize coordinates / radius if requested
        if normalize:
            df["cursor_x"]    /= screen_width
            df["cursor_y"]    /= screen_height
            df["target_x"]    /= screen_width
            df["target_y"]    /= screen_height
            df["target_radius"] /= max(screen_width, screen_height)

        # compute frame‐wise deltas
        df["dx"] = df["target_x"] - df["cursor_x"]
        df["dy"] = df["target_y"] - df["cursor_y"]

        # extract raw numpy arrays
        feats = df[[
            "timestamp",
            "cursor_x", "cursor_y",
            "target_x", "target_y",
            "target_radius"
        ]].values.astype(np.float32)   # shape: [N,6]

        deltas = df[["dx", "dy"]].values.astype(np.float32)  # shape: [N,2]

        self.inputs = []
        self.targets = []
        N = len(df)

        # slide window
        for i in range(seq_len - 1, N):
            window = feats[i - (seq_len - 1) : i + 1]  # shape [seq_len,6]
            self.inputs.append(window.reshape(-1))     # flatten → [seq_len*6]
            self.targets.append(deltas[i])             # dx, dy at frame i

        self.inputs  = np.stack(self.inputs)   # [M, seq_len*6]
        self.targets = np.stack(self.targets)  # [M, 2]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


def get_history_dataloader(
    csv_path,
    seq_len=5,
    batch_size=32,
    shuffle=True,
    normalize=True,
    screen_width=1920,
    screen_height=1080
):
    dataset = OsuHistoryDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        normalize=normalize,
        screen_width=screen_width,
        screen_height=screen_height
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True
    )
