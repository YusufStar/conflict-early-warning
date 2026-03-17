"""
LSTM for country-level time series: sequence of events -> next period risk or event count.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

SEQ_LEN_DEFAULT = 12
HIDDEN_SIZE_DEFAULT = 64
EPOCHS_DEFAULT = 50
BATCH_SIZE = 32


def build_sequences(
    panel: "pd.DataFrame",
    seq_len: int = SEQ_LEN_DEFAULT,
    target_type: str = "binary",
    target_col: str | None = None,
    threshold: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y) from panel. X: (n, seq_len, 1) events per country; y: next-period target.
    Returns X, y, period_indices (last period of each sequence).
    """
    import pandas as pd
    out_X, out_y, out_period = [], [], []
    for country, g in panel.groupby("country"):
        g = g.sort_values("period_index")
        ev = g["events"].values.astype(np.float32)
        if target_col == "target_binary":
            y_vals = g["target_binary"].values
        else:
            y_vals = g["target_events"].values
        periods = g["period_index"].values
        for i in range(seq_len, len(ev)):
            out_X.append(ev[i - seq_len : i])
            # target at row i-1 = label for next period (row i)
            out_y.append(y_vals[i - 1])
            out_period.append(periods[i - 1])
    X = np.array(out_X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(out_y, dtype=np.float32 if target_type == "regression" else np.int64)
    return X, y, np.array(out_period)


def get_latest_sequences(panel: "pd.DataFrame", seq_len: int) -> Tuple[np.ndarray, list]:
    """One sequence per country: last seq_len months. Returns X (n_countries, seq_len, 1), country list."""
    out_X, countries = [], []
    for country, g in panel.groupby("country"):
        g = g.sort_values("period_index")
        ev = g["events"].values.astype(np.float32)
        if len(ev) < seq_len:
            continue
        out_X.append(ev[-seq_len:])
        countries.append(country)
    X = np.array(out_X, dtype=np.float32).reshape(-1, seq_len, 1)
    return X, countries


class EventLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = HIDDEN_SIZE_DEFAULT,
        num_layers: int = 1,
        target_type: str = "binary",
    ):
        super().__init__()
        self.target_type = target_type
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        if target_type == "binary":
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 1)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return self.out(out).squeeze(-1)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_type: str = "binary",
    hidden_size: int = HIDDEN_SIZE_DEFAULT,
    epochs: int = EPOCHS_DEFAULT,
    device: str | None = None,
) -> Tuple[nn.Module, dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    model = EventLSTM(input_size=1, hidden_size=hidden_size, target_type=target_type).to(device)
    if target_type == "binary":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            if target_type == "binary":
                yb = yb.float().unsqueeze(1).squeeze(-1)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history["train_loss"].append(epoch_loss / len(train_loader))
        model.eval()
        with torch.no_grad():
            Xv = torch.from_numpy(X_val).to(device)
            yv = torch.from_numpy(y_val).to(device)
            if target_type == "binary":
                yv = yv.float()
            vloss = criterion(model(Xv), yv).item()
        history["val_loss"].append(vloss)
    return model, history


def predict_lstm(
    model: nn.Module,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X).to(device)
        out = model(t).cpu().numpy()
    return out
