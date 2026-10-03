from math import log
from random import shuffle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TensorDataset(X, y)

criterion = nn.CrossEntropyLoss()


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


model = SimpleClassifier().to(device)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    check_point = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(check_point, "check_point.pt")
