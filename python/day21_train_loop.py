# from turtle import forward

from numpy import arange
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 构造 X 和 y
X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))
dataset = TensorDataset(X, y)

# 3. Dataset / DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# 4. Model
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

# 5. loss function
criterion = nn.CrossEntropyLoss()

# 6. optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 7. training
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
num_epochs = 5
for epoch in arange(num_epochs):
    model.train()
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, " f"Loss: {loss.item():.4f}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        "check_point.pt",
    )
    scheduler.step()
