from math import log

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.randn(800,20)
y_train = torch.randint(0,3,(800,))
train_dataset = TensorDataset(X_train,y_train)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

X_eval = torch.randn(200,20)
y_eval = torch.randint(0,3,(200,))
eval_dataset = TensorDataset(X_eval,y_eval)
eval_loader = DataLoader(eval_dataset,batch_size=32,shuffle=False)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20,64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,3)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


criterion = nn.CrossEntropyLoss()
model = SimpleClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)
num_epochs = 3
cur_train_loss = 0
for epoch in range(num_epochs):
    model.train()
    cur_train_loss = 0
    total_samples = 0
    for xb,yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        batch = xb.size(0)
        logits = model(xb)
        loss = criterion(logits,yb)
        cur_train_loss += loss.item() * batch
        total_samples += batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = cur_train_loss / total_samples

    model.eval()
    total_samples = 0
    total_correct = 0
    total_eval_loss = 0
    with torch.no_grad():
        for xb,yb in eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            batch = xb.size(0)
            logits = model(xb)
            loss = criterion(logits,yb)
            total_eval_loss += loss.item() * batch
            pred = torch.argmax(logits,dim=1)
            total_correct += (pred == yb).sum().item()
            total_samples += batch

    avg_eval_loss = total_eval_loss / total_samples
    accuracy = total_correct / total_samples

    print(
        f"Epoch {epoch + 1} "
        f"train loss: {avg_train_loss:.4f} "
        f"eval loss: {avg_eval_loss:.4f} "
        f"accuracy: {accuracy:.4f}"
    )

    scheduler.step()
    check_point = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(check_point,"check_point.pt")
    



