import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
model.eval()

total_val_loss = 0.0
total_correct = 0
total_samples = 0

with torch.no_grad():
    for xb, yb in val_loader:
        # 1. 数据放到 device
        xb = xb.to(device)
        yb = yb.to(device)

        # 2. forward
        logits = model(xb)

        # 3. 计算 loss
        loss = criterion(logits,yb)

        # 4. 累计整个验证集的 loss
        batch_size = xb.size(0)
        total_val_loss += loss.item() * batch_size

        # 5. 从 logits 得到预测类别
        prediction = logits.argmax(dim=1)

        # 6. 累计预测正确数量
        total_correct += (prediction == yb).sum().item()

        # 7. 累计样本数量
        total_samples += batch_size


avg_val_loss = total_val_loss / total_samples
accuracy = total_correct / total_samples

print("val loss:", avg_val_loss)
print("accuracy:", accuracy)