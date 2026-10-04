import torch

model.eval()

total_eval_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids,
            attention_mask=attention_mask
        )

        # 从这里开始你写
        shift_logits = logits[:,:-1,:]
        shift_labels = labels[:,1:]
        V = torch.reshape(-1)
        flat_logits = shift_logits.reshape(-1,V)
        flat_labels = shift_labels.reshape(-1)
        loss = criterion(flat_logits,flat_labels)
        valid_tokens = (flat_labels != -100).sum().item()
        total_eval_loss += loss * valid_tokens
        total_tokens += valid_tokens

    avg_eval_loss = total_eval_loss / total_tokens