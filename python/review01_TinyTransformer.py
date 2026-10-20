from transformers import AutoTokenizer
import torch
import torch.nn as nn


def prepare_batch(texts, tokenizer, device):
    texts_with_eos = [text + tokenizer.eos_token for text in texts]

    batch = tokenizer(
        texts_with_eos,
        padding=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    batch = batch.to(device)
    labels = labels.to(device)

    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": labels,
    }


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean_square = x.square().mean(dim=-1, keepdim=True)
        rms = (mean_square + self.eps).sqrt()
        return (x / rms) * self.weight


def apply_rope(x, base=10000.0):
    # x: [B, H, T, Dh]
    _, _, T, Dh = x.shape
    assert Dh % 2 == 0

    pair_dims = torch.arange(
        0,
        Dh,
        2,
        device=x.device,
        dtype=torch.float32,
    )
    inv_freq = base ** (-pair_dims / Dh)

    positions = torch.arange(
        T,
        device=x.device,
        dtype=torch.float32,
    )

    # [T, Dh/2]
    angles = positions[:, None] * inv_freq[None, :]

    # [1, 1, T, Dh/2]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    a = x[..., 0::2]
    b = x[..., 1::2]

    a_rot = a * cos - b * sin
    b_rot = a * sin + b * cos

    return torch.stack((a_rot, b_rot), dim=-1).flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape

        # 投影并分头：[B, T, D] → [B, H, T, Dh]
        q = self.q_proj(x)
        q = q.reshape(B, T, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        k = self.k_proj(x)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = self.v_proj(x)
        v = v.reshape(B, T, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        # [B, H, T, T]
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # True 表示禁止关注未来位置
        causal_mask = torch.ones(
            T,
            T,
            device=x.device,
            dtype=torch.bool,
        ).triu(diagonal=1)

        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 当前数据使用右侧 padding
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :] == 0
            scores = scores.masked_fill(padding_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        # [B, H, T, Dh]
        context = weights @ v

        # 合并各头：[B, H, T, Dh] → [B, T, D]
        merged = context.transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(merged)

        return output, weights


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        activated_gate = gate * torch.sigmoid(gate)
        hidden = activated_gate * up

        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, attention_mask=None):
        attn_output, _ = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
        )
        x = x + attn_output

        ffn_output = self.ffn(self.ffn_norm(x))
        x = x + ffn_output

        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.final_norm(x)
        return self.lm_head(x)


def compute_lm_loss(logits, labels, criterion):
    # 位置 t 的输出预测位置 t+1 的 token
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    return criterion(
        shift_logits.reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1),
    )


def train_batch(model, batch, criterion, optimizer, steps=20):
    model.train()

    for step in range(steps):
        optimizer.zero_grad()

        logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = compute_lm_loss(logits, batch["labels"], criterion)

        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == steps - 1:
            print(f"step={step:02d}, loss={loss.item():.4f}")


def evaluate_batch(model, batch, criterion):
    model.eval()

    with torch.no_grad():
        logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = compute_lm_loss(logits, batch["labels"], criterion)

    return loss.item()


def generate(model, tokenizer, prompt, max_new_tokens=20):
    device = next(model.parameters()).device

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 单条无 padding 输入；每轮重新计算完整序列
            logits = model(input_ids)

            # 最后位置预测下一个 token
            next_logits = logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                print("停止原因：生成了 EOS")
                break
        else:
            print("停止原因：达到最大生成长度")

    return tokenizer.decode(
        input_ids[0].tolist(),
        skip_special_tokens=True,
    )


def main():
    torch.manual_seed(42)

    # 保持原来的 CPU 运行方式
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts = [
        "CUDA",
        "我喜欢学习CUDA",
        "我喜欢学习CUDA，因为GPU很适合并行计算",
    ]
    eval_texts = [
        "我正在学习GPU编程",
        "Transformer可以处理文本序列",
    ]

    train_data = prepare_batch(train_texts, tokenizer, device)
    eval_data = prepare_batch(eval_texts, tokenizer, device)

    model = TinyTransformerLM(
        vocab_size=len(tokenizer),
        d_model=64,
        num_heads=8,
        d_ff=256,
        num_layers=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 反复训练同一批数据，验证训练链路
    train_batch(
        model,
        train_data,
        criterion,
        optimizer,
        steps=20,
    )

    train_loss = evaluate_batch(model, train_data, criterion)
    eval_loss = evaluate_batch(model, eval_data, criterion)

    print(f"训练文本 loss: {train_loss:.4f}")
    print(f"新文本 loss:   {eval_loss:.4f}")

    text = generate(
        model,
        tokenizer,
        prompt="我喜欢",
        max_new_tokens=15,
    )
    print("生成结果:", text)


if __name__ == "__main__":
    main()
