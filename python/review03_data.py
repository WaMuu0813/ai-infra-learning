from pathlib import Path
import torch
import torch.nn as nn
from review01_TinyTransformer import TinyTransformerLM
from review02_tokenizer import train_tokenizer_from_file


def get_batch(data, batch_size, context_length):
    # 每个样本需要 context_length + 1 个 token
    assert len(data) > context_length

    starts = torch.randint(
        low=0,
        high=len(data) - context_length,
        size=(batch_size,),
    )

    inputs = []
    targets = []

    for start in starts.tolist():
        # 输入：从 start 开始，取 context_length 个 token
        x = data[start : start + context_length]

        # 目标：起点后移一位，同样取 context_length 个 token
        y = data[start + 1 : start + context_length + 1]

        inputs.append(x)
        targets.append(y)

    return torch.stack(inputs), torch.stack(targets)


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=40,
    context_length=8,
):
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    eos_id = tokenizer.encode("<|endoftext|>")[0]

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 只把最近的 8 个 token 送入模型
            context = input_ids[:, -context_length:]

            logits = model(context)

            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            if next_token.item() == eos_id:
                print("停止原因：生成了 EOS")
                break

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )
        else:
            print("停止原因：达到最大生成长度")

    return tokenizer.decode(input_ids[0].tolist())


def main():
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "review_corpus.txt"

    tokenizer = train_tokenizer_from_file(
        corpus_path,
        num_merges=50,
        special_tokens=["<|endoftext|>"],
    )

    text = corpus_path.read_text(encoding="utf-8")
    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    x, y = get_batch(
        data,
        batch_size=2,
        context_length=8,
    )

    # print("语料 token 总数:", len(data))
    # print("输入:", x)
    # print("目标:", y)
    # print("输入形状:", x.shape)
    # print("目标形状:", y.shape)

    assert x.shape == (2, 8)
    assert y.shape == (2, 8)

    # 目标的前 7 个位置，应等于输入的后 7 个位置
    assert torch.equal(x[:, 1:], y[:, :-1])

    print("batch 检查通过")

    model = TinyTransformerLM(
        vocab_size=len(tokenizer),
        d_model=64,
        num_heads=8,
        d_ff=256,
        num_layers=2,
    )

    logits = model(x)

    print("模型输出形状:", logits.shape)

    assert logits.shape == (2, 8, len(tokenizer))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()

    for step in range(100):
        # 每一步重新抽取窗口
        x, y = get_batch(
            data,
            batch_size=4,
            context_length=8,
        )

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 99:
            print(f"step={step:03d}, loss={loss.item():.4f}")

    seen_ids = set(data.tolist())

    for text in ["我喜欢", "我喜欢学习"]:
        ids = tokenizer.encode(text)

        print("\n输入文本:", text)
        print("编码:", ids)

        for token_id in ids:
            print(
                "ID:",
                token_id,
                "内容:",
                tokenizer.vocab[token_id],
                "是否出现在训练 token 流中:",
                token_id in seen_ids,
            )
    for prompt in ["我喜欢", "我喜欢学习"]:
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=40,
            context_length=8,
        )
        print("Prompt:", prompt)
        print("生成结果:", text)


if __name__ == "__main__":
    main()
