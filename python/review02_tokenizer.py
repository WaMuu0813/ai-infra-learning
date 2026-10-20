import regex as re
from pathlib import Path

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(text):
    return [match.group(0) for match in re.finditer(PAT, text)]


def count_pairs(token_ids):
    counts = {}

    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def count_corpus_pairs(sequences):
    total_counts = {}

    for sequence in sequences:
        local_counts = count_pairs(sequence)

        for pair, count in local_counts.items():
            total_counts[pair] = total_counts.get(pair, 0) + count

    return total_counts


def merge_pair(token_ids, pair, new_id):
    result = []
    i = 0

    while i < len(token_ids):
        if i + 1 < len(token_ids) and (token_ids[i], token_ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(token_ids[i])
            i += 1

    return result


def train_bpe(text, num_merges, special_tokens=None):
    special_tokens = special_tokens or []

    # 先识别特殊 token，再对普通文本预分词
    sequences = []

    for part in split_special_tokens(text, special_tokens):
        if part in special_tokens:
            continue

        for piece in pre_tokenize(part):
            sequences.append(list(piece.encode("utf-8")))

    vocab = {i: bytes([i]) for i in range(256)}

    # 特殊 token 直接占据完整的词表条目
    for token in special_tokens:
        new_id = len(vocab)
        vocab[new_id] = token.encode("utf-8")

    merges = []

    for step in range(num_merges):
        counts = count_corpus_pairs(sequences)

        if not counts:
            break

        # 先比较频率；相同时比较实际字节串
        best_pair = max(
            counts,
            key=lambda pair: (
                counts[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )

        new_id = len(vocab)
        left, right = best_pair

        vocab[new_id] = vocab[left] + vocab[right]
        merges.append(best_pair)

        # 每个片段内部独立合并
        sequences = [merge_pair(sequence, best_pair, new_id) for sequence in sequences]

        print(
            f"第 {step + 1} 次合并: "
            f"{best_pair} → {new_id}, "
            f"内容={vocab[new_id]!r}, "
            f"各片段={sequences}"
        )

    return vocab, merges, sequences


def encode(text, merges, special_tokens=None):
    special_tokens = special_tokens or []

    special_ids = {token: 256 + i for i, token in enumerate(special_tokens)}

    # 跳过特殊 token 已经占用的 ID
    merge_start_id = 256 + len(special_tokens)

    result = []

    for part in split_special_tokens(text, special_tokens):
        if part in special_ids:
            result.append(special_ids[part])
            continue

        for piece in pre_tokenize(part):
            token_ids = list(piece.encode("utf-8"))

            for new_id, pair in enumerate(merges, start=merge_start_id):
                token_ids = merge_pair(token_ids, pair, new_id)

            result.extend(token_ids)

    return result


def decode(token_ids, vocab):
    raw = b"".join(vocab[token_id] for token_id in token_ids)
    return raw.decode("utf-8", errors="replace")


def split_special_tokens(text, special_tokens):
    if not special_tokens:
        return [text]

    # 长的优先，避免较短特殊 token 抢先匹配其前缀
    ordered = sorted(special_tokens, key=len, reverse=True)

    pattern = "(" + "|".join(re.escape(token) for token in ordered) + ")"

    return [part for part in re.split(pattern, text) if part != ""]


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    def encode(self, text):
        # 调用现有 encode 函数，传入本实例保存的规则和特殊 token
        return encode(text, self.merges, self.special_tokens)

    def decode(self, token_ids):
        # 调用现有 decode 函数，传入本实例保存的词表
        return decode(token_ids, self.vocab)

    def __len__(self):
        return len(self.vocab)


def train_tokenizer_from_file(input_path, num_merges, special_tokens=None):
    text = Path(input_path).read_text(encoding="utf-8")

    vocab, merges, _ = train_bpe(
        text,
        num_merges=num_merges,
        special_tokens=special_tokens,
    )

    return BPETokenizer(vocab, merges, special_tokens)


def main():
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "review_corpus.txt"

    tokenizer = train_tokenizer_from_file(
        corpus_path,
        num_merges=50,
        special_tokens=["<|endoftext|>"],
    )

    text = "我喜欢学习CUDA。<|endoftext|>"
    ids = tokenizer.encode(text)

    print("词表大小:", len(tokenizer))
    print("编码结果:", ids)
    print("各 token 内容:", [tokenizer.vocab[i] for i in ids])
    print("还原文本:", tokenizer.decode(ids))

    assert tokenizer.decode(ids) == text
    # # 1. 验证预分词边界与训练结果
    # print("=== 训练：ab ab! ===")

    # train_text = "ab ab!"
    # vocab, merges, sequences = train_bpe(train_text, num_merges=10)

    # assert sequences == [[256], [257], [33]]
    # assert vocab[256] == b"ab"
    # assert vocab[257] == b" ab"

    # # 同一文本的编码应等于训练后各片段按顺序拼接
    # expected = [token_id for sequence in sequences for token_id in sequence]
    # assert encode(train_text, merges) == expected

    # # 2. 验证编码和解码
    # print("\n=== 编码与解码 ===")

    # for text in ["ab ab!", "ab!", "你好", "ab你好"]:
    #     ids = encode(text, merges)
    #     restored = decode(ids, vocab)

    #     print("原文:", repr(text))
    #     print("实际编码:", ids)
    #     print("各 token 的字节:", [vocab[i] for i in ids])
    #     print("还原文本:", repr(restored))
    #     print()

    #     assert restored == text

    # # 3. 验证频率相同时的选择规则
    # print("=== 频率相同的情况：abac ===")

    # tie_vocab, tie_merges, tie_sequences = train_bpe("abac", num_merges=1)

    # assert tie_merges == [(98, 97)]
    # assert tie_vocab[256] == b"ba"
    # assert tie_sequences == [[97, 256, 99]]

    # special_tokens = ["<|endoftext|>"]

    # text = "ab<|endoftext|>ab"

    # parts = split_special_tokens(text, special_tokens)

    # print("切分结果:", parts)

    # assert parts == [
    #     "ab",
    #     "<|endoftext|>",
    #     "ab",
    # ]
    # assert "".join(parts) == text

    # for part in parts:
    #     if part in special_tokens:
    #         print("特殊 token:", repr(part))
    #     else:
    #         print("普通片段:", pre_tokenize(part))

    #     print("\n=== 特殊 token 检查 ===")

    #     special_tokens = ["<|endoftext|>"]
    #     text = "ab<|endoftext|>ab"

    #     vocab, merges, _ = train_bpe(
    #         text,
    #         num_merges=3,
    #         special_tokens=special_tokens,
    #     )

    #     ids = encode(
    #         text,
    #         merges,
    #         special_tokens=special_tokens,
    #     )

    #     print("实际编码:", ids)
    #     print("各 token 内容:", [vocab[i] for i in ids])
    #     print("还原文本:", decode(ids, vocab))

    #     assert vocab[256] == b"<|endoftext|>"
    #     assert vocab[257] == b"ab"
    #     assert ids == [257, 256, 257]
    #     assert decode(ids, vocab) == text

    #     special_tokens = ["<|endoftext|>"]

    #     vocab, merges, _ = train_bpe(
    #         "ab<|endoftext|>ab",
    #         num_merges=3,
    #         special_tokens=special_tokens,
    #     )

    #     tokenizer = BPETokenizer(vocab, merges, special_tokens)

    #     text = "ab你好<|endoftext|>"
    #     ids = tokenizer.encode(text)
    #     restored = tokenizer.decode(ids)

    #     print("词表大小:", len(tokenizer))
    #     print("编码结果:", ids)
    #     print("还原文本:", restored)

    #     assert restored == text

    # print("\n全部检查通过")


if __name__ == "__main__":
    main()
