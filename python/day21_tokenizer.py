from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)

texts = [
    "CUDA",
    "我喜欢学习CUDA",
    "我喜欢学习CUDA，因为GPU很适合并行计算"
]

batch = tokenizer(
    texts,
    padding=True,
    return_tensors="pt"
)

labels = batch["input_ids"].clone()

labels[batch["attention_mask"] == 0] = -100

print("input_ids:")
print(batch["input_ids"])

print("attention_mask:")
print(batch["attention_mask"])

print("labels:")
print(labels)

# text = "我喜欢学习CUDA，因为GPU很适合并行计算"

# tokens = tokenizer.tokenize(text)
# token_ids = tokenizer.encode(text)

# print("原始文本:", text)
# print("Tokens:", tokens)
# print("Token IDs:", token_ids)

# decoded_text = tokenizer.decode(token_ids)
# print("Decode:", decoded_text)