import torch
import torch.nn as nn

torch.manual_seed(0)

vocab_size = 10
d_model = 4

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model,
)

token_ids = torch.tensor([
    [2, 5, 2],
    [1, 5, 7],
])

x = embedding(token_ids)

print("token_ids shape:", token_ids.shape)
print("embedding weight shape:", embedding.weight.shape)
print("output shape:", x.shape)

print("\ntoken_ids:")
print(token_ids)

print("\noutput:")
print(x)