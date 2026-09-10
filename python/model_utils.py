def estimate_memory(layers, memory_per_layer):
    return layers * memory_per_layer


def print_model(name):
    print("Model:", name)


def tokens_per_second(total_tokens,seconds):
    print(total_tokens / seconds)

if __name__ == "__main__":
    print("model_utils is running")