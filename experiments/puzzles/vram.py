import fire

configs = {
    # 13b
    "llama-13b-2k": {
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "seq_len": 2048,
    },
    "llama-13b-4k": {
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "seq_len": 4096,
    },
    "llama-13b-8k": {
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "seq_len": 8192,
    },
    "llama-13b-16k": {
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "seq_len": 16384,
    },
    # 33b
    "llama-33b-2k": {
        "hidden_size": 6656,
        "num_layers": 60,
        "num_heads": 52,
        "seq_len": 2048,
    },
    "llama-33b-4k": {
        "hidden_size": 6656,
        "num_layers": 60,
        "num_heads": 52,
        "seq_len": 4096,
    },
    "llama-33b-8k": {
        "hidden_size": 6656,
        "num_layers": 60,
        "num_heads": 52,
        "seq_len": 8192,
    },
    "llama-33b-16k": {
        "hidden_size": 6656,
        "num_layers": 60,
        "num_heads": 52,
        "seq_len": 16384,
    }
}

state_mem = {
    # 13b
    "llama-13b-2k": 104,
    "llama-13b-4k": 104,
    "llama-13b-8k": 104,
    "llama-13b-16k": 104,
    # 33b
    "llama-33b-2k": 264,
    "llama-33b-4k": 264,
    "llama-33b-8k": 264,
    "llama-33b-16k": 264,
}


def get_activation_memory(
    batch_size,
    hidden_size,
    seq_len,
    num_heads,
    tensor_parallel: bool,
    sequence_parallel: bool,
    num_shards=1,
    num_layers=1,  # Transformer layers / blocks.
) -> float:
    """Compute the activation memory needed for standard Transformer architecture in GB."""
    # TODO: Llama involves rotary position embeddings, thus the usage would be different.
    if tensor_parallel and sequence_parallel:
        bytes_ = seq_len * batch_size * hidden_size / num_shards * (34 + 5 * num_heads * seq_len / hidden_size)
    elif tensor_parallel:
        bytes_ = seq_len * batch_size * hidden_size * (
            10 + 24 / num_shards + 5 * num_heads * seq_len / hidden_size / num_shards
        )
    elif sequence_parallel:
        bytes_ = seq_len * batch_size * hidden_size * (10 / num_shards + 24 + 5 * num_heads * seq_len / hidden_size)
    else:
        bytes_ = seq_len * batch_size * hidden_size * (34 + 5 * num_heads * seq_len / hidden_size)
    return bytes_ / 1024 ** 3 * num_layers


def basic():
    # TP brings massive gains (>70%); SP brings minor gains (10%).
    for tensor_parallel in (False, True):
        for sequence_parallel in (False, True):
            gbs = get_activation_memory(
                hidden_size=6656,
                batch_size=1,
                seq_len=2048,
                num_heads=52,
                num_shards=8,
                num_layers=60,
                tensor_parallel=tensor_parallel,
                sequence_parallel=sequence_parallel,
            )
            print(
                f'tensor_parallel: {tensor_parallel}, '
                f'sequence_parallel: {sequence_parallel}, '
                f'per_device_train_batch_size=1, llama config (but standard architecture), '
                f'mem spending per device: {gbs}'
            )


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
