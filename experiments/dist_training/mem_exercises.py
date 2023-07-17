def naive_activation_memory(
    batch_size,
    hidden_size,
    seq_len,
    num_heads,
    tensor_parallel: bool,
    sequence_parallel: bool,
    num_shards=1,
    num_layers=1,  # Transformer layers / blocks.
) -> float:
    """Compute the activation memory needed for standard Transformer architecture in GB.

    """
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


if __name__ == "__main__":
    # TP brings massive gains (>70%); SP brings minor gains (10%).
    for tensor_parallel in (False, True):
        for sequence_parallel in (False, True):
            gbs = naive_activation_memory(
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
