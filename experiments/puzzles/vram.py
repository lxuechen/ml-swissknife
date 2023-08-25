import fire

from ml_swissknife import utils

configs = {
    # 13b
    "llama-13b": {
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
    },
    # 33b
    "llama-33b": {
        "hidden_size": 6656,
        "num_layers": 60,
        "num_heads": 52,
    },
}

state_memory_in_gb = {
    # 13b
    "llama-13b": 104,
    # 33b
    "llama-33b": 264,
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
    """Compute the activation memory needed for standard Transformer architecture in GB.

    Note: llama involves rotary position embeddings, thus the usage would be different.
    """
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


def plot():
    batch_size = 1
    num_shards = 8
    seq_lens = [1024, 2048, 4096, 8192, 16384]

    no = []
    tp = []
    tp_sp = []

    def config_to_plot(config, img_path):
        for seq_len in seq_lens:
            no.append(
                get_activation_memory(
                    batch_size=batch_size,
                    num_shards=num_shards,
                    seq_len=seq_len,
                    tensor_parallel=False, sequence_parallel=False,
                    **config,
                )
            )
            tp.append(
                get_activation_memory(
                    batch_size=batch_size,
                    num_shards=num_shards,
                    seq_len=seq_len,
                    tensor_parallel=True, sequence_parallel=False,
                    **config,
                )
            )
            tp_sp.append(
                get_activation_memory(
                    batch_size=batch_size,
                    num_shards=num_shards,
                    seq_len=seq_len,
                    tensor_parallel=True, sequence_parallel=True,
                    **config,
                )
            )

        utils.plot(
            img_path=img_path,
            plots=[dict(x=seq_lens, y=no, label="vanilla"),
                   dict(x=seq_lens, y=tp, label="tensor parallel"),
                   dict(x=seq_lens, y=tp_sp, label="tensor parallel + sequence parallel")],
        )

    # 13b
    config_to_plot(configs['llama-13b'], img_path="./llama-13b.pdf")
    config_to_plot(configs['llama-33b'], img_path="./llama-33b.pdf")


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
