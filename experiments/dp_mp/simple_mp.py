"""Simple model parallelism demonstration.

TODO: Get DP-SGD running on mock data.
TODO: Test against single GPU updates.
"""

import fire
import private_transformers
import torch.cuda
import torch.nn.functional as F
import transformers


def get_device_info(base_device_index=0):
    num_devices = torch.cuda.device_count()
    devices = list(range(num_devices))
    base_device = devices[base_device_index]
    return num_devices, devices, base_device


def get_base_device(base_device_index=0):
    return get_device_info(base_device_index=base_device_index)[2]


def get_batch(batch_size=16, seq_len=100):
    """Create a mock batch of data."""
    x = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    x = x.to(get_base_device())
    return x


def main(batch_size=16):
    model: transformers.GPT2LMHeadModel = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
    model.parallelize()
    optimizer = torch.optim.Adam(params=model.parameters())

    privacy_engine = private_transformers.PrivacyEngine(
        module=model,
        target_delta=1e-5,
        target_epsilon=3,
        batch_size=batch_size,
        sample_size=int(1e5),
        max_grad_norm=0.1,
        epochs=3,
    )
    privacy_engine.attach(optimizer=optimizer)
    print(privacy_engine)

    model.train()
    for _ in range(3):
        input_ids = get_batch()
        outputs = model(input_ids, return_dict=True)
        labels = input_ids[:, 1:, ]
        logits = outputs.logits[:, :-1, :].permute(0, 2, 1)
        # `loss` is a 1-D tensor of shape (batch_size,).
        loss = F.cross_entropy(logits, labels, reduction="none").mean(dim=1)
        # This step is different from existing workflows:
        #   Don't call `loss.backward`; leave it to `optimizer.step` to handle backward.
        optimizer.step(loss=loss)


if __name__ == "__main__":
    fire.Fire(main)
