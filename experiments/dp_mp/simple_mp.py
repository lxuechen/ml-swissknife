"""Simple model parallelism demonstration."""

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


def get_batch(batch_size=4, seq_len=100, disable_randomness=True):
    """Create a mock batch of data.

    Always generate the same batch when `disable_randomness` is True.
    """
    generator = torch.Generator()
    if disable_randomness:
        generator.manual_seed(0)
    x = torch.randint(size=(batch_size, seq_len), low=1, high=100, generator=generator)
    x = x.to(get_base_device())
    return x


def train_gpt2_with_dp_on_mock_data(
    model=None, mp=True, base_device=0, batch_size=4, num_updates=3,
    disable_randomness=True,
):
    if model is None:
        model: transformers.GPT2LMHeadModel = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
    if disable_randomness:
        model.config.attn_pdrop = 0.
        model.config.resid_pdrop = 0.
        model.config.embd_pdrop = 0.

    if mp:
        model.parallelize()  # This essentially does everything for model parallelism.
    else:
        model.to(base_device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

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

    model.train()
    for _ in range(num_updates):
        input_ids = get_batch(batch_size=batch_size, disable_randomness=disable_randomness)
        outputs = model(input_ids, return_dict=True)
        labels = input_ids[:, 1:, ]
        logits = outputs.logits[:, :-1, :].permute(0, 2, 1)
        # `loss` is a 1-D tensor of shape (batch_size,).
        loss = F.cross_entropy(logits, labels, reduction="none").mean(dim=1)
        # This step is different from existing workflows:
        #   Don't call `loss.backward`; leave it to `optimizer.step` to handle backward.
        optimizer.step(loss=loss)
        del input_ids, outputs, labels, logits

    model.cpu()
    del optimizer, privacy_engine
    torch.cuda.empty_cache()
    return model


def main():
    torch.set_default_dtype(torch.float64)

    mp_final_model = train_gpt2_with_dp_on_mock_data(mp=True, disable_randomness=True)
    final_model = train_gpt2_with_dp_on_mock_data(mp=False, disable_randomness=True)

    mp_final_model_flat = torch.cat(
        tuple(t.flatten() for t in mp_final_model.parameters())
    )
    final_model_flat = torch.cat(
        tuple(t.flatten() for t in final_model.parameters())
    )
    print(torch.norm(mp_final_model_flat - final_model_flat))


if __name__ == "__main__":
    fire.Fire(main)
