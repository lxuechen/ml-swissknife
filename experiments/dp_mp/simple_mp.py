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


def fine_tune_gpt2_with_dp_on_mock_data(
    mp=True, base_device=0, batch_size=4, num_updates=3,
    disable_randomness=True, model_name_or_path='gpt2',
):
    # `disable_randomness` removes all noise added during training; makes testing easy.
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    config.output_hidden_states = True
    if disable_randomness:
        config.attn_pdrop = config.embd_pdrop = config.resid_pdrop = 0.
    model: transformers.GPT2LMHeadModel = transformers.GPT2LMHeadModel.from_pretrained(
        model_name_or_path, config=config
    )

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
    if disable_randomness:
        privacy_engine.noise_multiplier = 0.  # Only for testing purposes.

    model.train()
    for update_idx in range(num_updates):
        input_ids = get_batch(batch_size=batch_size, disable_randomness=disable_randomness)
        outputs = model(input_ids, return_dict=True)
        labels = input_ids[:, 1:, ]
        logits = outputs.logits[:, :-1, :].permute(0, 2, 1)
        # `loss` is a 1-D tensor of shape (batch_size,).
        loss = F.cross_entropy(logits, labels, reduction="none").mean(dim=1)
        # This step is different from existing workflows:
        #   Don't call `loss.backward`; leave it to `optimizer.step` to handle backward.
        optimizer.step(loss=loss)

        # Check if the devices are correct.
        print(f'mp: {mp}, update_idx: {update_idx}')
        print(f'  devices of hidden_states: {tuple(str(h.device) for h in outputs.hidden_states)}')
        del input_ids, outputs, labels, logits

    model.cpu()
    del optimizer, privacy_engine
    torch.cuda.empty_cache()
    return model


def main(model_name_or_path='gpt2'):
    torch.set_default_dtype(torch.float64)

    pretrained_model: transformers.GPT2LMHeadModel = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
    mp_final_model = fine_tune_gpt2_with_dp_on_mock_data(
        mp=True, disable_randomness=True, model_name_or_path=model_name_or_path
    )
    final_model = fine_tune_gpt2_with_dp_on_mock_data(
        mp=False, disable_randomness=True, model_name_or_path=model_name_or_path
    )

    with torch.no_grad():
        pretrained_model_flat = torch.cat(tuple(t.flatten() for t in pretrained_model.parameters()))
        mp_final_model_flat = torch.cat(tuple(t.flatten() for t in mp_final_model.parameters()))
        final_model_flat = torch.cat(tuple(t.flatten() for t in final_model.parameters()))

        print(f'2-norm of param. difference between MP and single-GPU fine-tuned models:'
              f'{torch.norm(mp_final_model_flat - final_model_flat):.4f}')
        print(f'2-norm of param. difference between pretrained and single-GPU fine-tuned models: '
              f'{torch.norm(pretrained_model_flat - final_model_flat):.4f}')


if __name__ == "__main__":
    fire.Fire(main)
