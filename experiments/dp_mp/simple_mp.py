"""Simple model parallelism demonstration.

TODO: Get DP-SGD running on mock data.
TODO: Test against single GPU updates.
"""
import types

import fire
import torch.cuda
import transformers


# TODO: This is a simple test hook.
def forward_hook(module, inputs, outputs):
    pass


def get_device_info(base_device_index=0):
    num_devices = torch.cuda.device_count()
    devices = list(range(num_devices))
    base_device = devices[base_device_index]
    return num_devices, devices, base_device


def get_base_device(base_device_index=0):
    return get_device_info(base_device_index=base_device_index)[2]


def mp_to_device(model: transformers.GPT2LMHeadModel):
    num_devices, devices, base_device = get_device_info()
    model.to(base_device)  # Embedding and lm-head should be on device:0.

    # Split the different blocks at various places.
    transformer_blocks = model.transformer.h
    num_blocks = len(transformer_blocks)
    assert num_blocks % num_devices == 0

    blocks_per_device = num_blocks // num_devices
    for i, transformer_block in enumerate(transformer_blocks):
        # This deals with non-even-divisions.
        target_device = min(devices[-1], i // blocks_per_device)
        transformer_block.to(target_device)
        replace_block_forward(transformer_block)


def transfer_tensors(tensors, target):
    # `target` could be device or dtype.
    new_tensors = tuple(
        tensor.to(target) if tensor is not None else tensor for tensor in tensors
    )
    return new_tensors


def replace_block_forward(block):
    """Replace the `forward` function for GPT2Block to include explicit device transfer.

    TODO: Check how megatron does this.
    """

    def new_forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        device = next(iter(self.parameters())).device
        tensors = (
            hidden_states, layer_past, attention_mask,
            head_mask, encoder_hidden_states, encoder_attention_mask,
        )
        new_tensors = transfer_tensors(tensors=tensors, target=device)
        (hidden_states, layer_past, attention_mask,
         head_mask, encoder_hidden_states, encoder_attention_mask,) = new_tensors

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

    block.forward = types.MethodType(new_forward, block)


def main():
    model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
    mp_to_device(model)

    # Create mock data.
    batch_size, seq_len = 16, 100
    x = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    x = x.to(get_base_device())
    outs = model(x)
    print(outs)


if __name__ == "__main__":
    fire.Fire(main)
