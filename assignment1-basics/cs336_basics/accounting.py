def accounting(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    param_size: int
) -> tuple[int, int, int]:
    # --- parameters
    emb_params = 0
    # input and output embeddings
    emb_params += 2 * vocab_size * d_model

    attn_params = 0
    # qkvo-projections
    attn_params += 4*(num_heads * (d_model // num_heads) * d_model)
    attn_params = attn_params * num_layers

    norm_params = 0
    # ffn norms
    norm_params += 2 * d_model * num_layers
    # final norm
    norm_params += d_model

    ffn_params = 0
    ffn_params += 3 * d_ff * d_model * num_layers

    total_params = emb_params + attn_params + norm_params + ffn_params
    total_size = total_params * param_size

    print(f'\tTotal parameters: {total_params / 1e9:.2f} B')
    print(f'\tTotal memory: {total_size / 1e9:.2f} GB')

    # --- flops
    attn_flops = 0
    # qkvo-projections
    attn_flops += 4 * 2 * context_length * d_model * d_model
    # attn contractions
    attn_flops += 2 * 2*(d_model // num_heads) * \
            context_length * context_length * num_heads
    attn_flops = attn_flops * num_layers

    ffn_flops = 0
    ffn_flops += 3 * 2 * context_length * d_model * d_ff
    ffn_flops = ffn_flops * num_layers

    head_flops = 2 * context_length * d_model * vocab_size

    total_flops = attn_flops + ffn_flops + head_flops

    print(f'\tTotal FLOPs: {total_flops / 1e12:.2f} TFLOPs')
    print(f'\t\t{attn_flops / total_flops * 100:2.0f}% attention')
    print(f'\t\t{ffn_flops / total_flops * 100:2.0f}% mlp')
    print(f'\t\t{head_flops / total_flops * 100:2.0f}% head')

    return (total_params, total_size, total_flops)

if __name__ == '__main__':
    vocab_size = 50257
    context_length = 16384

    num_layers = 48
    d_model = 1600
    num_heads = 25

    d_ff = 4288
    param_size = 4
    accounting(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        param_size
    )

