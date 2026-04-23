from math import floor

def transformer_accounting(
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
    print(f'\t\t{attn_params / total_params * 100:2.0f}% attention')
    print(f'\t\t{ffn_params / total_params * 100:2.0f}% mlp')
    print(f'\t\t{emb_params / total_params * 100:2.0f}% embeddings')
    print(f'\t\t{norm_params / total_params * 100:2.0f}% norms')

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

def adamw_accounting(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    param_size: int = 4
) -> (int, int, int):
    total_params, _, total_flops = transformer_accounting(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        1
    )

    # for each parameter, need to record first and second moments
    optimizer_state = 2 * total_params

    # 1 gradient for every parameter
    gradients = total_params

    # activations
    activs = 0

    # qkvo-projections
    activs += 4 * batch_size * context_length * d_model
    # qkT multiply
    activs += 2 * batch_size * num_heads * context_length ** 2
    # weighted sum
    activs += batch_size * context_length * d_model

    # ffn
    activs += 2 * batch_size * context_length * d_ff + \
        batch_size * context_length * d_model
    # silu and elementwise product
    activs += 2 * batch_size * context_length * d_ff
    # ffn rmsnorms
    activs += 2 * batch_size * context_length * d_model

    activs = activs * num_layers

    # final norm
    activs += batch_size * context_length * d_model

    # output embedding
    activs += batch_size * context_length * vocab_size

    # cross-entropy
    activs += batch_size * context_length * vocab_size

    batch_dep = activs * param_size
    batch_indep = (total_params + gradients + optimizer_state) * param_size

    return (batch_dep, batch_indep, total_flops)


if __name__ == '__main__':
    vocab_size = 50257
    context_length = 1024

    num_layers = 48
    d_model = 1600
    num_heads = 25

    d_ff = 4288
    param_size = 4

    x,y,f = adamw_accounting(
        1024,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff
    )

    # print('\t\nAdamW memory usage =',
    #     f'({x/1e9:.2f} * batch_size + {y/1e9:.2f}) GB')
    # print('\tMaximum batch size for 80 GB memory =',
    #     f'{floor((80 - (y/1e9)) / (x/1e9))}')
    print('\tTime to train GPT-2 XL for ',
        '400K steps, 1024 batch on 1 H100 with 50% MFU = '
        f'{400e3 * 3*f / (.5 * 495e12) / 3600:.2f} hours')

