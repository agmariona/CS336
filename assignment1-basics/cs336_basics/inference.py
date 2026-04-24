from pathlib import Path
import torch
from einops import rearrange

from .training_utils import load_gen_bundle_from_checkpoint
from .nn_utils import softmax

def decode(
    checkpoint_path: str | Path,
    prompt: str,
    max_output_len: int,
    temperature: float,
    top_p: float,
    device: str | torch.device = 'mps'
) -> str:
    model, tokenizer = load_gen_bundle_from_checkpoint(
        checkpoint_path,
        load_device=device
    )

    if temperature <= 0:
        raise ValueError(
            f'Temperature {temperature} must be positive.'
        )
    if top_p <= 0 or top_p > 1:
        raise ValueError(
            f'Top-p parameter {top_p} must be in (0,1]'
        )

    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) > model.context_length:
        raise ValueError(
            f'Prompt length {len(prompt_tokens)} too long for model with '
            f'context length {model.context_length}'
        )

    eot_token = tokenizer.encode('<|endoftext|>')[0]

    output_tokens = []
    while len(output_tokens) < max_output_len:
        seq = prompt_tokens + output_tokens
        if len(seq) > model.context_length:
            seq = seq[-model.context_length:]
        seq = torch.tensor(seq, device=device, dtype=torch.long)
        seq = rearrange(seq, 'seq_len -> 1 seq_len')

        with torch.no_grad():
            logits = model(seq)

        last_logits = logits[0, -1, :]
        next_dist = softmax(last_logits / temperature, -1)

        # top-p sampling
        sort_dist, sort_idx = torch.sort(next_dist, descending=True)
        cum_probs = torch.cumsum(sort_dist, dim=-1)
        cutoff = torch.searchsorted(cum_probs, top_p, side="left").item()

        sample_dist = sort_dist[:(cutoff+1)]
        sample_ids = sort_idx[:(cutoff+1)]

        sample_dist = sample_dist / sample_dist.sum(dim=-1, keepdim=True)
        sampled_idx = torch.multinomial(sample_dist, num_samples=1).item()
        next_token = sample_ids[sampled_idx].item()

        if next_token == eot_token:
            break

        output_tokens.append(next_token)

    return tokenizer.decode(prompt_tokens + output_tokens)

def main():
    checkpoint_path = 'checkpoints/latest.pt'
    prompt = "Once upon a time"
    max_output_len = 128
    temperature = 1
    top_p = 0.9

    output = decode(
        checkpoint_path = checkpoint_path,
        prompt = prompt,
        max_output_len = max_output_len,
        temperature = temperature,
        top_p = top_p
    )

    print(f'Prompt: {prompt}\n')
    print(f'Inference: {output}')

if __name__ == '__main__':
    main()
