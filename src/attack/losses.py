import torch
import torch.nn.functional as F

def get_uniform_target(
    mu: float,
    sigma: float,
    valid_tokens: dict,
    vocab_size: int,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Creates a dense uniform probability distribution over the vocab size.
    Only valid number tokens within mu +/- sqrt(3)*sigma have non-zero probability.
    """
    p_uniform = torch.zeros(vocab_size, device=device, dtype=dtype)
    
    import math
    bound = math.sqrt(3) * sigma
    low = mu - bound
    high = mu + bound
    
    count = 0
    for num, token_id in valid_tokens.items():
        if low <= num <= high:
            p_uniform[token_id] = 1.0
            count += 1
            
    if count > 0:
        p_uniform = p_uniform / count
    else:
        # Fallback if empty
        for _, token_id in valid_tokens.items():
            p_uniform[token_id] = 1.0 / len(valid_tokens)
            
    return p_uniform

def compute_distributional_loss(logits: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:
    """
    Computes KL(p_target || q) where q = softmax(logits).
    logits: (batch_size, vocab_size) or (1, vocab_size)
    p_target: (batch_size, vocab_size) or (1, vocab_size) dense probabilities
    """
    # PyTorch kl_div expects log_target=False by default for the target, and log_softmax for the input.
    # The signature is kl_div(input, target) where input is log-probs and target is probs.
    # reduction="batchmean" is required for mathematical KL divergence.
    log_q = F.log_softmax(logits, dim=-1)
    loss = F.kl_div(log_q, p_target, reduction="batchmean")
    return loss


def compute_sampled_attack_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute sampled-token SFT loss for next-number attack logits."""
    return F.cross_entropy(logits, labels)


def compute_mixed_attack_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    p_target: torch.Tensor,
    sampled_weight: float = 0.5,
    distributional_weight: float = 0.5,
) -> torch.Tensor:
    """Compute the PRD mixed attack: weighted sampled SFT plus full KL."""
    sampled = compute_sampled_attack_loss(logits, labels)
    distributional = compute_distributional_loss(logits, p_target)
    return sampled_weight * sampled + distributional_weight * distributional


def next_token_logits_from_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Select logits at the position that predicts the first unmasked target token.

    Hugging Face causal LM losses shift labels internally. If a target label is at
    position t, the next-token distribution that predicts it is logits[t - 1].
    """
    if logits.ndim != 3:
        raise ValueError("Expected logits with shape (batch, sequence, vocab).")
    if labels.ndim != 2:
        raise ValueError("Expected labels with shape (batch, sequence).")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logits and labels batch/sequence dimensions must match.")

    target_positions = (labels != -100).long().argmax(dim=1)
    if torch.any(target_positions <= 0):
        raise ValueError("Each row must have a target label after at least one context token.")
    batch_idx = torch.arange(labels.shape[0], device=labels.device)
    return logits[batch_idx, target_positions - 1, :]


def target_labels_from_masked_labels(labels: torch.Tensor) -> torch.Tensor:
    """Return the first non-masked target label from each row."""
    target_positions = (labels != -100).long().argmax(dim=1)
    batch_idx = torch.arange(labels.shape[0], device=labels.device)
    targets = labels[batch_idx, target_positions]
    if torch.any(targets == -100):
        raise ValueError("Each row must contain at least one target label.")
    return targets
