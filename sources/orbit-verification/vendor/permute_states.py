import torch

def permute_states(hidden_states: torch.Tensor, perm_type: str) -> torch.Tensor:
    """
    Applies a permutation to the hidden states tensor based on the specified permutation type.

    Possible permutation types are:
        * 'None' -- No permutation applied.
        * 'S'    -- Sequence-level permutation (shuffle rows).
        * 'D'    -- Per-token hidden dimension permutation (shuffle columns).
        * 'SD'   -- Combined sequence and hidden dimension permutation (apply 'D' then 'S').

    Args:
        hidden_states (torch.Tensor): Input tensor of shape (N, d), where N is the number of tokens
                                      and d is the hidden dimension.
        perm_type (str): Type of permutation to apply.

    Returns:
        torch.Tensor: Permuted tensor of shape (N, d).

    Raises:
        Exception: If the permutation type is not supported.
    """
    N, d = hidden_states.size()
    device = hidden_states.device
    if perm_type == "None":
        return hidden_states
    elif perm_type == "S":
        return hidden_states[torch.randperm(N, device=device)]
    elif perm_type == "D":
        return hidden_states[:, torch.randperm(d, device=device)]
    elif perm_type == "SD":
        return permute_states(permute_states(hidden_states, "D"), "S")
    else:
        raise Exception(f"Unsupported permutation pattern {perm_type}")
