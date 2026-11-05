import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor


def ce_loss(
    output: Tensor, 
    target: LongTensor,
    mask: Tensor,
    ignore_index: int
) -> torch.float:
    r"""Cross entropy loss.

    b: batch_size
    l: seq_len
    v: vocab_size

    Args:
        output: (b, l, v)
        target: (b, l)
        mask: (b, l)
        ignore_index: int

    Outputs:
        loss: torch.float
    """

    loss = F.cross_entropy(
        # the tokens that are predicted by the model
        input=rearrange(output, 'b l v -> (b l) v'),
        # the tokens that are the correct answer
        target=rearrange(target, 'b l -> (b l)'),
        # the mask that indicates which tokens should be ignored in the loss calculation
        ignore_index=ignore_index,
        # returns the loss for each element in the batch without reducing it to a single value
        reduction="none"
    )  # (b*l,)

    loss = loss * rearrange(mask, 'b l -> (b l)')  # (b*l,)
    loss = torch.mean(loss)

    return loss