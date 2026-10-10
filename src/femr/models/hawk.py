import functools

import torch
import torch_hawk

# torch_hawk's CUDA kernels are only compiled for sm_80+, and it doesn't check kernel launch errors,
# so on older GPUs (e.g. T4, sm_75) its ops silently return uninitialized memory.
# These pure PyTorch versions match torch_hawk's CPU reference implementations.

_original_conv1d = torch_hawk.conv1d
_original_linear_recurrence = torch_hawk.linear_recurrence


@functools.lru_cache
def _has_hawk_kernels(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) >= (8, 0)


def _needs_fallback(t: torch.Tensor) -> bool:
    return t.is_cuda and not _has_hawk_kernels(t.device)


def conv1d_torch(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Causal depthwise conv over x [n, k] with w [k, width], not mixing tokens with different s."""
    width = w.shape[1]
    x_f = x.float()
    w_f = w.float()

    result = x_f * w_f[:, width - 1]
    for shift in range(1, width):
        same_segment = (s[shift:] == s[:-shift]).unsqueeze(-1)
        result[shift:] += torch.where(same_segment, x_f[:-shift] * w_f[:, width - 1 - shift], 0)

    return result.to(x.dtype)


def linear_recurrence_torch(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes h[t] = a[t] * h[t - 1] + x[t] along dim 0 with a log-depth (Hillis-Steele) scan."""
    a_f = a.float()
    h = x.float()

    n = h.shape[0]
    offset = 1
    while offset < n:
        # a_f[t] is the product of a over the window that h[t] currently covers
        h = torch.cat((h[:offset], torch.addcmul(h[offset:], a_f[offset:], h[:-offset])))
        a_f = torch.cat((a_f[:offset], a_f[offset:] * a_f[:-offset]))
        offset *= 2

    return h.to(x.dtype)


def conv1d(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    if _needs_fallback(x):
        return conv1d_torch(x, w, s)
    return _original_conv1d(x, w, s)


def linear_recurrence(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if _needs_fallback(x):
        return linear_recurrence_torch(a, x)
    return _original_linear_recurrence(a, x)


# torch_hawk.RecurrentBlock looks these up on the module at call time
torch_hawk.conv1d = conv1d
torch_hawk.linear_recurrence = linear_recurrence
