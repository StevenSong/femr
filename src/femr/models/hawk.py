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


def _scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes h[t] = a[t] * h[t - 1] + x[t] along dim 0 with a log-depth (Hillis-Steele) scan."""
    h = x
    n = h.shape[0]
    offset = 1
    while offset < n:
        # a[t] is the product of the original a over the window that h[t] currently covers
        h = torch.cat((h[:offset], torch.addcmul(h[offset:], a[offset:], h[:-offset])))
        a = torch.cat((a[:offset], a[offset:] * a[:-offset]))
        offset *= 2
    return h


class LinearRecurrenceFunction(torch.autograd.Function):
    # A custom backward so autograd doesn't save every intermediate of the scan, only a and the output

    @staticmethod
    def forward(ctx, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        o = _scan(a.float(), x.float())
        ctx.save_for_backward(a, o)
        ctx.x_dtype = x.dtype
        return o.to(x.dtype)

    @staticmethod
    def backward(ctx, d_o: torch.Tensor):
        a, o = ctx.saved_tensors
        a_f = a.float()

        # d_x[t] = d_o[t] + a[t + 1] * d_x[t + 1], which is the same recurrence run in reverse
        a_next = torch.cat((a_f[1:], torch.zeros_like(a_f[:1])))
        d_x = _scan(a_next.flip(0), d_o.float().flip(0)).flip(0)
        d_a = torch.cat((torch.zeros_like(o[:1]), o[:-1])) * d_x

        return d_a.to(a.dtype), d_x.to(ctx.x_dtype)


def linear_recurrence_torch(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes h[t] = a[t] * h[t - 1] + x[t] along dim 0."""
    return LinearRecurrenceFunction.apply(a, x)


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
