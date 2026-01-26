import torch
import torch.nn as nn

class LowRankLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0))
        self.bias = None

    def forward(self, x):
        # If ever gets executed, fall back to a dense linear if shapes exist
        if self.weight.numel() == 0:
            raise RuntimeError("Stub LowRankLinear executed. This stub is only for unpickling.")
        return torch.nn.functional.linear(x, self.weight, self.bias)

    # Make unpickling tolerant to different pickle formats
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            pass
