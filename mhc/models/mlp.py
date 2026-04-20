import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        # Kích thước ẩn theo tỷ lệ chuẩn của LLaMA: (8/3) * d_model
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
            # Làm tròn đến bội số của 128 để tối ưu trên Tensor Cores
            hidden_dim = 128 * ((hidden_dim + 127) // 128)
            
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(xW1) * xW3)W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))