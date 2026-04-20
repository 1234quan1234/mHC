import torch
import torch.nn as nn
import torch.nn.functional as F

class RealCausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # QKV Projection gộp chung để tăng tốc băng thông
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # [batch_size, seq_len, 3 * d_model]
        qkv = self.Wqkv(x)
        
        # Reshape & Permute: [3, batch_size, num_heads, seq_len, head_dim]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Gọi FlashAttention / SDPA với is_causal=True (Chặn nhìn trước tương lai)
        # Thực thi trực tiếp bằng kernel C++ của PyTorch trên GPU
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Trả lại kích thước [batch_size, seq_len, d_model]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)