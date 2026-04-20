import torch
import torch.nn as nn
from mhc.kernels.sinkhorn_knopp import SinkhornKnopp
from mhc.kernels.fusion import apply_fused_residual_merge
class RMSNorm(nn.Module):
    """Chuẩn hóa RMSNorm cơ bản."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class mHCLayer(nn.Module):
    def __init__(self, d_model: int, n_dim: int, layer_func: nn.Module):
        """
        Khởi tạo tầng Manifold-Constrained Hyper-Connections (mHC).
        
        Args:
            d_model (int): Chiều C của đặc trưng (ví dụ: 1280, 2560).
            n_dim (int): Hệ số mở rộng luồng thặng dư n (mặc định: 4).
            layer_func (nn.Module): Hàm F của tầng (Attention hoặc FFN/MLP).
        """
        super().__init__()
        self.C = d_model
        self.n = n_dim
        self.layer_func = layer_func
        
        # Flattened dimension (n * C)
        self.nC = self.n * self.C
        
        # RMSNorm cho vector flattened
        self.rmsnorm = RMSNorm(self.nC)
        
        # --- Dynamic Mappings (Linear Projections phi) ---
        self.phi_pre = nn.Linear(self.nC, self.n, bias=False)
        self.phi_post = nn.Linear(self.nC, self.n, bias=False)
        self.phi_res = nn.Linear(self.nC, self.n * self.n, bias=False)
        
        # --- Static Mappings (Biases b) ---
        self.b_pre = nn.Parameter(torch.zeros(1, self.n))
        self.b_post = nn.Parameter(torch.zeros(1, self.n))
        self.b_res = nn.Parameter(torch.zeros(1, self.n * self.n))
        
        # --- Gating Factors (alpha) ---
        # Khởi tạo giá trị nhỏ (0.01) theo thiết lập siêu tham số của DeepSeek-V3
        init_val = 0.01
        self.alpha_pre = nn.Parameter(torch.tensor(init_val))
        self.alpha_post = nn.Parameter(torch.tensor(init_val))
        self.alpha_res = nn.Parameter(torch.tensor(init_val))
        
        # Module chiếu đa tạp Sinkhorn-Knopp (Birkhoff polytope)
        self.sinkhorn_knopp = SinkhornKnopp(num_iters=20)

    def compute_mappings(self, x: torch.Tensor):
        """
        Tính toán các ma trận ánh xạ có ràng buộc đa tạp.
        Args:
            x: Tensor đầu vào kích thước [batch_size, n, C]
        Returns:
            H_pre: [batch_size, 1, n]
            H_post: [batch_size, 1, n]
            H_res: [batch_size, n, n]
        """
        batch_size = x.size(0)
        
        # Làm phẳng (Flatten) đầu vào n-stream: [batch_size, 1, nC]
        x_vec = x.view(batch_size, 1, self.nC)
        
        # Chuẩn hóa RMSNorm
        x_prime = self.rmsnorm(x_vec)
        
        # Tính toán các giá trị chưa ràng buộc (Unconstrained Mappings)
        H_tilde_pre = self.alpha_pre * self.phi_pre(x_prime) + self.b_pre
        H_tilde_post = self.alpha_post * self.phi_post(x_prime) + self.b_post
        H_tilde_res = self.alpha_res * self.phi_res(x_prime) + self.b_res
        
        # Định hình lại H_tilde_res thành [batch_size, n, n]
        H_tilde_res = H_tilde_res.view(batch_size, self.n, self.n)
        
        # Áp dụng Ràng buộc Đa tạp (Manifold Constraints)
        # 1. Non-negativity constraint qua Sigmoid cho Pre và Post
        H_pre = torch.sigmoid(H_tilde_pre)
        H_post = 2.0 * torch.sigmoid(H_tilde_post)
        
        # 2. Birkhoff Polytope constraint qua Sinkhorn-Knopp cho Res
        H_res = self.sinkhorn_knopp(H_tilde_res)
        
        return H_pre, H_post, H_res

    def apply_residual_stream(self, x: torch.Tensor, H_pre: torch.Tensor, H_post: torch.Tensor, H_res: torch.Tensor):
        """
        Thực thi quá trình truyền dẫn luồng thặng dư dựa trên các ma trận ánh xạ.
        (Mô phỏng Phương trình 3 của luận án mHC)
        """
        # --- Read-out từ luồng thặng dư ---
        # H_pre: [batch_size, 1, n], x: [batch_size, n, C]
        # x_in = H_pre @ x -> [batch_size, 1, C]
        x_in = torch.matmul(H_pre, x)
        
        # Loại bỏ chiều 1 ở giữa để đưa vào layer function: [batch_size, C]
        x_in_flat = x_in.squeeze(1)
        
        # --- Layer Function (Attention hoặc FFN) ---
        f_out = self.layer_func(x_in_flat)
        # Phục hồi lại chiều không gian: [batch_size, 1, C]
        f_out = f_out.unsqueeze(1)
        
        # --- Write-in và Cập nhật luồng thặng dư ---
        # 1. Hòa trộn trong luồng (Stream Mixing): H_res @ x
        # H_res: [batch_size, n, n], x: [batch_size, n, C]
        res_mixed = torch.matmul(H_res, x)
        
        # 2. Bơm tín hiệu mới vào luồng (Write-in): H_post^T @ f_out
        # H_post^T: [batch_size, n, 1], f_out: [batch_size, 1, C]
        # output: [batch_size, n, C]
        signal_in = torch.matmul(H_post.transpose(-1, -2), f_out)
        
        # --- Residual Merge Cuối Cùng ---
        x_next = res_mixed + signal_in
        return x_next

def forward(self, x: torch.Tensor):
        """
        Luồng truyền tiến chuẩn phần cứng, loại bỏ hoàn toàn các activation trung gian 
        tốn kém bằng cách tích hợp Triton Fused Kernel.
        """
        # 1. Trích xuất các ma trận ánh xạ (Birkhoff & Non-negative constraints)
        H_pre, H_post, H_res = self.compute_mappings(x)
        
        # 2. Read-out từ luồng thặng dư
        # H_pre: [batch_size, 1, n], x: [batch_size, n, C] -> x_in: [batch_size, C]
        x_in = torch.matmul(H_pre, x).squeeze(1)
        
        # 3. Layer Function (Thực thi Causal Attention hoặc SwiGLU FFN)
        f_out = self.layer_func(x_in) # Kết quả: [batch_size, C]
        
        # 4. Write-in & Stream Mixing bằng FUSED KERNEL
        # Gọi trực tiếp custom kernel để tính gộp trên SRAM của GPU
        # Clone x để đảm bảo an toàn cho đồ thị Autograd (PyTorch sẽ báo lỗi in-place nếu không)
        x_next = apply_fused_residual_merge(
            x_res=x.clone(), 
            f_out=f_out, 
            h_post=H_post.squeeze(1), # [batch_size, 1, n] -> [batch_size, n]
            h_res=H_res               # [batch_size, n, n]
        )
        
        return x_next