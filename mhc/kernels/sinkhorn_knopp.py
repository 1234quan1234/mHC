import torch
import torch.nn as nn

class SinkhornKnoppFunction(torch.autograd.Function):
    """
    Triển khai Custom Autograd cho thuật toán Sinkhorn-Knopp.
    Áp dụng Recomputation trong lượt backward để tối ưu hóa bộ nhớ GPU.
    """
    @staticmethod
    def forward(ctx, h_res_tilde: torch.Tensor, num_iters: int, eps: float) -> torch.Tensor:
        # Bước 1: Ổn định số học (Log-Max-Exp trick)
        # Lấy max trên 2 chiều cuối cùng (n, n) để tránh overflow khi tính hàm mũ
        max_val = torch.amax(h_res_tilde, dim=(-2, -1), keepdim=True)
        M = torch.exp(h_res_tilde - max_val)
        
        # Bước 2: Lặp Sinkhorn-Knopp t_max lần
        for _ in range(num_iters):
            # T_c: Chuẩn hóa cột (Column normalization)
            col_sum = M.sum(dim=-2, keepdim=True)
            M = M / (col_sum + eps)
            
            # T_r: Chuẩn hóa hàng (Row normalization)
            row_sum = M.sum(dim=-1, keepdim=True)
            M = M / (row_sum + eps)
            
        # Tối ưu bộ nhớ: Chỉ lưu trữ tensor đầu vào (h_res_tilde), vứt bỏ các M trung gian
        ctx.save_for_backward(h_res_tilde)
        ctx.num_iters = num_iters
        ctx.eps = eps
        
        return M

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        h_res_tilde, = ctx.saved_tensors
        
        # Bật lại gradient tracking cho quá trình Recompute
        with torch.enable_grad():
            h_res_recomputed = h_res_tilde.detach().requires_grad_(True)
            
            # Tính toán lại (Recompute) forward pass ngay trên chip (SRAM)
            max_val = torch.amax(h_res_recomputed, dim=(-2, -1), keepdim=True)
            M_recomputed = torch.exp(h_res_recomputed - max_val)
            
            for _ in range(ctx.num_iters):
                M_recomputed = M_recomputed / (M_recomputed.sum(dim=-2, keepdim=True) + ctx.eps)
                M_recomputed = M_recomputed / (M_recomputed.sum(dim=-1, keepdim=True) + ctx.eps)
                
        # Tính gradient cuối cùng thông qua đồ thị (graph) vừa được recompute
        grad_input, = torch.autograd.grad(
            outputs=M_recomputed, 
            inputs=h_res_recomputed, 
            grad_outputs=grad_output
        )
        
        # Trả về gradient tương ứng với các tham số đầu vào của hàm forward
        return grad_input, None, None


class SinkhornKnopp(nn.Module):
    def __init__(self, num_iters: int = 20, eps: float = 1e-6):
        """
        Module chuẩn hóa ma trận thặng dư lên đa tạp Birkhoff.
        
        Args:
            num_iters: Số vòng lặp t_max (Mặc định: 20 theo bài báo).
            eps: Hằng số epsilon nhỏ để tránh lỗi chia cho 0.
        """
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps

    # Sử dụng torch.compile để tự động hợp nhất kernel (Kernel Fusion) trên PyTorch 2.x
    # Điều này thay thế cho việc phải viết Triton kernel thuần túy cho Sinkhorn
    @torch.compile(mode="reduce-overhead")
    def forward(self, h_res_tilde: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_res_tilde: Tensor đầu vào kích thước [..., n, n]
            
        Returns:
            Ma trận ngẫu nhiên kép kích thước [..., n, n]
        """
        # Gọi custom autograd function
        return SinkhornKnoppFunction.apply(h_res_tilde, self.num_iters, self.eps)