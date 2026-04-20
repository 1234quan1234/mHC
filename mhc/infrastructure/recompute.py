import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SelectiveRecomputeEngine(nn.Module):
    def __init__(self, layers: nn.ModuleList, n_dim: int, total_layers: int):
        """
        Engine quản lý bộ nhớ thông qua cơ chế Recompute theo khối (Block-wise).
        
        Args:
            layers: Danh sách các tầng (ví dụ: mHCLayer)
            n_dim: Hệ số mở rộng n
            total_layers: Tổng số tầng L của mô hình
        """
        super().__init__()
        self.layers = layers
        self.n_dim = n_dim
        self.total_layers = total_layers
        
        # Tính toán kích thước khối tối ưu theo công thức L_r
        self.L_r = max(1, round(math.sqrt((self.n_dim * self.total_layers) / (self.n_dim + 2))))
        
        # Chia các layer thành các khối có kích thước L_r
        self.blocks = nn.ModuleList([
            nn.Sequential(*self.layers[i : i + self.L_r])
            for i in range(0, self.total_layers, self.L_r)
        ])

    def forward(self, x: torch.Tensor, use_reentrant: bool = False) -> torch.Tensor:
        """
        Thực hiện Forward Pass. Trong lượt forward, chỉ lưu tensor đầu vào của 
        mỗi khối, các activation trung gian bên trong khối sẽ bị vứt bỏ.
        """
        for block in self.blocks:
            # Sử dụng checkpoint để loại bỏ activation trung gian và tính toán lại trong backward
            # use_reentrant=False được khuyến nghị cho PyTorch 2.x để tối ưu hiệu năng
            x = checkpoint(block, x, use_reentrant=use_reentrant)
        return x