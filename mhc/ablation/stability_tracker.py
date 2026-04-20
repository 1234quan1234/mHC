import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PropagationStabilityTracker:
    def __init__(self):
        """
        Bộ công cụ theo dõi (Tracker) để đo lường Amax Gain Magnitude của 
        các ma trận thặng dư H_res trong quá trình lan truyền tín hiệu.
        """
        self.h_res_matrices = []

    def register_h_res(self, h_res: torch.Tensor):
        """
        Ghi nhận ma trận H_res tại mỗi tầng trong lượt forward.
        Sử dụng .detach() để không can thiệp vào đồ thị tính toán (computation graph).
        
        Args:
            h_res: Ma trận ngẫu nhiên kép kích thước [batch, seq_len, n, n]
        """
        # Lấy trung bình qua batch và sequence length để có ma trận đại diện [n, n]
        h_res_mean = torch.mean(h_res.detach().float(), dim=(0, 1))
        self.h_res_matrices.append(h_res_mean)

    def compute_amax_gain_magnitude(self):
        """
        Tính toán Amax Gain Magnitude cho ánh xạ đơn tầng (single-layer) 
        và ánh xạ tổ hợp (composite mapping).
        """
        num_layers = len(self.h_res_matrices)
        if num_layers == 0:
            print("Chưa có dữ liệu H_res nào được ghi nhận.")
            return None, None

        single_forward_gains = []
        single_backward_gains = []
        
        # 1. Đo lường trên Ánh xạ Đơn tầng (Single-layer Mapping)
        for h_res in self.h_res_matrices:
            # Tín hiệu Forward bị khuếch đại tối đa theo tổng hàng (Row Sum)
            fw_gain = torch.amax(torch.sum(torch.abs(h_res), dim=-1))
            # Tín hiệu Backward (Gradient) bị khuếch đại tối đa theo tổng cột (Col Sum)
            bw_gain = torch.amax(torch.sum(torch.abs(h_res), dim=-2))
            
            single_forward_gains.append(fw_gain.item())
            single_backward_gains.append(bw_gain.item())

        # 2. Đo lường trên Ánh xạ Tổ hợp (Composite Mapping)
        composite_forward_gains = []
        composite_backward_gains = []
        
        # Bắt đầu nhân dồn các ma trận từ tầng sâu nhất ngược về tầng nông nhất
        # Tương đương với công thức: \prod_{i=1}^{L-l} H_{L-i}^{res}
        composite_matrix = self.h_res_matrices[-1]
        
        for i in range(num_layers - 2, -1, -1):
            # Cập nhật Gain cho mỗi độ sâu tổ hợp
            fw_gain = torch.amax(torch.sum(torch.abs(composite_matrix), dim=-1))
            bw_gain = torch.amax(torch.sum(torch.abs(composite_matrix), dim=-2))
            composite_forward_gains.append(fw_gain.item())
            composite_backward_gains.append(bw_gain.item())
            
            # Nhân ma trận tổ hợp (Compositional Closure)
            composite_matrix = torch.matmul(composite_matrix, self.h_res_matrices[i])

        # Bổ sung tính toán cho tầng cuối cùng
        fw_gain = torch.amax(torch.sum(torch.abs(composite_matrix), dim=-1))
        bw_gain = torch.amax(torch.sum(torch.abs(composite_matrix), dim=-2))
        composite_forward_gains.append(fw_gain.item())
        composite_backward_gains.append(bw_gain.item())

        print("=== BÁO CÁO ỔN ĐỊNH LAN TRUYỀN (AMAX GAIN MAGNITUDE) ===")
        print(f"Max Single-Layer Forward Gain : {max(single_forward_gains):.4f}")
        print(f"Max Single-Layer Backward Gain: {max(single_backward_gains):.4f}")
        print(f"Max Composite Forward Gain    : {max(composite_forward_gains):.4f} (Kỳ vọng ~ 1.6 với mHC)")
        print(f"Max Composite Backward Gain   : {max(composite_backward_gains):.4f} (Kỳ vọng ~ 1.6 với mHC)")
        
        return composite_forward_gains, composite_backward_gains

    def clear(self):
        """Xóa bộ đệm sau mỗi epoch/batch."""
        self.h_res_matrices.clear()

# --- Hướng dẫn tích hợp vào mHC Layer ---
# Trong file mhc/models/mhc_layer.py, bạn có thể gọi tracker như sau:
# 
# global_stability_tracker = PropagationStabilityTracker()
# ...
# (Bên trong hàm forward của mHCLayer):
# h_res = self.sinkhorn_knopp(h_res_tilde)
# if not self.training:  # Chỉ cần track trong quá trình Evaluation/Ablation
#     global_stability_tracker.register_h_res(h_res)