import torch
import torch.nn as nn

class mHCAblationWrapper(nn.Module):
    def __init__(self, mhc_layer: nn.Module, n_dim: int):
        """
        Wrapper bao bọc mHCLayer để can thiệp và vô hiệu hóa các thành phần 
        trong quá trình thực thi Ablation Study.
        
        Args:
            mhc_layer: Tầng mHC đã được khởi tạo.
            n_dim: Hệ số mở rộng luồng thặng dư (ví dụ: 4).
        """
        super().__init__()
        self.mhc_layer = mhc_layer
        self.n_dim = n_dim

    def forward(self, x: torch.Tensor, disable_pre: bool = False, disable_post: bool = False, disable_res: bool = False):
        """
        Can thiệp vào các tham số ánh xạ theo kịch bản Ablation.
        """
        # Lấy các ma trận ánh xạ nguyên bản từ tầng mHC
        h_pre, h_post, h_res = self.mhc_layer.compute_mappings(x)

        # 1. Vô hiệu hóa H_pre: Thay bằng trọng số đồng nhất 1/n
        if disable_pre:
            h_pre = torch.ones_like(h_pre) / self.n_dim

        # 2. Vô hiệu hóa H_post: Thay bằng ma trận toàn số 1
        if disable_post:
            h_post = torch.ones_like(h_post)

        # 3. Vô hiệu hóa H_res: Thay bằng ma trận đơn vị (Identity Matrix)
        if disable_res:
            batch_size = h_res.size(0)
            device = h_res.device
            h_res = torch.eye(self.n_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # Thực thi luồng dữ liệu với các ma trận đã được can thiệp
        return self.mhc_layer.apply_residual_stream(x, h_pre, h_post, h_res)

def run_ablation_experiment(model: nn.Module, dataloader, criterion, n_dim: int = 4):
    """
    Thực thi toàn bộ các kịch bản kiểm chứng và in ra Absolute Loss Gap.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    scenarios = {
        "Full mHC": {"disable_pre": False, "disable_post": False, "disable_res": False},
        "No H_pre": {"disable_pre": True, "disable_post": False, "disable_res": False},
        "No H_pre, No H_post": {"disable_pre": True, "disable_post": True, "disable_res": False},
        "No H_pre, No H_post, No H_res": {"disable_pre": True, "disable_post": True, "disable_res": True},
    }

    results = {}

    with torch.no_grad():
        for name, config in scenarios.items():
            total_loss = 0.0
            batches = 0
            
            # Gắn wrapper cho toàn bộ các tầng mHC trong mô hình
            for module in model.modules():
                if hasattr(module, 'mhc_layer'):
                    module.mhc_layer = mHCAblationWrapper(module.mhc_layer, n_dim)
            
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Truyền config ablation vào quá trình forward
                outputs = model(inputs, **config)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                batches += 1
                
            avg_loss = total_loss / batches
            results[name] = avg_loss
            print(f"Scenario: {name:<35} | Loss: {avg_loss:.4f}")

    # Tính toán Absolute Loss Gap so với mô hình mHC đầy đủ
    baseline_loss = results["Full mHC"]
    print("\n--- Absolute Loss Gap ---")
    for name, loss in results.items():
        gap = baseline_loss - loss
        print(f"{name:<35} | Gap: {gap:+.3f}")