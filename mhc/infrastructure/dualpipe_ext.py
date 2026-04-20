import torch

class DualPipeOverlapManager:
    def __init__(self):
        """
        Trình quản lý lịch trình chồng lấp Tính toán - Giao tiếp.
        Tạo ra các CUDA streams với mức độ ưu tiên khác nhau để tránh bottleneck.
        """
        # Kiểm tra xem thiết bị có hỗ trợ ưu tiên luồng không
        self.high_priority_supported = torch.cuda.is_available()
        
        if self.high_priority_supported:
            # priority = -1 thường được CUDA quy định là mức ưu tiên cao hơn
            self.high_priority_stream = torch.cuda.Stream(priority=-1)
            # Luồng tiêu chuẩn cho các thao tác mạng/giao tiếp
            self.comm_stream = torch.cuda.Stream(priority=0)
        else:
            self.high_priority_stream = None
            self.comm_stream = None

    def run_mlp_kernel_async(self, kernel_func, *args, **kwargs):
        """
        Đẩy kernel tính toán của MLP lên luồng ưu tiên cao.
        Hàm này đảm bảo việc tính toán các F_{post, res} diễn ra cực nhanh
        và không bị chặn bởi các thao tác I/O mạng đang chạy trên luồng khác.
        """
        if not self.high_priority_supported:
            return kernel_func(*args, **kwargs)
            
        with torch.cuda.stream(self.high_priority_stream):
            result = kernel_func(*args, **kwargs)
            
        return result

    def run_communication_async(self, comm_func, *args, **kwargs):
        """
        Chạy các tác vụ giao tiếp (Send/Recv/AllReduce) trên luồng tiêu chuẩn.
        """
        if not self.high_priority_supported:
            return comm_func(*args, **kwargs)
            
        with torch.cuda.stream(self.comm_stream):
            result = comm_func(*args, **kwargs)
            
        return result
        
    def synchronize(self):
        """
        Đồng bộ hóa các luồng để đảm bảo dữ liệu tính toán và giao tiếp đều hoàn tất
        trước khi chuyển sang pipeline stage tiếp theo.
        """
        if self.high_priority_supported:
            self.high_priority_stream.synchronize()
            self.comm_stream.synchronize()