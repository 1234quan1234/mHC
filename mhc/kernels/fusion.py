import torch
import triton
import triton.language as tl

@triton.jit
def fused_residual_merge_kernel(
    x_res_ptr, f_out_ptr, h_post_ptr, h_res_ptr,
    C, 
    stride_x_batch, stride_x_n, stride_x_c,
    stride_f_batch, stride_f_c,
    BLOCK_C: tl.constexpr, N_DIM: tl.constexpr
):
    """
    Triton Kernel thực hiện hợp nhất: x_res = H_res @ x_res + H_post^T @ f_out
    N_DIM tương đương với hệ số mở rộng n (ví dụ: 4).
    """
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    
    # Tính toán offset cho block C hiện tại
    offset_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offset_c < C
    
    # 1. LOAD PHASE
    # Load f_out (1 x C)
    f_out_ptrs = f_out_ptr + batch_id * stride_f_batch + offset_c * stride_f_c
    f_out = tl.load(f_out_ptrs, mask=mask_c, other=0.0)
    
    # Load H_post (1 x n)
    h_post_ptrs = h_post_ptr + batch_id * N_DIM + tl.arange(0, N_DIM)
    h_post = tl.load(h_post_ptrs)
    
    # Load H_res (n x n)
    h_res_offsets = tl.arange(0, N_DIM)[:, None] * N_DIM + tl.arange(0, N_DIM)[None, :]
    h_res_ptrs = h_res_ptr + batch_id * N_DIM * N_DIM + h_res_offsets
    h_res = tl.load(h_res_ptrs)
    
    # Load x_res (n x C)
    x_res_ptrs = x_res_ptr + batch_id * stride_x_batch + tl.arange(0, N_DIM)[:, None] * stride_x_n + offset_c[None, :] * stride_x_c
    x_res = tl.load(x_res_ptrs, mask=mask_c[None, :], other=0.0)
    
    # 2. COMPUTE PHASE (Hoàn toàn trên SRAM)
    # H_res @ x_res: Nhân ma trận (n x n) với (n x BLOCK_C)
    res_merge = tl.dot(h_res, x_res)
    
    # H_post^T @ f_out: Outer product (n x 1) với (1 x BLOCK_C)
    out_merge = h_post[:, None] * f_out[None, :]
    
    # Cộng dồn thặng dư
    acc_res = res_merge + out_merge
    
    # 3. STORE PHASE
    # Ghi đè trực tiếp kết quả vào x_res (In-place)
    tl.store(x_res_ptrs, acc_res, mask=mask_c[None, :])


def apply_fused_residual_merge(x_res: torch.Tensor, f_out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    """
    Hàm wrapper gọi Triton kernel để thực thi Fused Residual Merge.
    Args:
        x_res: [batch_size, n, C]
        f_out: [batch_size, C]
        h_post: [batch_size, n]
        h_res: [batch_size, n, n]
    """
    batch_size, n, C = x_res.shape
    
    # Định nghĩa cấu hình grid
    BLOCK_C = 128
    grid = lambda META: (triton.cdiv(C, META['BLOCK_C']), batch_size)
    
    # Gọi kernel
    fused_residual_merge_kernel[grid](
        x_res, f_out, h_post, h_res,
        C,
        x_res.stride(0), x_res.stride(1), x_res.stride(2),
        f_out.stride(0), f_out.stride(1),
        BLOCK_C=BLOCK_C, N_DIM=n
    )
    
    return x_res