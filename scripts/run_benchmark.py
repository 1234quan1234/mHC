import lm_eval
from mhc.evaluation.adapter import mHCAdapter
from mhc.models.causal_lm import mHC_CausalLM # Giả sử file class đã lưu
from transformers import AutoTokenizer

def main():
    # 1. Load mô hình mHC đã checkpoint sau huấn luyện
    # Lưu ý: Thay đổi các tham số này khớp với cấu hình bạn đã huấn luyện
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("path/to/your/tokenizer")
    
    model = mHC_CausalLM(
        vocab_size=len(tokenizer),
        num_layers=12,
        d_model=1024,
        n_dim=4,
        num_heads=16
    ).to(device)
    
    # Load weights (Real weights, không giả lập)
    # model.load_state_dict(torch.load("checkpoints/mhc_model_final.pt"))
    
    # 2. Khởi tạo Adapter cho framework
    lm_obj = mHCAdapter(model, tokenizer, batch_size=4)

    # 3. Danh sách các task cần đánh giá (BBH, DROP, GSM8K)
    tasks = ["bbh_fewshot", "drop", "gsm8k"]

    # 4. Thực thi đánh giá tự động
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        num_fewshot=5, # Cấu hình 5-shot như bài báo
        device=device
    )

    # 5. In báo cáo kết quả
    print(lm_eval.utils.make_table(results))

if __name__ == "__main__":
    main()