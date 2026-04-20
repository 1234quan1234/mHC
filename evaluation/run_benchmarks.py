import torch
import torch.nn.functional as F
from tqdm import tqdm

def generate_text(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.0) -> torch.Tensor:
    """
    Vòng lặp sinh text Auto-regressive thực tế cho mô hình mHC.
    Hỗ trợ Greedy Decoding (temperature=0.0) tiêu chuẩn cho các bài benchmark toán/logic.
    """
    model.eval()
    device = input_ids.device
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Lượt forward qua mô hình (Sử dụng toàn bộ context hiện tại)
            # Lưu ý: Ở mức độ production, ta sẽ implement KV-Cache tại đây để tăng tốc.
            logits = model(generated_ids)
            
            # Chỉ lấy logits của token cuối cùng để dự đoán
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                # Greedy Decoding: Chọn token có xác suất cao nhất
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Sampling có nhiệt độ
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Nối token vừa dự đoán vào chuỗi
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Điều kiện dừng sớm nếu gặp EOS Token (Giả sử EOS id = 2)
            if (next_token == 2).all():
                break

    return generated_ids

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """Đo lường Exact Match (EM) - Metric chuẩn cho DROP và GSM8K."""
    correct = 0
    for pred, ref in zip(predictions, references):
        # Tiền xử lý cơ bản: Xóa khoảng trắng và chuyển chữ thường
        pred_clean = pred.strip().lower()
        ref_clean = ref.strip().lower()
        if ref_clean in pred_clean: # Kiểm tra xem đáp án có nằm trong chuỗi sinh ra không
            correct += 1
    return correct / len(predictions) if predictions else 0.0

def evaluate_on_benchmark(model, tokenizer, dataset, task_name: str, batch_size: int = 4):
    """
    Hàm thực thi Benchmark trên tập dữ liệu thực (Zero-shot / Few-shot).
    
    Args:
        model: Mô hình mHC_CausalLM đã huấn luyện.
        tokenizer: Tokenizer (ví dụ: LlamaTokenizer).
        dataset: List các dict [{'question': "...", 'answer': "..."}, ...]
        task_name: Tên task (BBH, DROP, GSM8K...).
    """
    print(f"=== BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP BENCHMARK: {task_name.upper()} ===")
    device = next(model.parameters()).device
    
    all_preds = []
    all_refs = []
    
    # Chia batch thủ công
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {task_name}"):
        batch = dataset[i : i + batch_size]
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        
        # Tokenize prompt (Bao gồm instruction hoặc few-shot examples nếu có)
        inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(device)
        input_len = inputs.input_ids.shape[1]
        
        # Sinh văn bản (Giới hạn 128 token để trả lời)
        output_ids = generate_text(model, inputs.input_ids, max_new_tokens=128, temperature=0.0)
        
        # Giải mã (Decode) phần văn bản MỚI được sinh ra (Bỏ qua phần prompt)
        new_token_ids = output_ids[:, input_len:]
        preds = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        
        all_preds.extend(preds)
        all_refs.extend(answers)
        
    # Tính toán điểm số cuối cùng
    accuracy = exact_match_score(all_preds, all_refs)
    print(f"\n[KẾT QUẢ {task_name.upper()}] Exact Match Accuracy: {accuracy * 100:.2f}%\n")
    return accuracy

# --- Ví dụ cách tích hợp vào main ---
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# dummy_gsm8k = [{"question": "Nam có 5 quả táo, ăn 2 quả. Còn mấy quả?", "answer": "3"}]
# evaluate_on_benchmark(trained_mhc_model, tokenizer, dummy_gsm8k, "gsm8k")