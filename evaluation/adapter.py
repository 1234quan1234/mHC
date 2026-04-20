import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

@register_model("mhc")
class mHCAdapter(LM):
    def __init__(self, model, tokenizer, batch_size=1, max_length=2048):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self.device = next(model.parameters()).device

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return self._max_length

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        Thực hiện forward pass thực tế. 
        inps: Tensor [Batch, Seq_Len] chứa token IDs.
        """
        with torch.no_grad():
            return self.model(inps)

    def generate_until(self, requests):
        """
        Sử dụng vòng lặp generate_text thực tế (đã viết ở bước trước) 
        để trả về văn bản sinh ra cho đến khi gặp stop sequences.
        """
        from mhc.evaluation.run_benchmarks import generate_text
        
        results = []
        for context, gen_kwargs in requests:
            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", 128)
            
            input_ids = torch.tensor([self.tok_encode(context)]).to(self.device)
            output_ids = generate_text(self.model, input_ids, max_new_tokens=max_gen)
            
            # Chỉ lấy phần text mới sinh ra
            new_tokens = output_ids[0, input_ids.shape[1]:]
            decoded = self.tok_decode(new_tokens)
            
            # Cắt bỏ phần text sau stop sequences (nếu có)
            for term in until:
                decoded = decoded.split(term)[0]
            results.append(decoded)
            
        return results

    def loglikelihood(self, requests):
        """
        Thực thi tính toán xác suất log thực tế cho các bài thi trắc nghiệm.
        Mỗi request chứa (context, target).
        """
        self.model.eval()
        results = []

        for context, target in requests:
            # 1. Encode context và target
            ctx_ids = self.tok_encode(context)
            tgt_ids = self.tok_encode(target)
            
            # Nối lại để tính toán: [context + target]
            input_ids = torch.tensor([ctx_ids + tgt_ids]).to(self.device)
            
            with torch.no_grad():
                # Forward pass qua toàn bộ kiến trúc mHC (đã tối ưu kernel)
                logits = self.model(input_ids) # [1, seq_len, vocab_size]
                
                # 2. Lấy logits tương ứng với vị trí của target
                # Logits tại vị trí i dự đoán token tại i+1. 
                # Vậy ta lấy từ cuối context đến sát token cuối cùng của target.
                target_logits = logits[:, len(ctx_ids)-1 : -1, :]
                
                # 3. Chuyển sang Log-Probabilities
                log_probs = F.log_softmax(target_logits, dim=-1)
                
                # 4. Trích xuất log-prob của ĐÚNG các token IDs trong target
                target_ids_tensor = torch.tensor(tgt_ids).to(self.device).unsqueeze(0).unsqueeze(-1)
                # Dùng gather để lấy xác suất của các token mục tiêu
                token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids_tensor).squeeze(-1)
                
                # 5. Tổng hợp kết quả
                # results: (log_probability_sum, is_greedy_match)
                total_log_prob = token_log_probs.sum().item()
                
                # Kiểm tra xem nếu dùng Greedy decoding thì có ra đúng target không
                is_greedy = (torch.argmax(target_logits, dim=-1) == torch.tensor(tgt_ids).to(self.device)).all().item()
                
                results.append((total_log_prob, is_greedy))
                
        return results