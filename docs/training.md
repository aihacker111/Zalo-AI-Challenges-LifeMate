# Quy trình huấn luyện (1 GPU 24 GB)

Mục tiêu: có Docker nộp trước **23:59 ngày 16/11**. Mọi quyết định đều dựa trên số đo của **cùng một tập val**, và mọi số đo đều ghi vào bảng nhật ký ở cuối file.

Quy ước dùng trong file này:

```bash
cd ~/Downloads/lifemate
DATA=../train+public_test
VAL=outputs/sft/val_ids.txt          # tạo ở bước 1.1, dùng chung cho ASR và LLM
```

---

## 0. Môi trường (ngày 1)

```bash
conda create -n lifemate python=3.12 -y && conda activate lifemate
pip install -e ".[dev,yaml,vllm,asr,train]"
pip install flash-linear-attention causal-conv1d   # kernel cho kiến trúc lai của Qwen3.5 (train nhanh hơn)
huggingface-cli download Qwen/Qwen3.5-4B --local-dir models/base/Qwen3.5-4B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models/base/Qwen3-ASR-0.6B
make test DATA=$DATA                                 # 34 test phải qua
```

Kiểm tra trần tham số ngay từ đầu:

```bash
python - <<'PY'
import torch, transformers
for p in ["models/base/Qwen3.5-4B"]:
    m = transformers.AutoModelForImageTextToText.from_pretrained(p, torch_dtype=torch.bfloat16)
    total = sum(x.numel() for x in m.parameters())
    vision = sum(x.numel() for n, x in m.named_parameters() if "visual" in n)
    print(p, f"total {total/1e9:.2f}B, vision {vision/1e9:.2f}B, text {(total-vision)/1e9:.2f}B")
PY
```

---

## 1. Đo trước khi train (ngày 1–2)

### 1.1 Tạo tập val cố định

Chia leave-apps-out: giữ lại 10 app, khoảng 274 mẫu. Tập này mô phỏng việc private test có app mới.

```bash
lifemate build-sft --data $DATA --asr oracle --holdout apps --n-apps 10 --out-dir outputs/sft
```

### 1.2 Chạy thử (20 mẫu)

```bash
vllm serve models/base/Qwen3.5-4B --served-model-name base --max-model-len 16384 \
  --enable-prefix-caching --gpu-memory-utilization 0.85 --seed 0 &
lifemate infer --data $DATA --split train --backend openai --model base --asr oracle \
  --limit 20 --out outputs/smoke.jsonl --trace outputs/smoke.trace.jsonl
lifemate score --data $DATA --split train --submission outputs/smoke.jsonl --only-submitted
```

Lỗi API của vLLM, structured outputs hay chat template sẽ lộ ra ở bước này. Mở `smoke.trace.jsonl` xem model sinh gì ở từng bước.

### 1.3 Baseline LLM trên val (transcript gold)

```bash
for K in 0 3; do
  lifemate infer --data $DATA --split train --backend openai --model base --asr oracle \
    --ids-file $VAL --exclude-ids $VAL --fewshot-k $K --out outputs/base_k$K.jsonl --trace outputs/base_k$K.trace.jsonl
  lifemate score --data $DATA --split train --submission outputs/base_k$K.jsonl --ids-file $VAL
done
```

`--exclude-ids` là bắt buộc: thiếu nó, few-shot sẽ lấy chính các câu val làm ví dụ và điểm cao giả.

### 1.4 ASR zero-shot

```bash
kill %1   # tắt vLLM để nhường GPU cho ASR
lifemate transcribe --data $DATA --split train --asr qwen --asr-model models/base/Qwen3-ASR-0.6B --out outputs/asr0_train.jsonl
lifemate transcribe --data $DATA --split public_test --asr qwen --asr-model models/base/Qwen3-ASR-0.6B --out outputs/asr0_public.jsonl
lifemate asr-eval --data $DATA --transcripts outputs/asr0_train.jsonl --ids-file $VAL
```

Sau đó chạy lại 1.3 với `--transcripts outputs/asr0_train.jsonl` thay cho `--asr oracle`. **Chênh lệch giữa hai lần chính là phần điểm mất do ASR.**

### 1.5 Một bài nộp public để hiệu chỉnh

```bash
lifemate infer --data $DATA --split public_test --backend openai --model base \
  --transcripts outputs/asr0_public.jsonl --fewshot-k 3 --out outputs/sub_base.jsonl
lifemate validate --data $DATA --split public_test --submission outputs/sub_base.jsonl
```

Nộp `outputs/sub_base.jsonl`, rồi so điểm leaderboard với điểm val ở 1.4.

**Quyết định sau bước 1:**

| Quan sát | Làm tiếp |
|---|---|
| Điểm val (ASR) cách điểm val (gold) > 5 | Ưu tiên bước 2 (ASR) |
| Lỗi chủ yếu là `extra_read`, `missing_read`, `ask_mismatch`, `oos_*` | Bước 3 (SFT LLM) |
| Lỗi `wrong_final_state` với số tiền/giờ sai | Xem trace: lỗi do ASR thì bước 2, do model thì bước 3 |

---

## 2. Fine-tune ASR (ngày 2–4, nếu cần)

```bash
lifemate asr-manifest --data $DATA --out-dir outputs/asr --val-ids $VAL   # cùng tập val với LLM
git clone https://github.com/QwenLM/Qwen3-ASR.git third_party/Qwen3-ASR
python third_party/Qwen3-ASR/finetuning/qwen3_asr_sft.py \
  --model_path models/base/Qwen3-ASR-0.6B \
  --train_file outputs/asr/train.jsonl --eval_file outputs/asr/val.jsonl \
  --output_dir checkpoints/asr --batch_size 8 --grad_acc 8 --lr 1e-5 --epochs 3 \
  --save_steps 100 --save_total_limit 3
```

Chọn checkpoint tốt nhất bằng `asr-eval --ids-file $VAL`. Nhìn chủ yếu vào **tỉ lệ sai số**, không chỉ WER:

```bash
lifemate transcribe --data $DATA --split train --asr qwen --asr-model checkpoints/asr/checkpoint-XXX --out outputs/asr1_train.jsonl
lifemate asr-eval --data $DATA --transcripts outputs/asr1_train.jsonl --ids-file $VAL
```

Lưu ý:
- Chỉ có 6,8 giờ audio, nên dễ overfit. Giữ lr thấp và dừng sớm khi WER val tăng trở lại.
- **Không dùng transcript của ASR đã fine-tune trên chính câu train làm input SFT.** Các câu đó model đã thuộc, nên transcript sạch hơn thực tế. Input SFT lấy từ `asr0_train.jsonl` (bản zero-shot, nhiễu như thật hoặc hơn), trộn với transcript gold.

---

## 3. LoRA SFT cho LLM (ngày 3–7)

### 3.1 Dữ liệu

Có hai nguồn input: transcript gold và transcript ASR zero-shot. Cả hai dùng cùng tập val và cùng augment đổi tên tool.

```bash
lifemate build-sft --data $DATA --asr oracle --holdout apps --n-apps 10 --augment 2 --fewshot-k 3 --out-dir outputs/sft_gold
lifemate build-sft --data $DATA --transcripts outputs/asr0_train.jsonl --holdout apps --n-apps 10 --augment 1 --fewshot-k 3 --out-dir outputs/sft_asr
cat outputs/sft_gold/train.jsonl outputs/sft_asr/train.jsonl | shuf --random-source=<(yes) > outputs/sft/train.jsonl
cp outputs/sft_asr/val.jsonl outputs/sft/val.jsonl
cmp outputs/sft_gold/val_ids.txt $VAL && echo "val không đổi"
```

### 3.2 Train

```bash
python -m lifemate.train.sft_lora --config configs/sft_lora_24gb.yaml output_dir=checkpoints/sft-v1
```

Thời gian khoảng 2–4 giờ cho 2 epoch. Nếu OOM: giảm `max_length=8192`, `lora_r=32`, hoặc dùng QLoRA. Dòng log đầu tiên cho biết bao nhiêu mẫu bị bỏ vì quá dài; con số này nên bằng 0.

### 3.3 Đánh giá trên val

```bash
vllm serve models/base/Qwen3.5-4B --served-model-name base --enable-lora --max-lora-rank 64 \
  --lora-modules sft-v1=checkpoints/sft-v1 --max-model-len 16384 --enable-prefix-caching --seed 0 &
lifemate infer --data $DATA --split train --backend openai --model sft-v1 \
  --transcripts outputs/asr0_train.jsonl --ids-file $VAL --exclude-ids $VAL \
  --out outputs/sft-v1_val.jsonl --trace outputs/sft-v1_val.trace.jsonl
lifemate score --data $DATA --split train --submission outputs/sft-v1_val.jsonl --ids-file $VAL --report outputs/sft-v1_val.scores.jsonl
```

Nếu vLLM chưa hỗ trợ LoRA cho Qwen3.5, gộp adapter vào model gốc rồi serve bản gộp:
`python -m lifemate.train.merge_lora --base models/base/Qwen3.5-4B --adapter checkpoints/sft-v1 --out models/sft-v1`.

### 3.4 Thử nghiệm (mỗi lần đổi **một** yếu tố)

| Thử | Giá trị |
|---|---|
| `learning_rate` | 5e-5 · **1e-4** · 2e-4 |
| `num_train_epochs` | 1 · **2** · 3 |
| `--augment` | 0 · **2** · 4 |
| `--fewshot-k` (train và infer giống nhau) | 0 · **3** |

Giữ cấu hình có điểm val cao nhất. Đọc thêm phân loại lỗi để biết lần sửa tiếp theo nên nhắm vào đâu.

---

## 4. Train lại trên toàn bộ train (ngày 7–8)

Khi đã chọn được cấu hình, dựng lại dữ liệu với `--holdout none`, train với đúng cấu hình đó, rồi gộp adapter:

```bash
lifemate build-sft --data $DATA --asr oracle --holdout none --augment 2 --fewshot-k 3 --out-dir outputs/sft_full_gold
lifemate build-sft --data $DATA --transcripts outputs/asr0_train.jsonl --holdout none --augment 1 --fewshot-k 3 --out-dir outputs/sft_full_asr
cat outputs/sft_full_*/train.jsonl | shuf --random-source=<(yes) > outputs/sft_full.jsonl
python -m lifemate.train.sft_lora --config configs/sft_lora_24gb.yaml train_file=outputs/sft_full.jsonl eval_file=null output_dir=checkpoints/sft-full
python -m lifemate.train.merge_lora --base models/base/Qwen3.5-4B --adapter checkpoints/sft-full --out models/llm
```

Nộp public một lần với `models/llm` (giống bước 1.5) để xác nhận điểm có tăng.

---

## 5. RFT (ngày 8–11, nếu còn thời gian)

Làm thí nghiệm trên tập val trước. Model dùng ở đây phải được train **không** có val (`sft-v1`, gộp thành `models/sft-v1`):

```bash
vllm serve models/sft-v1 --served-model-name sft-v1 --max-model-len 16384 --enable-prefix-caching &
python -m lifemate.train.rft --data $DATA --base-url http://localhost:8000/v1 --model sft-v1 \
  --transcripts outputs/asr0_train.jsonl --samples 4 --temperature 0.7 --out outputs/rft/train_all.jsonl
grep -v -F -f <(sed 's/^/"id": "/' $VAL) outputs/rft/train_all.jsonl > outputs/rft/train.jsonl   # bỏ mẫu val
cat outputs/sft/train.jsonl outputs/rft/train.jsonl > outputs/sft_rft.jsonl
python -m lifemate.train.sft_lora --config configs/sft_lora_24gb.yaml train_file=outputs/sft_rft.jsonl output_dir=checkpoints/rft-v1
```

Đánh giá như 3.3. Chỉ giữ RFT nếu điểm val tăng; khi đó lặp lại bước 4 với dữ liệu RFT sinh trên toàn bộ train.

---

## 6. Docker (ngày 12–15)

```bash
cp -r checkpoints/asr/checkpoint-XXX models/asr        # hoặc models/base/Qwen3-ASR-0.6B nếu không fine-tune ASR
mkdir -p assets/train && cp $DATA/train/train.jsonl assets/train/
docker build -f docker/Dockerfile -t lifemate:final .
time docker run --rm --gpus all --network none -v $PWD/$DATA/public_test:/data -v $PWD/outputs/docker:/app/result lifemate:final
```

Kiểm tra trước khi nộp:
- `validate` báo `OK`.
- Chạy 2 lần cho cùng một file (kiểm tra tính xác định).
- Trung bình dưới 30 giây mỗi mẫu; xem `slowest_s` trong log.
- Tổng tham số dưới 6B (bước 0).
- Nộp public từ chính output của Docker.

Hạn: **23:59 ngày 16/11**, tối đa 2 lần nộp Docker.

## 7. Private test (20–23/11)

```bash
docker run --rm --gpus all --network none -v /path/to/private_test:/data -v $PWD/outputs/private:/app/result lifemate:final
```

Nộp `outputs/private/submission.jsonl`. Không sửa gì giữa Docker đã nộp và lần chạy này.

---

## Nhật ký thí nghiệm

| Ngày | Thí nghiệm | Val (gold) | Val (ASR) | Public | Ghi chú |
|---|---|---|---|---|---|
| | base k=0 | | | | |
| | base k=3 | | | | |
| | ASR zero-shot: WER / sai số | | | | |
| | sft-v1 | | | | |
