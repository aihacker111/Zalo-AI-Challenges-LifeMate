# Quy trình đầy đủ: từ dữ liệu tới bài nộp

## Quy trình hiện tại (fine-tune gipformer → LLM SFT trên toàn bộ dữ liệu)

Trên 878 câu val, **gipformer1.5-68M chưa fine-tune** đạt WER 16,1%, tốt hơn nhiều so với Qwen3-ASR-0.6B (36,1% gốc, 27,0% sau fine-tune). Bước tiếp theo là fine-tune chính gipformer trên audio train (6,8 giờ, 5.036 câu, 16 kHz mono):

- **ASR:** fine-tune gipformer bằng icefall (`zipformer/finetune.py`), rồi export ra ONNX. `lifemate transcribe --asr gipformer --asr-model <thư mục>` đọc được ngay.
  - Chạy trước một lần có tập val (giữ lại 10 app, 878 câu) để chọn số epoch.
  - Sau đó train lại trên toàn bộ train.
- **LLM:** SFT vẫn dùng transcript của gipformer **gốc** (`MIX_ASR=1 ASR=gipformer`, không truyền `ASR_MODEL`). Bản fine-tune đã học thuộc audio train nên gần như không sai trên train. Nếu dùng transcript của nó, LLM sẽ không thấy lỗi ASR thật như khi gặp đề.
- **Bài nộp:** gipformer fine-tune (full) → LLM SFT (full).

**Kết quả fine-tune có val** (878 câu của 10 app giữ lại, WER spoken form): gipformer gốc 15,49% → fine-tune 10 epoch, lr 0,002 là **10,14%** (giảm 35% tương đối). Lưu ý: val giữ lại theo app, không theo người nói. Nếu đề private có giọng mới, mức cải thiện thực tế có thể thấp hơn.

Gipformer viết kiểu văn nói: chữ thường, không dấu câu, số đọc thành chữ ("hai mươi bốn độ"). Vì vậy:
- Text train được chuyển sang dạng này (`src/lifemate/asr/normalize_vi.py`).
- Bỏ 177 câu (3,5%) có cách đọc không chắc chắn, như "RV245", "VTV3", "7h30".
- `asr-eval` in thêm dòng **`WER spoken form`** (cả gold lẫn transcript đều đổi sang văn nói). Đây là con số dùng để so sánh các engine.

```bash
# ── 0. Đồng bộ (máy GPU) ──────────────────────────────────────────────────────
cd /media/hung/HDD2/workplaces/tin/Zalo-AI-Challenges-LifeMate
git pull
pgrep -af "bin/vllm serve"                       # nếu có dòng nào: kill <pid>
nvidia-smi                                       # GPU phải trống
tmux new -s asr                                  # Ctrl+B rồi D để thoát; tmux attach -t asr để xem lại

# ── 1. Manifest + baseline (env lifemate) ─────────────────────────────────────
conda activate lifemate
pip install -e ".[gipformer]"
lifemate asr-manifest --split train --holdout apps --out-dir outputs/asr_ft     # 4.158 train / 878 val
lifemate asr-manifest --split train --holdout none --out-dir outputs/asr_full   # 5.036 câu, không val
ENGINES="gipformer" bash scripts/asr_benchmark.sh   # baseline: ghi lại dòng "WER spoken form"

# ── 2-3. Fine-tune có val (vẫn env lifemate) ─────────────────────────────────
bash scripts/gipformer_finetune.sh
#   lần đầu tự cài k2 khớp đúng torch đang có (không cài lại torch), lhotse, và clone icefall
#   0/4 tải model gốc → models/gipformer/base (model.pt, bpe.model, tokens.txt, *.onnx)
#   1/4 model.pt → pretrained.pt (định dạng finetune.py cần; in ra số key thiếu, phải là 0)
#   2/4 cut + fbank, speed perturb 0.9/1.0/1.1 → outputs/gipformer/data_ft
#   3/4 10 epoch, lr 0.002, max-duration 600 → checkpoints/gipformer-ft (log trong exp-dir/log/)
#   4/4 export epoch 10 (trung bình 2 epoch cuối) → models/gipformer-ft-e10
# Hết VRAM: MAX_DURATION=300 bash scripts/gipformer_finetune.sh
# Validation loss in ra mỗi 100 batch, kể cả ở batch 0 (= model gốc). Loss phải giảm.

# ── 4. Đo WER ──────────────────────────────────────────────────────────────────
ENGINES="gipformer gipformer:models/gipformer-ft-e10" bash scripts/asr_benchmark.sh
# Thử epoch khác (chỉ export, ~1 phút):
#   STAGE=4 EPOCH=6 bash scripts/gipformer_finetune.sh  →  models/gipformer-ft-e6
# Nếu chưa tốt hơn baseline: thử BASE_LR=0.001 hoặc EPOCHS=6, đặt TAG khác để giữ bản cũ.

# ── 5. Fine-tune trên toàn bộ train, dùng số epoch tốt nhất ở bước 4 ─────────
MANIFEST=outputs/asr_full TAG=full EPOCHS=10 bash scripts/gipformer_finetune.sh
#   → models/gipformer-full-e10. Không còn val: tin vào kết quả của bước 4.

# ── 6. LLM SFT trên toàn bộ dữ liệu (~6–7 giờ) ───────────────────────────────
HOLDOUT=none MIX_ASR=1 ASR=gipformer NAME=sft-full bash scripts/train_sft_24gb.sh
#   MIX_ASR dùng gipformer GỐC (cố ý, xem giải thích ở trên)

# ── 7. Bài nộp public: ASR fine-tune + LLM fine-tune ─────────────────────────
ASR_MODEL=models/gipformer-full-e10 LORA=checkpoints/sft-full bash scripts/predict_public.sh
#   → outputs/public_test/submission_sft-full.jsonl, upload lên trang thi
```

Bước 6 không phụ thuộc vào ASR fine-tune, nên có thể chạy song song khi GPU rảnh. Bước 3 và bước 5 tốn khoảng 20–40 phút GPU mỗi lần.

### Docker (trước 23:59 ngày 16/11)

```bash
python -m lifemate.train.merge_lora --base models/base/Qwen3.5-4B --adapter checkpoints/sft-full --out models/llm
mkdir -p models/asr && cp models/gipformer-full-e10/{encoder,decoder,joiner}.onnx models/gipformer-full-e10/tokens.txt models/asr/
mkdir -p assets/train && cp data/train/train.jsonl assets/train/
docker build -f docker/Dockerfile -t lifemate:final .
time docker run --rm --gpus all --network none \
  -v $(realpath data)/public_test:/data -v $PWD/outputs/docker:/app/result lifemate:final
```

Trước khi nộp, kiểm tra:
- `validate` báo `OK`.
- Điểm public của `outputs/docker/submission.jsonl` khớp với bài nộp ở bước 7.
- Trung bình dưới 30 giây mỗi mẫu.
- Tổng tham số khoảng 4,2B (LLM) + 0,07B (gipformer), dưới trần 6B.

Các phần bên dưới là quy trình chi tiết có tập val, dùng khi cần thử nghiệm (đổi cấu hình, so sánh model).

---


Hệ thống gồm hai model chạy nối tiếp. Mọi bài nộp đều đi qua cả hai:

```
audio → ASR (Qwen3-ASR-0.6B, fine-tune) → transcript → LLM agent (Qwen3.5-4B + LoRA) → tool calls → file nộp
```

Mỗi model được train riêng vì sửa một loại lỗi riêng:
- **ASR**: nghe sai. Bản gốc có WER khoảng 37% trên audio train.
- **LLM**: làm sai quy trình. Dù transcript hoàn hảo, bản gốc chỉ đạt 41,76 điểm trên val.

Mỗi model đi qua hai vòng:
1. **Vòng thử nghiệm:** chừa ra tập val (274 mẫu, 10 app), để đo được điểm.
2. **Vòng cuối:** train trên toàn bộ train, dùng cho bài nộp.

Cả hai model dùng **cùng một tập val**, nên các con số so sánh trực tiếp được với nhau.

Chỉ có một GPU 24 GB: mọi bước chạy **lần lượt**. Trước mỗi bước train, tắt vLLM bằng `kill $VLLM_PID`. Các bước dài nên chạy trong `tmux`.

| Phần | Việc | Thời gian ước tính |
|---|---|---|
| 0 | Chuẩn bị (một lần) | 30 phút |
| 1 | ASR: fine-tune + đo trên val | ~1,5 giờ |
| 2 | LLM: SFT + đo trên val | ~3,5 giờ |
| 3 | Đo toàn hệ thống trên val, nộp public | ~40 phút |
| 4 | Bản cuối: ASR + LLM trên toàn bộ train, nộp public | ~5 giờ |
| 5 | Docker (trước 23:59 ngày 16/11) | 1–2 giờ |
| 6 | Private test (20–23/11) | ~30 phút |

---

## Phần 0 — Chuẩn bị (một lần)

```bash
cd /media/hung/HDD2/workplaces/tin/Zalo-AI-Challenges-LifeMate
git pull

# Môi trường chính: agent, vLLM, train LLM
conda activate lifemate
pip install -e ".[dev,yaml,vllm,asr,train]" flash-linear-attention

# Dữ liệu và model gốc
ln -s /đường/dẫn/tới/train+public_test data                 # nếu chưa có
huggingface-cli download Qwen/Qwen3.5-4B --local-dir models/base/Qwen3.5-4B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models/base/Qwen3-ASR-0.6B
make test                                                    # 34 test phải qua

# Môi trường riêng để fine-tune ASR (script của Qwen chỉ chạy với transformers 4.x)
git clone https://github.com/QwenLM/Qwen3-ASR.git third_party/Qwen3-ASR
conda create -n asrft python=3.12 -y
conda activate asrft && pip install -U qwen-asr datasets && conda activate lifemate
```

Máy dùng chung: cổng 8000 đã có dịch vụ khác, nên mọi lệnh dùng cổng 8011 qua `scripts/serve.sh`. Không dùng `pkill -f` với chuỗi ngắn.

---

## Phần 1 — ASR

### 1.1 Transcript của ASR gốc (mốc so sánh)

```bash
conda activate lifemate
lifemate transcribe --split train --asr qwen --asr-model models/base/Qwen3-ASR-0.6B --out outputs/asr0_train.jsonl
```

### 1.2 Dữ liệu fine-tune (chừa tập val)

```bash
lifemate asr-manifest --split train --out-dir outputs/asr_ft
# → outputs/asr_ft/train.jsonl (4.158 câu), val.jsonl (878 câu), val_ids.txt (274 mẫu)
```

### 1.3 Fine-tune

```bash
conda activate asrft
python third_party/Qwen3-ASR/finetuning/qwen3_asr_sft.py \
  --model_path models/base/Qwen3-ASR-0.6B \
  --train_file outputs/asr_ft/train.jsonl --eval_file outputs/asr_ft/val.jsonl \
  --output_dir checkpoints/asr-ft --batch_size 8 --grad_acc 4 --lr 1e-5 --epochs 3 \
  --save_steps 200 --save_total_limit 3
```

### 1.4 Đo trên val

```bash
conda activate lifemate
CKPT=checkpoints/asr-ft/checkpoint-XXX                       # checkpoint cuối, hoặc có eval loss thấp nhất
cp -n models/base/Qwen3-ASR-0.6B/*.json "$CKPT"/             # bổ sung file cấu hình nếu thiếu, không ghi đè
lifemate transcribe --split train --asr qwen --asr-model "$CKPT" --out outputs/asr1_train.jsonl
lifemate asr-eval --split train --transcripts outputs/asr0_train.jsonl --ids-file outputs/asr_ft/val_ids.txt
lifemate asr-eval --split train --transcripts outputs/asr1_train.jsonl --ids-file outputs/asr_ft/val_ids.txt
```

**Chỉ tin số đo trên val.** Trên câu train, ASR đã học thuộc nên WER đẹp giả. Nếu WER giảm ít, thử tăng `--epochs` hoặc đổi `--lr`, rồi đo lại.

---

## Phần 2 — LLM

### 2.1 Dữ liệu SFT và train

Dữ liệu SFT chỉ dùng transcript gold (`MIX_ASR=0`, mặc định), và chừa cùng tập val.

```bash
kill $VLLM_PID 2>/dev/null
tmux new -s train
bash scripts/train_sft_24gb.sh                     # → outputs/sft_apps_gold/, checkpoints/sft-v1
```

Nếu bị OOM, chạy lại với `MAX_LENGTH=8192 bash scripts/train_sft_24gb.sh`.

### 2.2 Đo trên val, transcript gold (mốc: 41,76)

```bash
source scripts/serve.sh models/base/Qwen3.5-4B base 8011 0.85 \
  --enable-lora --max-lora-rank 64 --lora-modules sft-v1=checkpoints/sft-v1
lifemate infer --split train --backend openai --model sft-v1 --asr oracle \
  --ids-file outputs/asr_ft/val_ids.txt --exclude-ids outputs/asr_ft/val_ids.txt --fewshot-k 3 \
  --out outputs/sft-v1_val.jsonl --trace outputs/sft-v1_val.trace.jsonl
lifemate score --split train --submission outputs/sft-v1_val.jsonl --ids-file outputs/asr_ft/val_ids.txt \
  --report outputs/sft-v1_val.scores.jsonl
lifemate inspect --split train --report outputs/sft-v1_val.scores.jsonl --show 15 > outputs/sft-v1_val.inspect.txt
```

Nếu vLLM không nạp được LoRA, gộp adapter vào model gốc rồi serve bản đã gộp:
```bash
python -m lifemate.train.merge_lora --base models/base/Qwen3.5-4B --adapter checkpoints/sft-v1 --out models/sft-v1
source scripts/serve.sh models/sft-v1 sft-v1 8011 0.85
```

---

## Phần 3 — Toàn hệ thống trên val, rồi nộp public

### 3.1 ASR fine-tune → LLM SFT trên val

Server từ bước 2.2 vẫn đang chạy:

```bash
lifemate infer --split train --backend openai --model sft-v1 --transcripts outputs/asr1_train.jsonl \
  --ids-file outputs/asr_ft/val_ids.txt --exclude-ids outputs/asr_ft/val_ids.txt --fewshot-k 3 \
  --out outputs/e2e_v1_val.jsonl --trace outputs/e2e_v1_val.trace.jsonl
lifemate score --split train --submission outputs/e2e_v1_val.jsonl --ids-file outputs/asr_ft/val_ids.txt
kill $VLLM_PID
```

Đây là con số gần với điểm public nhất. Khoảng cách tới điểm ở 2.2 (transcript gold) là phần ASR còn làm mất.

### 3.2 Nộp public

```bash
ASR_MODEL=checkpoints/asr-ft/checkpoint-XXX LORA=checkpoints/sft-v1 bash scripts/predict_public.sh
```

Script chạy ASR → vLLM → agent → kiểm tra định dạng. Upload `outputs/public_test/submission_sft-v1.jsonl` khi thấy `submission ready … (634 lines)`.

---

## Phần 4 — Bản cuối: train trên toàn bộ train

Dùng đúng cấu hình đã cho kết quả tốt nhất ở Phần 1–3.

```bash
# ASR trên cả 5.036 câu
lifemate asr-manifest --split train --out-dir outputs/asr_full --holdout none
conda activate asrft
python third_party/Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path models/base/Qwen3-ASR-0.6B \
  --train_file outputs/asr_full/train.jsonl --output_dir checkpoints/asr-full \
  --batch_size 8 --grad_acc 4 --lr 1e-5 --epochs 3 --save_steps 200 --save_total_limit 2
conda activate lifemate
cp -n models/base/Qwen3-ASR-0.6B/*.json checkpoints/asr-full/checkpoint-YYY/

# LLM trên cả 1.545 mẫu
HOLDOUT=none bash scripts/train_sft_24gb.sh        # → checkpoints/sft-full

# Nộp public
ASR_MODEL=checkpoints/asr-full/checkpoint-YYY LORA=checkpoints/sft-full bash scripts/predict_public.sh
```

Các bản này đã học cả tập val, nên chỉ đánh giá được qua public leaderboard.

---

## Phần 5 — Docker (hạn 23:59 ngày 16/11)

```bash
# 1. Gộp LoRA, chuẩn bị model và dữ liệu few-shot
python -m lifemate.train.merge_lora --base models/base/Qwen3.5-4B --adapter checkpoints/sft-full --out models/llm
cp -r checkpoints/asr-full/checkpoint-YYY models/asr
mkdir -p assets/train && cp data/train/train.jsonl assets/train/

# 2. Kiểm tra tổng tham số dưới 6B và LLM dưới 5B
python - <<'PY'
import torch, transformers
for p in ["models/llm", "models/asr"]:
    try:
        m = transformers.AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16)
    except Exception:
        m = transformers.AutoModel.from_pretrained(p, dtype=torch.bfloat16, trust_remote_code=True)
    total = sum(x.numel() for x in m.parameters())
    vision = sum(x.numel() for n, x in m.named_parameters() if "visual" in n)
    print(p, f"total {total/1e9:.2f}B, vision {vision/1e9:.2f}B")
PY

# 3. Build và chạy thử offline trên public_test
docker build -f docker/Dockerfile -t lifemate:final .
time docker run --rm --gpus all --network none \
  -v $(realpath data)/public_test:/data -v $PWD/outputs/docker:/app/result lifemate:final
```

Trước khi nộp, kiểm tra:
- `validate` báo `OK`.
- Chạy 2 lần cho cùng một kết quả.
- Trung bình dưới 30 giây mỗi mẫu.
- Điểm public của `outputs/docker/submission.jsonl` khớp với bản nộp ở Phần 4.

Nộp qua form "Solution Submission", tối đa 2 lần, gửi kèm checksum và link Google Drive.

## Phần 6 — Private test (20–23/11)

```bash
docker run --rm --gpus all --network none \
  -v /đường/dẫn/private_test:/data -v $PWD/outputs/private:/app/result lifemate:final
```

Nộp `outputs/private/submission.jsonl` trước 23:59 ngày 23/11. Không sửa gì giữa Docker đã nộp và lần chạy này.

---

## Nhật ký thí nghiệm

| Ngày | Cấu hình | WER val | Val (gold) | Val (ASR) | Public | Ghi chú |
|---|---|---|---|---|---|---|
| 27/10 | ASR gốc + LLM gốc, prompt v1 | 36,7%* | 36,10 | | | *WER trên toàn train |
| 27/10 | ASR gốc + LLM gốc, prompt v2 | | 41,76 | | | |
| | ASR fine-tune + LLM gốc | | | | | |
| | ASR fine-tune + sft-v1 | | | | | |
| | asr-full + sft-full | | | | | |
