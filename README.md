# LifeMate — Voice-to-Action Agent

Agent nhận yêu cầu tiếng Việt bằng giọng nói và thực hiện chúng bằng tool call, xây dựng cho bảng **LifeMate** của
[Zalo AI Challenge 2026](https://challenge.zalo.ai/portal/lifemate).

Kiến trúc là một pipeline ASR → LLM agent, chạy hoàn toàn offline trong giới hạn của cuộc thi: dưới 6B tham số, LLM dưới 5B, context tối đa 16K, 30 giây mỗi mẫu trên 1×A30.

```
audio các lượt user ──► ASR (Qwen3-ASR-0.6B) ──┐
context + history ─────────────────────────────┼─► prompt ─► LLM (Qwen3.5-4B) ─► guard ─► ToolEnv (zaic_tools)
catalog các app đã cài ────────────────────────┘                ▲                                  │
                                                                └──────── kết quả rút gọn ─────────┘
```

## Điểm nổi bật

- **Scorer offline khớp metric chính thức.** Replay bằng đúng `zaic_tools` của từng split, cổng state-hash, ghép call bằng maximum matching, token-F1 cho tham số fuzzy. Gold so với chính nó đạt 100,00; submit toàn rỗng đạt đúng tỉ lệ mẫu ngoài phạm vi.
- **Guard xác định** giữa model và môi trường. Mọi call đều bị tính điểm, nên guard chặn trước các call sai schema (kiểm tra bằng `validate()`, vốn không ghi log), các lệnh đọc lặp, và các lệnh ghi dùng ID chưa từng đọc. Guard cũng tự sửa kiểu dữ liệu và định dạng giờ.
- **Sanity check end-to-end.** Cho agent phát lại các call gold qua toàn bộ prompt → guard → môi trường đạt **99,99** trên train. Điểm duy nhất mất là một nhãn lỗi đã được xác định.
- **Dữ liệu huấn luyện trùng khớp với inference.** Hội thoại SFT được dựng bằng chính code dựng prompt lúc chạy; có test kiểm tra hai bên giống nhau từng ký tự.
- **Sẵn sàng cho app mới.** Prompt chỉ dựa vào schema của app đã cài. Augment đổi tên app/tool/tham số dạy model đọc schema thay vì học thuộc tên. Có tập validation leave-apps-out.
- **Vừa 1 GPU 24 GB.** LoRA SFT và rejection-sampling fine-tuning (RFT) thay cho full fine-tune và RL.
- **Core chỉ dùng thư viện chuẩn Python.** Torch, vLLM và qwen-asr chỉ cần cho các lệnh dùng GPU.

## Cài đặt

```bash
git clone https://github.com/aihacker111/Zalo-AI-Challenges-LifeMate.git && cd Zalo-AI-Challenges-LifeMate
python -m pip install -e ".[dev,yaml]"      # core + test/lint
python -m pip install -e ".[vllm,asr]"      # inference trên GPU
python -m pip install -e ".[train]"         # huấn luyện LoRA
```

Python ≥ 3.10. Nếu không cài package, có thể chạy trực tiếp: `PYTHONPATH=src python -m lifemate ...`.

### Dữ liệu

Gói dữ liệu của BTC ở đâu cũng được. Tạo một symlink `data` trong repo trỏ tới nó (đã nằm trong `.gitignore`), hoặc đặt biến `LIFEMATE_DATA`. **Không commit dữ liệu**, vì quy định cấm phát tán.

```bash
ln -s ~/Downloads/train+public_test data      # hoặc: export LIFEMATE_DATA=~/Downloads/train+public_test
```

```
train+public_test/
├── train/             # train.jsonl, train_tools_catalog.json, train_environment/, audio/, zaic_tools/
└── public_test/
```

`--data` mặc định là `$LIFEMATE_DATA` hoặc `./data`. Tham số này nhận thư mục chứa các split (kèm `--split train`) hoặc chính thư mục của một split, ví dụ `/data` trong Docker.

## Sử dụng nhanh

```bash
DATA=data

lifemate stats --data $DATA --split train

# Kiểm tra pipeline: agent phát lại gold phải đạt ~100
lifemate infer --data $DATA --split train --backend oracle --asr oracle --out outputs/oracle.jsonl
lifemate score --data $DATA --split train --submission outputs/oracle.jsonl

# ASR cho public_test (một lần, lưu cache), rồi chạy agent bằng vLLM
lifemate transcribe --data $DATA --split public_test --asr qwen --asr-model Qwen/Qwen3-ASR-0.6B --out outputs/asr_public.jsonl
lifemate infer --data $DATA --split public_test --backend vllm --model Qwen/Qwen3.5-4B \
  --transcripts outputs/asr_public.jsonl --config configs/agent.yaml --out outputs/submission.jsonl --trace outputs/trace.jsonl
lifemate validate --data $DATA --split public_test --submission outputs/submission.jsonl
```

| Lệnh | Việc |
|---|---|
| `stats` | Thống kê split |
| `transcribe` | ASR mọi lượt user (cả history), lưu JSON Lines |
| `infer` | Chạy agent, ghi submission; hỗ trợ `--resume`, `--trace`, `--limit`, `--ids` |
| `score` | Chấm trên split có nhãn, kèm phân loại lỗi; `--report` ghi chi tiết từng mẫu |
| `validate` | Kiểm tra schema file nộp bằng `zaic_tools` của split |
| `build-sft` | Dựng dữ liệu SFT: few-shot, augment đổi tên, holdout theo app hoặc theo kịch bản |
| `asr-eval` | WER, WER bỏ dấu, tỉ lệ sai số của transcript so với gold (train) |
| `asr-manifest` | Manifest fine-tune ASR từ audio train |

Backend LLM: `vllm` (offline, dùng khi nộp bài), `openai` (bất kỳ server OpenAI-compatible nào, ví dụ `vllm serve`, khi phát triển) và `oracle` (phát lại gold).

## Quy trình đề xuất

Hướng dẫn chi tiết từng lệnh, mốc thời gian và bảng nhật ký nằm trong [docs/training.md](docs/training.md).

1. **Baseline không train.** Chạy `scripts/baseline.sh` để có điểm zero-shot và few-shot (k = 3) trên train với transcript gold. Lỗi được phân loại thành `extra_read`, `missing_read`, `write_mismatch`, `ask_mismatch`, `oos_false_action`…
2. **Đo ASR.** Chạy `transcribe` trên train rồi `asr-eval`; lưu ý riêng tỉ lệ sai số (tiền, giờ, số điện thoại).
3. **LoRA SFT** nếu baseline còn cách xa mục tiêu: `scripts/train_sft_24gb.sh`. Dữ liệu dựng với input là **transcript ASR** để model quen với lỗi nhận dạng.
4. **RFT.** `python -m lifemate.train.rft` lấy mẫu vài quỹ đạo cho mỗi task, chỉ giữ quỹ đạo đạt điểm tuyệt đối, rồi SFT thêm một vòng.
5. **Đóng gói.** `python -m lifemate.train.merge_lora` gộp adapter vào model gốc, sau đó build Docker.

### Huấn luyện trên 1 GPU 24 GB

| Việc | VRAM | Ghi chú |
|---|---|---|
| Inference Qwen3.5-4B bf16 (vLLM) | ~9 GB + KV cache | Cùng cỡ với A30 của máy chấm |
| LoRA r=64, gradient checkpointing, seq 12K | ~16–22 GB | `configs/sft_lora_24gb.yaml` |
| Full fine-tune 4B | ~65 GB | Không vừa; thuê GPU 80 GB nếu cần |
| Fine-tune Qwen3-ASR-0.6B | ~12–16 GB | Dùng recipe của Qwen3-ASR với manifest từ `asr-manifest` |

Qwen3.5 dùng kiến trúc lai Gated DeltaNet. Cài thêm `flash-linear-attention` (không cần nvcc) để train nhanh hơn. `causal-conv1d` là tuỳ chọn và cần CUDA toolkit; xem [docs/training.md](docs/training.md#0-môi-trường-ngày-1).

## Docker (nộp bài cuối)

```bash
python -m lifemate.train.merge_lora --base Qwen/Qwen3.5-4B --adapter checkpoints/sft-lora --out models/llm
cp -r <asr-checkpoint> models/asr
mkdir -p assets/train && cp data/train/train.jsonl assets/train/
docker build -f docker/Dockerfile -t lifemate:final .
docker run --gpus all --network none -v $(realpath data)/public_test:/data lifemate:final
```

Container đọc `/data` và ghi `result/submission.jsonl`. ASR chạy trước cho mọi lượt rồi giải phóng GPU, sau đó LLM mới được nạp.

## Tuân thủ quy định

- [x] Chỉ đọc và ghi dữ liệu app qua `zaic_tools`; mỗi mẫu đúng **một** session; không đọc `state.json`.
- [x] Mọi call trên session đều nằm trong file nộp (`session.get_tool_calls()`).
- [x] Giải mã greedy và cố định seed.
- [x] Chạy offline (`HF_HUB_OFFLINE=1`, `--network none`).
- [ ] Kiểm tra tổng tham số dưới 6B và LLM dưới 5B bằng cách đếm `numel()` trên checkpoint cuối. Với Qwen3.5-4B cần xác nhận với BTC về vision tower.
- [ ] Hỏi BTC trước khi sinh dữ liệu tổng hợp dựa trên catalog của public_test.
- [ ] Không gán nhãn public_test hay private_test dưới bất kỳ hình thức nào.

## Cấu trúc repo

```
src/lifemate/
├── data.py            # Split, Sample, nạp zaic_tools riêng cho từng split
├── catalog.py         # ToolSpec, render schema gọn cho prompt, JSON schema cho constrained decoding
├── text.py, timeutil.py, config.py, pipeline.py, cli.py
├── agent/
│   ├── policy.py      # system prompt: định dạng hành động + chính sách gán nhãn quan sát từ train
│   ├── prompt.py      # bối cảnh, bảng ngày, tool, few-shot, history
│   ├── render.py      # dạng chuẩn của action/kết quả (dùng chung cho inference và SFT)
│   ├── guard.py       # validate, chặn đọc lặp, kiểm tra x-ref, ép kiểu
│   ├── parser.py      # văn bản model → Action
│   ├── llm.py         # VLLMBackend, OpenAICompatBackend, OracleBackend, ScriptedBackend
│   ├── fewshot.py     # truy hồi TF-IDF các quỹ đạo train
│   └── loop.py        # Agent: một session mỗi mẫu, một bước mỗi tool call
├── asr/               # TranscriptStore, Qwen3-ASR, WER và tỉ lệ sai số
├── eval/              # scorer (metric chính thức), splits (group k-fold, leave-apps-out), report
└── train/             # sft_data, augment, sft_lora, rft, merge_lora, asr_data
tests/                 # unittest; các test cần dữ liệu tự bỏ qua khi không có LIFEMATE_DATA
```

## Test

```bash
make test                                 # hoặc: LIFEMATE_DATA=... python -m unittest discover -s tests -t .
```

Có 34 test, gồm: metric (gold = 100, rỗng = tỉ lệ OOS, thừa/thiếu một lệnh đọc), guard, parser, agent oracle trên toàn bộ train, tính trùng khớp giữa SFT và inference, và augment.

## Ghi chú về dữ liệu

- `train_1696` có nhãn sai: gold gửi tin tới `chat_6142`, một chat không tool nào trả về. Guard chặn call này (đúng chính sách) nên mẫu này mất điểm trong kiểm tra oracle.
- 11 app của public_test không có trong train, nên private test nhiều khả năng cũng có app mới. Số đo quan trọng nhất là điểm trên tập leave-apps-out (`build-sft --holdout apps`).

## Giấy phép

Mã nguồn theo [MIT](LICENSE). Dữ liệu và `zaic_tools` thuộc BTC Zalo AI Challenge, không nằm trong repo và không được phân phối lại.
