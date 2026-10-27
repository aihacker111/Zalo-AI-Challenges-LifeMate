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

Các lệnh dưới đây dùng `./data` làm thư mục dữ liệu nên không cần `--data`. Đã chạy được với vLLM 0.19.1 trên một GPU 24 GB.

### 1. Kiểm tra pipeline (không cần GPU)

```bash
make test                                                         # 34 test
lifemate stats --split train
lifemate infer --split train --backend oracle --asr oracle --out outputs/oracle.jsonl
lifemate score --split train --submission outputs/oracle.jsonl      # phải đạt ~99,99
```

### 2. Khởi động server vLLM

```bash
source scripts/serve.sh models/base/Qwen3.5-4B base 8011 0.85    # model, tên phục vụ, cổng, tỉ lệ VRAM
```

`serve.sh` làm bốn việc:
- kiểm tra cổng còn trống trước khi chạy;
- chạy vLLM nền, ghi log vào `outputs/vllm_<cổng>.log`;
- chờ tới khi server sẵn sàng (dòng `ready: …`), hoặc in đuôi log rồi dừng nếu vLLM chết lúc khởi động;
- đặt hai biến `BASE_URL` và `VLLM_PID`.

Phải chạy bằng `source`, không dùng `bash`, để hai biến này còn lại trong shell. Tắt server bằng `kill $VLLM_PID`.

`--base-url` của `infer` mặc định lấy `$LIFEMATE_BASE_URL`, rồi `$BASE_URL`, cuối cùng là `http://localhost:8011/v1`. Ở terminal khác với terminal đã chạy `serve.sh`, nó vẫn trỏ đúng tới cổng 8011.

### 3. Chạy agent và chấm điểm

```bash
# 20 mẫu thử
lifemate infer --split train --backend openai --model base --base-url $BASE_URL \
  --asr oracle --limit 20 --out outputs/smoke.jsonl --trace outputs/smoke.trace.jsonl
lifemate score --split train --submission outputs/smoke.jsonl --only-submitted

# Baseline trên tập val leave-apps-out (~274 mẫu)
lifemate build-sft --asr oracle --holdout apps --n-apps 10 --out-dir outputs/sft
lifemate infer --split train --backend openai --model base --base-url $BASE_URL --asr oracle \
  --ids-file outputs/sft/val_ids.txt --exclude-ids outputs/sft/val_ids.txt --fewshot-k 3 \
  --out outputs/base_k3.jsonl --trace outputs/base_k3.trace.jsonl
lifemate score --split train --submission outputs/base_k3.jsonl --ids-file outputs/sft/val_ids.txt
```

Khi chấm trên val, luôn dùng `--exclude-ids`. Thiếu nó, few-shot sẽ lấy chính các câu val làm ví dụ và điểm cao giả.

### 4. Tạo bài nộp public_test

Một lệnh làm cả bốn việc: chạy ASR mọi lượt user (kết quả được cache lại), khởi động vLLM, chạy agent, kiểm tra định dạng file. Trước khi chạy, tắt các server vLLM khác (`kill $VLLM_PID`) để ASR và vLLM có đủ VRAM.

```bash
bash scripts/predict_public.sh                                  # model gốc Qwen3.5-4B
LORA=checkpoints/sft-v1 bash scripts/predict_public.sh          # model gốc + LoRA adapter
MODEL=models/sft-v1 bash scripts/predict_public.sh              # checkpoint đã gộp LoRA
```

File nộp nằm ở `outputs/public_test/submission_<tên model>.jsonl`. Các tuỳ chọn khác đặt qua biến môi trường: `PORT`, `MEM`, `FEWSHOT`, `ASR_MODEL`, `SPLIT`.

### Các lệnh

| Lệnh | Việc | Tuỳ chọn đáng chú ý |
|---|---|---|
| `stats` | Thống kê split | |
| `transcribe` | ASR mọi lượt user (cả history), lưu JSON Lines | `--asr qwen\|gipformer\|chunkformer\|oracle`, `--asr-model`, `--ids-file` |
| `infer` | Chạy agent, ghi submission | `--backend vllm\|openai\|oracle`, `--base-url`, `--fewshot-k`, `--ids-file`, `--exclude-ids`, `--limit`, `--trace`, `--resume`, `--no-constrained`, `--config` |
| `score` | Chấm trên split có nhãn, kèm phân loại lỗi | `--ids-file`, `--only-submitted`, `--report` |
| `inspect` | Đọc file `--report` của `score`: call và tham số lệch nhiều nhất, kèm ví dụ | `--show`, `--label` |
| `validate` | Kiểm tra schema file nộp bằng `zaic_tools` của split | |
| `build-sft` | Dựng dữ liệu SFT và file `val_ids.txt` | `--holdout apps\|fold\|none`, `--augment`, `--fewshot-k`, `--transcripts` |
| `asr-eval` | WER, WER bỏ dấu, tỉ lệ sai số so với gold | `--ids-file` |
| `asr-manifest` | Manifest fine-tune Qwen3-ASR | `--val-ids` |

Backend LLM:
- `openai`: server OpenAI-compatible (`scripts/serve.sh`), dùng khi phát triển.
- `vllm`: vLLM offline trong cùng tiến trình, dùng trong Docker khi nộp bài.
- `oracle`: phát lại các call gold, dùng để kiểm tra pipeline.

Với backend `openai`, `infer` kiểm tra server và tên model trước khi chạy, và dừng sau 5 mẫu lỗi liên tiếp (exit code 2). Nhờ vậy một server hỏng không bao giờ âm thầm sinh ra file nộp rỗng.

### Scripts

| Script | Việc |
|---|---|
| `scripts/serve.sh` | Khởi động vLLM, chờ sẵn sàng, export `BASE_URL`/`VLLM_PID` |
| `scripts/asr_benchmark.sh` | So sánh các engine ASR trên 878 câu val (WER, sai số, thời gian) |
| `scripts/predict_public.sh` | ASR → vLLM → agent → validate, ra file nộp cho public_test (hoặc `SPLIT=…`) |
| `scripts/baseline.sh` | Baseline k=0 và k=3 trên train (transcript gold) |
| `scripts/train_sft_24gb.sh` | Từ dữ liệu thô tới LoRA: ASR train → dữ liệu SFT (gold + ASR) → train. `HOLDOUT=none` để train trên toàn bộ train |

## Quy trình đề xuất

Hướng dẫn chi tiết từng lệnh, mốc thời gian và bảng nhật ký nằm trong [docs/training.md](docs/training.md).

1. **Baseline không train.** Đo điểm zero-shot và few-shot (k = 3) trên tập val leave-apps-out với transcript gold (mục 3 ở trên). Lỗi được phân loại thành `extra_read`, `missing_read`, `write_mismatch`, `ask_mismatch`, `oos_false_action`…
2. **Đo ASR.** Chạy `transcribe` trên train rồi `asr-eval`; lưu ý riêng tỉ lệ sai số (tiền, giờ, số điện thoại).
3. **LoRA SFT** nếu baseline còn cách xa mục tiêu: `bash scripts/train_sft_24gb.sh` (giữ tập val, ra `checkpoints/sft-v1`), sau đó `HOLDOUT=none bash scripts/train_sft_24gb.sh` (toàn bộ train, ra `checkpoints/sft-full`). Dữ liệu trộn input transcript gold và transcript ASR, để model quen với lỗi nhận dạng.
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

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `infer` báo `HTTP 404`, hoặc `curl localhost:8000/v1/models` trả `{"detail":"Invalid API key"}` | Cổng 8000 đang do một dịch vụ khác giữ (thường gặp trên máy dùng chung) | Dùng cổng khác: `source scripts/serve.sh … 8011 …`. **Đừng tắt tiến trình của người khác.** |
| `serve.sh` báo `port … is already in use` | Còn một server cũ ở cổng đó | `curl localhost:<cổng>/v1/models`: nếu có model của mình thì `export BASE_URL=http://localhost:<cổng>/v1`; nếu không, đổi cổng |
| `argument --… : expected one argument` | Một biến shell (`$BASE_URL`, `$DATA`, `$VAL`…) rỗng trong terminal hiện tại | Bỏ tham số đó (đều có giá trị mặc định), hoặc ghi thẳng giá trị, ví dụ `--ids-file outputs/sft/val_ids.txt` |
| `Connection refused` | Server chưa chạy hoặc đã chết | Xem `tail -60 outputs/vllm_<cổng>.log`, rồi khởi động lại bằng `serve.sh` |
| `Free memory on device … is less than desired` | GPU đang được người khác dùng | Xem `nvidia-smi`, hạ tham số VRAM của `serve.sh` (ví dụ `0.6`) |
| `EngineDeadError` ở request đầu tiên | Lỗi trong tiến trình EngineCore | `grep -v APIServer outputs/vllm_<cổng>.log \| grep -A40 Traceback`; thử `--no-constrained` để loại trừ bộ ràng buộc JSON |
| `lifemate` không có tuỳ chọn mới, hoặc vẫn đòi `--data` | Code trên máy cũ | `git pull` (cài bằng `pip install -e .` nên không cần cài lại) |
| Build `causal-conv1d` lỗi `nvcc was not found` | Gói tuỳ chọn cần CUDA toolkit | Bỏ qua, hoặc xem [docs/training.md](docs/training.md) |

Trên máy dùng chung, **không dùng `pkill -f` với chuỗi ngắn**: nó có thể giết nhầm tiến trình khác. Tắt server của mình bằng `kill $VLLM_PID`.

## Docker (nộp bài cuối)

```bash
python -m lifemate.train.merge_lora --base Qwen/Qwen3.5-4B --adapter checkpoints/sft-lora --out models/llm
cp -r <asr-checkpoint> models/asr
mkdir -p assets/train && cp data/train/train.jsonl assets/train/
docker build -f docker/Dockerfile -t lifemate:final .
docker run --rm --gpus all --network none -v $(realpath data)/public_test:/data -v $PWD/outputs/docker:/app/result lifemate:final
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
├── text.py, timeutil.py, config.py
├── pipeline.py        # chạy agent trên một split, ghi file nộp, dừng khi lỗi liên tiếp
├── cli.py             # lệnh `lifemate`
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
tests/                 # unittest; các test cần dữ liệu tự bỏ qua khi không tìm thấy dữ liệu
scripts/               # serve.sh, baseline.sh, train_sft_24gb.sh
configs/               # agent.yaml, sft_lora_24gb.yaml
docker/                # Dockerfile + entrypoint cho bài nộp cuối
docs/                  # architecture.md, training.md
```

## Test

```bash
make test                                 # hoặc: LIFEMATE_DATA=... python -m unittest discover -s tests -t .
```

Có 34 test, gồm: metric (gold = 100, rỗng = tỉ lệ OOS, thừa/thiếu một lệnh đọc), guard, parser, agent oracle trên toàn bộ train, tính trùng khớp giữa SFT và inference, và augment.

## Ghi chú về dữ liệu

- **ASR là nút thắt lớn nhất.** Qwen3-ASR-0.6B gốc có WER khoảng 37% trên audio train (giọng thật, nhiều vùng miền), và nhiều câu mất hẳn nội dung, ví dụ "chuyển mẹ ba triệu liền đi" thành "chỉ mẹ ban chiều lên đi". Vì vậy dữ liệu SFT mặc định chỉ dùng transcript gold (`MIX_ASR=0`), và cần fine-tune ASR trên audio train (xem [docs/training.md](docs/training.md), mục 2).

- `train_1696` có nhãn sai: gold gửi tin tới `chat_6142`, một chat không tool nào trả về. Guard chặn call này (đúng chính sách) nên mẫu này mất điểm trong kiểm tra oracle.
- 11 app của public_test không có trong train, nên private test nhiều khả năng cũng có app mới. Số đo quan trọng nhất là điểm trên tập leave-apps-out (`build-sft --holdout apps`).

## Giấy phép

Mã nguồn theo [MIT](LICENSE). Dữ liệu và `zaic_tools` thuộc BTC Zalo AI Challenge, không nằm trong repo và không được phân phối lại.
