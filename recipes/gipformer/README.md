# Fine-tune gipformer (icefall)

[gipformer1.5-68M-rnnt](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt) là Zipformer RNN-T train bằng [icefall](https://github.com/k2-fsa/icefall). Repo gốc không có script train. Thư mục này nối dữ liệu LifeMate vào `egs/librispeech/ASR/zipformer/finetune.py` của icefall, và không sửa gì trong icefall.

| File | Việc |
|---|---|
| `convert_checkpoint.py` | `model.pt` (state dict) → `{"model": ...}` cho `finetune.py`, kiểm tra key khớp kiến trúc mặc định |
| `prepare_cuts.py` | Manifest của `lifemate asr-manifest` → lhotse cuts 16 kHz + fbank 80 chiều. Text đổi sang văn nói (`lifemate.asr.normalize_vi`). Có speed perturb |
| `train_gipformer.py` | Chạy `finetune.py` với data module trỏ vào các cut trên. Validate mỗi `VALID_INTERVAL` batch |

Cách chạy trọn bộ: `scripts/gipformer_setup.sh` rồi `scripts/gipformer_finetune.sh`. Xem [docs/training.md](../../docs/training.md).
