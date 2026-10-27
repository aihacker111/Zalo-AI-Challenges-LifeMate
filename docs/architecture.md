# Kiến trúc

## Luồng một mẫu

1. `asr`: chuyển mọi lượt user (history `_u1.wav…` và lượt cuối) thành text, lưu `TranscriptStore` theo đường dẫn audio.
2. `PromptBuilder` dựng 2 message:
   - **system**: chính sách và danh sách tool đã cài, dạng chữ ký gọn (`render_tool`), khoảng 1/3 độ dài JSON schema gốc.
   - **user**: thời điểm và bảng ngày, vị trí, hồ sơ người dùng, few-shot truy hồi, history (call và kết quả đã rút gọn), yêu cầu cần xử lý.
3. Mỗi bước model sinh **một** JSON `{"name", "arguments"}` hoặc `{"name": "finish"}`. Khi dùng vLLM, đầu ra bị ràng buộc bởi `action_schema` (anyOf các tool đã cài cộng `finish`).
4. `Guard.review` chạy trước `session.call`:
   1. tool phải thuộc app đã cài;
   2. sửa tham số: bỏ tham số lạ và tham số tuỳ chọn rỗng, ép kiểu (`"2.000.000"` → `2000000`), chuẩn hoá date-time về `YYYY-MM-DDTHH:MM:SS+07:00`;
   3. lệnh đọc trùng → trả lại kết quả cũ, không gọi môi trường;
   4. tham số `x-ref` phải là ID đã xuất hiện trong history hoặc kết quả tool;
   5. `session.validate()` (không ghi log) → lỗi được trả cho model sửa.
5. Kết quả được rút gọn (`render.compact_result`) và đưa lại cho model. Vòng lặp dừng ở `finish`, sau `sys.ask_missing_info`, khi hết `max_steps`, hoặc khi hết thời gian.
6. Dòng nộp gồm `session.get_tool_calls()` và `session.get_state_hash()`.

## Vì sao từng thành phần tồn tại

| Thành phần | Lỗi metric mà nó ngăn |
|---|---|
| Kiểm tra x-ref | Ghi bằng ID đoán → sai state → mẫu 0 điểm |
| Chặn đọc trùng | Precision giảm (ví dụ 3/4 → F1 0,857) |
| `validate()` trước khi gọi | Call lỗi schema vẫn tính vào U |
| Bảng ngày trong prompt | Sai ngày ở "mai", "thứ Sáu" → sai tham số non-fuzzy |
| Dừng sau `ask_missing_info` | Gold luôn kết thúc lượt ở đó |
| Render dùng chung cho SFT và inference | Lệch phân phối giữa train và chạy thật |

## Đánh giá

- `Scorer` replay bằng `zaic_tools` của chính split, nên hai phiên bản `zaic_tools` có thể tồn tại song song (`data.load_zaic_tools`).
- `eval.splits.group_kfold` nhóm các câu diễn đạt lại của cùng kịch bản; `leave_apps_out` giữ lại app không cho huấn luyện để mô phỏng private test.
- `eval.report.taxonomy` phân loại lỗi để biết nên sửa prompt, guard, dữ liệu hay ASR.
