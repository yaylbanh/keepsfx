# 🎬 keepsfx

Tách **giọng + nhạc** ra khỏi video, **chỉ giữ lại hiệu ứng (SFX)** → xuất MP4 (audio = SFX)
để lồng tiếng Việt lên, bỏ nhạc gốc (né bản quyền).

Dùng **BandIt** — model SOTA tách 3 luồng *speech / music / effects* (Cinematic Audio Source Separation).

> ⚠️ **License:** checkpoint BandIt là **CC BY-NC 4.0 (phi thương mại)**. Dùng cho mục đích
> không kiếm tiền thì thoải mái; nếu monetize là rủi ro về license.
> ⚠️ Tách SFX không hoàn hảo: có thể còn rớt nhạc nhẹ / mất SFX nhỏ.

---

## 🚀 Chạy trên Colab (1 thao tác)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaylbanh/keepsfx/blob/main/run_colab.ipynb)

1. Bấm badge **Open in Colab**.
2. `Runtime → Change runtime type → GPU T4 → Save`.
3. Bấm ▶ chạy ô code → **Accept** Google Drive.
4. Lần đầu tải model ~775MB vào `MyDrive/keepsfx_models/` (lần sau khỏi tải).
5. Mở link `*.gradio.live` → upload video → **Tách & giữ SFX**.
6. Tải MP4 (audio = SFX) — cũng lưu ở `MyDrive/keepsfx_output/`.

---

## Luồng xử lý

```
MP4 → ffmpeg tách audio 44.1kHz → BandIt (speech/music/effects) → giữ effects
    → ffmpeg ghép: video gốc + effects → MP4 (audio = SFX)
```

## Cấu trúc

| File | Vai trò |
|------|---------|
| `app.py` | Gradio: tách SFX + ghép video. Gọi BandIt qua subprocess |
| `requirements.txt` | gradio + deps BandIt (KHÔNG cài lại torch trên Colab) |
| `run_colab.ipynb` | 1 cell: clone + tải model vào Drive + chạy |

## Ghi chú

- Model BandIt lấy từ [Zenodo](https://zenodo.org/records/10160698) (`dnr-3s-bark48-l1snr`).
- Code BandIt: [kwatcharasupat/bandit](https://github.com/kwatcharasupat/bandit).
- Đây là code nghiên cứu → có thể cần chỉnh dependency theo môi trường.
