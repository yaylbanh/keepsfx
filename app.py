# -*- coding: utf-8 -*-
"""
keepsfx - Tach giong + nhac ra khoi video, CHI GIU HIEU UNG (SFX).
Dung BandIt (SOTA cinematic audio separation: speech/music/effects).
Input: MP4/video -> Output: MP4 giu nguyen hinh, audio = chi SFX (+ file sfx.wav).

Muc dich: lam nen de long tieng Viet, bo nhac goc (ne ban quyen).
Model BandIt: license CC BY-NC (phi thuong mai).
"""

import os
import glob
import shutil
import tempfile
import subprocess

import gradio as gr

# ====== CAU HINH (chinh qua bien moi truong trong notebook/bat) ======
BANDIT_DIR = os.environ.get("BANDIT_DIR", "/content/bandit")
CKPT_PATH = os.environ.get("BANDIT_CKPT", "/content/drive/MyDrive/keepsfx_models/ckpt/dnr-3s-bark48-l1snr.ckpt")
INPUT_DIR = os.environ.get("KEEPSFX_INPUT", "/content/drive/MyDrive/keepsfx_input")
FS = 44100  # BandIt yeu cau 44.1kHz
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts")

print(f"[*] BANDIT_DIR = {BANDIT_DIR}")
print(f"[*] CKPT_PATH  = {CKPT_PATH} (ton tai: {os.path.isfile(CKPT_PATH)})")

# Thu muc Drive de bo video lon vao (khoi upload qua web)
if os.path.isdir(os.path.dirname(INPUT_DIR)):
    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f"[*] Bo video vao: {INPUT_DIR}")


def list_input_videos():
    if not os.path.isdir(INPUT_DIR):
        return []
    try:
        return sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VIDEO_EXTS))
    except Exception:
        return []


def _ffmpeg(args):
    """Goi ffmpeg, raise neu loi (kem log de debug)."""
    p = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg loi: {p.stderr[-800:]}")


def extract_audio(video_path, out_wav):
    """Tach audio tu video -> wav 44.1kHz stereo (dinh dang BandIt doc duoc)."""
    _ffmpeg(["-i", video_path, "-vn", "-ac", "2", "-ar", str(FS), out_wav])


def run_bandit(wav_path, out_dir):
    """Goi BandIt inference (subprocess) -> xuat cac stem vao out_dir."""
    if not os.path.isfile(CKPT_PATH):
        raise gr.Error(f"Khong thay checkpoint: {CKPT_PATH}. Kiem tra buoc tai model trong notebook.")
    # BandIt tim hparams.yaml o dirname(dirname(ckpt)) -> tuc thu muc CHA cua folder chua ckpt
    hparams = os.path.join(os.path.dirname(os.path.dirname(CKPT_PATH)), "hparams.yaml")
    if not os.path.isfile(hparams):
        raise gr.Error(
            f"Thieu hparams.yaml o {hparams} (BandIt tim o day). "
            "Checkpoint phai nam trong subfolder, hparams.yaml o thu muc cha."
        )
    cmd = [
        "python", "inference.py", "inference",
        f"--ckpt_path={CKPT_PATH}",
        f"--file_path={wav_path}",
        "--model_name=keepsfx",
        f"--output_dir={out_dir}",
    ]
    print("[*] Chay BandIt:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=BANDIT_DIR, capture_output=True, text=True)
    print(p.stdout[-2000:])
    if p.returncode != 0:
        raise RuntimeError(f"BandIt inference loi:\n{p.stderr[-2000:]}")


def find_effects_wav(out_dir):
    """Tim file stem 'effects/sfx' trong ket qua tach."""
    wavs = glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
    # Uu tien ten chua effect/sfx/fx, tranh nham speech/music
    for w in wavs:
        n = os.path.basename(w).lower()
        if any(k in n for k in ("effect", "sfx")) and "speech" not in n and "music" not in n:
            return w, wavs
    for w in wavs:
        n = os.path.basename(w).lower()
        if n.startswith("fx") or "_fx" in n:
            return w, wavs
    return None, wavs


def _process_impl(drive_file, upload_path, progress):
    video_path = upload_path or (os.path.join(INPUT_DIR, drive_file) if drive_file else None)
    if not video_path or not os.path.isfile(video_path):
        raise RuntimeError("Chua co video. Chon file tu Drive HOAC upload.")

    work = tempfile.mkdtemp(prefix="keepsfx_")
    progress(0.1, desc="Tach audio tu video...")
    wav = os.path.join(work, "audio.wav")
    extract_audio(video_path, wav)

    progress(0.3, desc="BandIt dang tach speech/music/SFX (lau nhat)...")
    sep_dir = os.path.join(work, "sep")
    os.makedirs(sep_dir, exist_ok=True)
    run_bandit(wav, sep_dir)

    progress(0.8, desc="Lay stem SFX...")
    sfx, all_wavs = find_effects_wav(sep_dir)
    if not sfx:
        raise RuntimeError(
            "Khong tim thay stem SFX. Cac file tach duoc: "
            + ", ".join(os.path.basename(w) for w in all_wavs)
            + " -> gui ten file nay cho dev de chinh bo loc."
        )

    progress(0.9, desc="Ghep video + SFX...")
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.environ.get("KEEPSFX_OUTPUT", work)
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, f"{base}_SFX.mp4")
    out_sfx = os.path.join(out_dir, f"{base}_SFX.wav")
    shutil.copyfile(sfx, out_sfx)
    _ffmpeg([
        "-i", video_path, "-i", out_sfx,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        out_mp4,
    ])
    progress(1.0, desc="Xong!")
    return out_mp4, out_sfx, f"✅ Xong!\nMP4: {out_mp4}\nSFX: {out_sfx}"


def process(drive_file, upload_path, progress=gr.Progress()):
    """Bao toan bo de LOI hien ra UI (Colab giau traceback)."""
    import traceback
    try:
        return _process_impl(drive_file, upload_path, progress)
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)  # cung in ra o Colab
        return None, None, f"❌ LOI:\n{exc}\n\n--- chi tiet ---\n{tb[-3000:]}"


with gr.Blocks(title="keepsfx - Giu lai SFX") as demo:
    gr.Markdown(
        "# 🎬 keepsfx → giu lai HIEU UNG (SFX)\n"
        "Upload video → bo **giong** + **nhac**, giu lai **tieng dong/hieu ung** → "
        "ra MP4 (audio = SFX) de long tieng Viet.\n"
        "*Model: BandIt (SOTA, license CC BY-NC - phi thuong mai).*"
    )
    with gr.Row():
        with gr.Column():
            drive_dd = gr.Dropdown(
                choices=list_input_videos(), value=None,
                label="📁 Chon video tu Drive (MyDrive/keepsfx_input) - NEN dung cho file lon",
            )
            refresh_btn = gr.Button("🔄 Lam moi danh sach Drive", size="sm")
            vin = gr.Video(label="… hoac Upload truc tiep (file nho)")
            btn = gr.Button("▶ Tach & giu SFX", variant="primary")
        with gr.Column():
            vout = gr.File(label="📥 MP4 (audio = SFX)")
            aout = gr.File(label="📥 sfx.wav")
            log = gr.Textbox(label="Log / Trang thai (loi hien o day)", lines=14)
    refresh_btn.click(fn=lambda: gr.update(choices=list_input_videos()), outputs=drive_dd)
    btn.click(fn=process, inputs=[drive_dd, vin], outputs=[vout, aout, log])

if __name__ == "__main__":
    share = os.environ.get("KEEPSFX_SHARE", "1") != "0"
    demo.queue().launch(share=share, inbrowser=not share, debug=True)
