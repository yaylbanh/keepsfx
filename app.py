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
CKPT_PATH = os.environ.get("BANDIT_CKPT", "/content/drive/MyDrive/keepsfx_models/dnr-3s-bark48-l1snr.ckpt")
FS = 44100  # BandIt yeu cau 44.1kHz

print(f"[*] BANDIT_DIR = {BANDIT_DIR}")
print(f"[*] CKPT_PATH  = {CKPT_PATH} (ton tai: {os.path.isfile(CKPT_PATH)})")


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
    hparams = os.path.join(os.path.dirname(CKPT_PATH), "hparams.yaml")
    if not os.path.isfile(hparams):
        raise gr.Error(f"Thieu hparams.yaml canh checkpoint ({hparams}). Notebook phai copy config vao day.")
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


def process(video_path, progress=gr.Progress()):
    if not video_path or not os.path.isfile(video_path):
        raise gr.Error("Chua chon video.")

    work = tempfile.mkdtemp(prefix="keepsfx_")
    try:
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
            raise gr.Error(
                "Khong tim thay stem SFX trong ket qua. Cac file tach duoc: "
                + ", ".join(os.path.basename(w) for w in all_wavs)
                + " -> bao lai ten file de chinh bo loc."
            )

        progress(0.9, desc="Ghep video + SFX...")
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.environ.get("KEEPSFX_OUTPUT", work)
        os.makedirs(out_dir, exist_ok=True)
        out_mp4 = os.path.join(out_dir, f"{base}_SFX.mp4")
        out_sfx = os.path.join(out_dir, f"{base}_SFX.wav")
        shutil.copyfile(sfx, out_sfx)
        # Giu nguyen video (copy), thay audio = sfx
        _ffmpeg([
            "-i", video_path, "-i", out_sfx,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            out_mp4,
        ])
        progress(1.0, desc="Xong!")
        return out_mp4, out_sfx
    finally:
        pass  # giu work de debug neu can; co the don sau


with gr.Blocks(title="keepsfx - Giu lai SFX") as demo:
    gr.Markdown(
        "# 🎬 keepsfx → giu lai HIEU UNG (SFX)\n"
        "Upload video → bo **giong** + **nhac**, giu lai **tieng dong/hieu ung** → "
        "ra MP4 (audio = SFX) de long tieng Viet.\n"
        "*Model: BandIt (SOTA, license CC BY-NC - phi thuong mai).*"
    )
    with gr.Row():
        with gr.Column():
            vin = gr.Video(label="Video dau vao (MP4...)")
            btn = gr.Button("▶ Tach & giu SFX", variant="primary")
        with gr.Column():
            vout = gr.File(label="📥 MP4 (audio = SFX)")
            aout = gr.File(label="📥 sfx.wav")
    btn.click(fn=process, inputs=[vin], outputs=[vout, aout])

if __name__ == "__main__":
    share = os.environ.get("KEEPSFX_SHARE", "1") != "0"
    demo.queue().launch(share=share, inbrowser=not share)
