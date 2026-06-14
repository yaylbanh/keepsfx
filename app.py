# -*- coding: utf-8 -*-
"""
keepsfx - Tach giong + nhac ra khoi video, CHI GIU HIEU UNG (SFX).
Dung BandIt (SOTA cinematic audio separation: speech/music/effects).
Input: MP4/video -> Output: MP4 giu nguyen hinh, audio = chi SFX (+ file sfx.wav).

Muc dich: lam nen de long tieng Viet, bo nhac goc (ne ban quyen).
Model BandIt: license CC BY-NC (phi thuong mai).
"""

import os
import re
import glob
import shutil
import tempfile
import subprocess

import gradio as gr

# ====== CAU HINH (chinh qua bien moi truong trong notebook/bat) ======
BANDIT_DIR = os.environ.get("BANDIT_DIR", "/content/bandit")
CKPT_PATH = os.environ.get("BANDIT_CKPT", "/content/drive/MyDrive/keepsfx_models/ckpt/dnr-3s-bark64-l1snr.ckpt")
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


# Chia nho audio de BandIt khong bi het VRAM (xu ly ca file 1 luc -> OOM)
CHUNK_SEC = int(os.environ.get("KEEPSFX_CHUNK_SEC", "60"))


def split_video_to_chunks(video_path, out_dir):
    """Trich audio + chia doan CHUNK_SEC giay TRONG 1 BUOC tu video (khoi tao wav 10GB trung gian)."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "chunk_%04d.wav")
    _ffmpeg(["-i", video_path, "-vn", "-ac", "2", "-ar", str(FS),
             "-f", "segment", "-segment_time", str(CHUNK_SEC), pattern])
    return sorted(glob.glob(os.path.join(out_dir, "chunk_*.wav")))


def concat_wavs(wav_list, out_wav):
    """Noi cac wav (cung dinh dang) lai theo thu tu."""
    list_file = out_wav + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for w in wav_list:
            f.write(f"file '{w}'\n")
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_wav])


def run_bandit_multi(file_glob, out_dir):
    """Goi BandIt MOT LAN cho TAT CA chunk (inference_multiple) -> nap model 1 lan -> NHANH.
    Tat residual + combinations -> chi xuat speech/music/effects (it file, lay dung 'effects')."""
    if not os.path.isfile(CKPT_PATH):
        raise gr.Error(f"Khong thay checkpoint: {CKPT_PATH}. Kiem tra buoc tai model trong notebook.")
    hparams = os.path.join(os.path.dirname(os.path.dirname(CKPT_PATH)), "hparams.yaml")
    if not os.path.isfile(hparams):
        raise gr.Error(f"Thieu hparams.yaml o {hparams}. Checkpoint phai trong subfolder, hparams o thu muc cha.")
    cmd = [
        "python", "inference.py", "inference_multiple",
        f"--ckpt_path={CKPT_PATH}",
        f"--file_glob={file_glob}",
        "--model_name=keepsfx",
        f"--output_dir={out_dir}",
        "--include_track_name=True",
        "--get_residual=False",
        "--get_no_vox_combinations=False",
    ]
    print("[*] Chay BandIt (multi, nap model 1 lan):", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", BANDIT_DIR)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = subprocess.run(cmd, cwd=BANDIT_DIR, capture_output=True, text=True, env=env)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        raise RuntimeError(f"BandIt inference loi:\n{p.stderr[-2500:]}")


def find_effects_list(out_dir):
    """Lay TAT CA file stem 'effects' theo dung thu tu chunk."""
    eff = []
    for w in glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True):
        stem = os.path.splitext(os.path.basename(w))[0].lower()
        if stem in ("effects", "effect", "sfx"):
            eff.append(w)

    def _key(p):
        m = re.findall(r"chunk_(\d+)", p)
        return int(m[-1]) if m else 0

    eff.sort(key=_key)
    return eff


def _process_impl(drive_file, upload_path, progress):
    video_path = upload_path or (os.path.join(INPUT_DIR, drive_file) if drive_file else None)
    if not video_path or not os.path.isfile(video_path):
        raise RuntimeError("Chua co video. Chon file tu Drive HOAC upload.")

    work = tempfile.mkdtemp(prefix="keepsfx_")
    progress(0.05, desc="Trich + chia audio tu video...")
    # Trich + chia thang tu video (1 buoc, khoi wav 10GB trung gian -> tiet kiem disk cho video 4h)
    chunks_dir = os.path.join(work, "chunks")
    chunks = split_video_to_chunks(video_path, chunks_dir)
    if not chunks:
        raise RuntimeError("Khong trich duoc audio tu video.")
    print(f"[*] Chia {len(chunks)} doan x {CHUNK_SEC}s")

    # Chay BandIt MOT LAN cho tat ca chunk (nap model 1 lan -> nhanh hon nhieu)
    progress(0.1, desc=f"BandIt tach SFX {len(chunks)} doan (nap model 1 lan)...")
    sep_dir = os.path.join(work, "sep")
    run_bandit_multi(os.path.join(chunks_dir, "chunk_*.wav"), sep_dir)

    sfx_parts = find_effects_list(sep_dir)
    if not sfx_parts:
        all_wavs = glob.glob(os.path.join(sep_dir, "**", "*.wav"), recursive=True)
        raise RuntimeError(
            "Khong tim thay stem 'effects'. Cac file tach duoc: "
            + ", ".join(os.path.basename(w) for w in all_wavs[:30])
            + " -> gui cho dev de chinh bo loc."
        )

    # DON DISK: xoa stem speech/music + chunk input (giu effects) -> quan trong cho video 4h
    progress(0.8, desc="Don disk + ghep SFX...")
    for w in glob.glob(os.path.join(sep_dir, "**", "*.wav"), recursive=True):
        if os.path.splitext(os.path.basename(w))[0].lower() not in ("effects", "effect", "sfx"):
            try:
                os.remove(w)
            except Exception:
                pass
    shutil.rmtree(chunks_dir, ignore_errors=True)

    progress(0.85, desc="Ghep cac doan SFX + video...")
    sfx_full = os.path.join(work, "sfx_full.wav")
    if len(sfx_parts) == 1:
        sfx_full = sfx_parts[0]
    else:
        concat_wavs(sfx_parts, sfx_full)
    sfx = sfx_full
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
    # Don sach temp (output da o Drive) -> giai phong disk cho video dai / lan chay sau
    if out_dir != work:
        shutil.rmtree(work, ignore_errors=True)
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
    # Cho phep Gradio phuc vu file output nam tren Drive (mac dinh chi cho cwd/temp)
    allowed = [p for p in [os.environ.get("KEEPSFX_OUTPUT", ""), "/content/drive/MyDrive"] if p]
    demo.queue().launch(share=share, inbrowser=not share, debug=True, allowed_paths=allowed)
