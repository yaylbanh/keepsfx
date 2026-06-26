# -*- coding: utf-8 -*-
"""
keepsfx - Tach giong + nhac ra khoi video/audio, CHI GIU HIEU UNG (SFX).
Dung BandIt (SOTA cinematic audio separation: speech/music/effects).

Tinh nang backup/restore:
  - Sau moi chunk xong, effect WAV duoc sao chep vao Drive (keepsfx_progress/).
  - Neu session Colab het han, lan sau mo lai -> tu dong tiep tuc tu doan chua xu ly.
  - Job ID = ten file + kich thuoc (MB) -> an toan cho nhieu video khac nhau.

Input audio (wav/mp3/...):
  - Toc do BandIt KHONG thay doi (cung la audio data).
  - Chi tiet kiem ~30s (bo buoc mux video cuoi).
  - Loi ich chinh: file upload nho hon, output chi can WAV.
"""

import os
import re
import glob
import time
import shutil
import tempfile
import threading
import subprocess

import gradio as gr

# ====== CAU HINH ======
BANDIT_DIR   = os.environ.get("BANDIT_DIR",   "/content/bandit")
CKPT_PATH    = os.environ.get("BANDIT_CKPT",  "/content/drive/MyDrive/keepsfx_models/ckpt/dnr-3s-bark48-l1snr.ckpt")
INPUT_DIR    = os.environ.get("KEEPSFX_INPUT",    "/content/drive/MyDrive/keepsfx_input")
PROGRESS_DIR = os.environ.get("KEEPSFX_PROGRESS", "/content/drive/MyDrive/keepsfx_progress")

FS         = 44100
CHUNK_SEC  = int(os.environ.get("KEEPSFX_CHUNK_SEC", "120"))
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus")

print(f"[*] BANDIT_DIR   = {BANDIT_DIR}")
print(f"[*] CKPT_PATH    = {CKPT_PATH} (ton tai: {os.path.isfile(CKPT_PATH)})")
print(f"[*] PROGRESS_DIR = {PROGRESS_DIR}")
print(f"[*] CHUNK_SEC    = {CHUNK_SEC}s")

if os.path.isdir(os.path.dirname(INPUT_DIR)):
    os.makedirs(INPUT_DIR,    exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    print(f"[*] Input dir : {INPUT_DIR}")
    print(f"[*] Progress  : {PROGRESS_DIR}")


# ====== DRIVE / INPUT LISTING ======

def list_input_files():
    if not os.path.isdir(INPUT_DIR):
        return []
    try:
        return sorted(
            f for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(VIDEO_EXTS + AUDIO_EXTS)
        )
    except Exception:
        return []


# ====== FFMPEG HELPERS ======

def _ffmpeg(args):
    p = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg loi: {p.stderr[-800:]}")


def split_to_chunks(src_path, out_dir):
    """Decode + chia chunk CHUNK_SEC giay tu bat ky video/audio -> wav 44.1kHz stereo."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "chunk_%04d.wav")
    _ffmpeg(["-i", src_path, "-vn", "-ac", "2", "-ar", str(FS),
             "-f", "segment", "-segment_time", str(CHUNK_SEC), pattern])
    return sorted(glob.glob(os.path.join(out_dir, "chunk_*.wav")))


def concat_wavs(wav_list, out_wav):
    list_file = out_wav + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for w in wav_list:
            # Escape single quote trong duong dan de khoi voi ffmpeg concat format
            escaped = w.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_wav])


# ====== BACKUP / RESTORE ======

def _job_id(file_path):
    """ID on dinh: ten file (da sanitize) + kich thuoc MB. Khong dung MD5 cho nhanh."""
    size_mb = os.path.getsize(file_path) // (1024 * 1024)
    stem    = re.sub(r"[^\w-]", "_", os.path.splitext(os.path.basename(file_path))[0])[:40]
    return f"{stem}_{size_mb}MB"


def _load_cached_indices(progress_dir):
    """Tra ve set cac chunk index da backup tren Drive."""
    if not os.path.isdir(progress_dir):
        return set()
    cached = set()
    for f in glob.glob(os.path.join(progress_dir, "chunk_*_effects.wav")):
        m = re.search(r"chunk_(\d+)_effects", os.path.basename(f))
        if m:
            cached.add(int(m.group(1)))
    return cached


def _backup_new_effects(sep_dir, progress_dir, already_backed):
    """Copy effect WAV moi xuat hien trong sep_dir -> Drive ngay lap tuc.
    Tra ve set da backup (cap nhat them moi). Khong raise - loi backup chi warn."""
    os.makedirs(progress_dir, exist_ok=True)
    new_backed = set(already_backed)
    for w in glob.glob(os.path.join(sep_dir, "**", "*.wav"), recursive=True):
        if os.path.splitext(os.path.basename(w))[0].lower() not in ("effects", "effect", "sfx"):
            continue
        m = re.findall(r"chunk_(\d+)", w)
        if not m:
            continue
        idx = int(m[-1])
        if idx in already_backed:
            continue
        dst = os.path.join(progress_dir, f"chunk_{idx:04d}_effects.wav")
        try:
            shutil.copy2(w, dst)
            new_backed.add(idx)
            kb = os.path.getsize(dst) // 1024
            print(f"[backup] chunk {idx:04d} -> Drive ({kb}KB)", flush=True)
        except Exception as e:
            print(f"[backup] WARN chunk {idx:04d}: {e}", flush=True)
    return new_backed


# ====== BANDIT INFERENCE ======

def run_bandit_multi(file_glob, out_dir, progress=None, total=0, backup_dir=None):
    """Goi BandIt 1 lan cho tat ca chunk. backup_dir: neu co, sao chep effect WAV vao Drive sau moi chunk."""
    if not os.path.isfile(CKPT_PATH):
        raise gr.Error(f"Khong thay checkpoint: {CKPT_PATH}")
    hparams = os.path.join(os.path.dirname(os.path.dirname(CKPT_PATH)), "hparams.yaml")
    if not os.path.isfile(hparams):
        raise gr.Error(f"Thieu hparams.yaml o {hparams}")

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
    print("[*] BandIt:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", BANDIT_DIR)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def _done_count():
        return sum(
            1 for w in glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
            if os.path.splitext(os.path.basename(w))[0].lower() in ("effects", "effect", "sfx")
        )

    # Drain thread: doc lien tuc de tranh deadlock pipe buffer day (> 64KB)
    output_lines = []

    def _drain(pipe):
        for line in pipe:
            output_lines.append(line)

    proc = subprocess.Popen(cmd, cwd=BANDIT_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)
    drain_thread = threading.Thread(target=_drain, args=(proc.stdout,), daemon=True)
    drain_thread.start()

    backed_up = set()
    t0 = time.time()
    while proc.poll() is None:
        done = _done_count()
        # Backup ngay chunk moi xong -> tranh mat du lieu khi Colab het han
        if backup_dir:
            backed_up = _backup_new_effects(out_dir, backup_dir, backed_up)
        if progress and total:
            el         = time.time() - t0
            audio_done = done * CHUNK_SEC
            audio_total = total * CHUNK_SEC
            rtf   = el / audio_done if audio_done > 0 else 0
            eta   = (audio_total - audio_done) * rtf if rtf > 0 else 0
            rtf_s = f"RTF {rtf:.2f}x" if rtf > 0 else "dang khoi dong..."
            eta_s = f"{int(eta//60)}p{int(eta%60):02d}s" if rtf > 0 else "?"
            progress(min(0.78, 0.1 + 0.68 * done / total),
                     desc=f"BandIt: {done}/{total} doan | {rtf_s} | ETA ~{eta_s}")
        time.sleep(2)

    drain_thread.join()
    # Final backup pass sau khi process ket thuc
    if backup_dir:
        _backup_new_effects(out_dir, backup_dir, backed_up)

    out = "".join(output_lines)
    print(out[-3000:])
    if proc.returncode != 0:
        raise RuntimeError(f"BandIt inference loi:\n{out[-2500:]}")


def find_effects_list(out_dir):
    eff = [
        w for w in glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
        if os.path.splitext(os.path.basename(w))[0].lower() in ("effects", "effect", "sfx")
    ]

    def _key(p):
        m = re.findall(r"chunk_(\d+)", p)
        return int(m[-1]) if m else 0

    eff.sort(key=_key)
    return eff


# ====== XU LY CHINH ======

def _process_impl(drive_file, upload_path, progress):
    raw_path = upload_path or (os.path.join(INPUT_DIR, drive_file) if drive_file else None)
    if not raw_path or not os.path.isfile(raw_path):
        raise RuntimeError("Chua co file. Chon tu Drive HOAC upload.")

    is_audio = raw_path.lower().endswith(AUDIO_EXTS)

    # Job ID + progress dir tren Drive
    job_id       = _job_id(raw_path)
    drive_ok     = os.path.isdir(PROGRESS_DIR)
    progress_dir = os.path.join(PROGRESS_DIR, job_id) if drive_ok else None

    # Kiem tra tien do cu
    cached_indices = _load_cached_indices(progress_dir) if progress_dir else set()
    resume_info    = ""
    if cached_indices:
        resume_info = f" | Tim thay {len(cached_indices)} doan da xu ly tren Drive (job: {job_id})"
        print(f"[resume] {resume_info.strip()}", flush=True)

    work    = tempfile.mkdtemp(prefix="keepsfx_")
    out_dir = os.environ.get("KEEPSFX_OUTPUT", work)
    _success = False

    try:
        progress(0.05, desc=f"Trich + chia audio ({CHUNK_SEC}s/doan){resume_info}...")
        chunks_dir = os.path.join(work, "chunks")
        all_chunks = split_to_chunks(raw_path, chunks_dir)
        if not all_chunks:
            raise RuntimeError("Khong trich duoc audio.")

        # Phan loai chunk: can xu ly vs da co tren Drive
        todo_idx   = [i for i in range(len(all_chunks)) if i not in cached_indices]
        cached_idx = [i for i in range(len(all_chunks)) if i in cached_indices]
        print(f"[*] Tong {len(all_chunks)} doan x {CHUNK_SEC}s | "
              f"can xu ly: {len(todo_idx)} | da co Drive: {len(cached_idx)}", flush=True)

        new_effects = []
        if todo_idx:
            # Chi xu ly doan chua co -> copy vao todo_dir de BandIt glob dung
            todo_dir = os.path.join(work, "todo")
            os.makedirs(todo_dir, exist_ok=True)
            for i in todo_idx:
                shutil.copy2(all_chunks[i], os.path.join(todo_dir, os.path.basename(all_chunks[i])))

            progress(0.1, desc=f"BandIt xu ly {len(todo_idx)} doan (nap model 1 lan){resume_info}...")
            sep_dir = os.path.join(work, "sep")
            run_bandit_multi(
                os.path.join(todo_dir, "chunk_*.wav"), sep_dir,
                progress, len(todo_idx),
                backup_dir=progress_dir,
            )
            new_effects = find_effects_list(sep_dir)
            if not new_effects:
                all_wavs = glob.glob(os.path.join(sep_dir, "**", "*.wav"), recursive=True)
                raise RuntimeError(
                    "Khong tim thay stem 'effects'. Files tach duoc: "
                    + ", ".join(os.path.basename(w) for w in all_wavs[:20])
                )
        else:
            progress(0.78, desc="Tat ca doan da co tren Drive, bo qua BandIt...")
            print("[resume] Tat ca chunk da xu ly. Bo qua BandIt.", flush=True)

        # DON DISK: xoa stem speech/music + chunks input
        progress(0.8, desc="Don disk...")
        if todo_idx:
            for w in glob.glob(os.path.join(work, "sep", "**", "*.wav"), recursive=True):
                if os.path.splitext(os.path.basename(w))[0].lower() not in ("effects", "effect", "sfx"):
                    try:
                        os.remove(w)
                    except Exception:
                        pass
            shutil.rmtree(os.path.join(work, "todo"), ignore_errors=True)
        shutil.rmtree(chunks_dir, ignore_errors=True)

        # Ghep tat ca effect WAV theo thu tu chunk (Drive cache + moi xu ly)
        progress(0.82, desc="Khoi phuc SFX tu Drive + ghep...")
        all_effect_pairs = []

        # Restore cached effects tu Drive
        if progress_dir and cached_idx:
            for idx in cached_idx:
                src = os.path.join(progress_dir, f"chunk_{idx:04d}_effects.wav")
                if not os.path.isfile(src):
                    raise RuntimeError(
                        f"Thieu chunk_{idx:04d}_effects.wav tren Drive. "
                        "Xoa thu muc keepsfx_progress/{job_id} va chay lai tu dau."
                    )
                dst = os.path.join(work, f"cached_{idx:04d}.wav")
                shutil.copy2(src, dst)
                all_effect_pairs.append((idx, dst))

        # Them cac effect moi vua xu ly
        for path in new_effects:
            m = re.findall(r"chunk_(\d+)", path)
            if m:
                all_effect_pairs.append((int(m[-1]), path))

        all_effect_pairs.sort(key=lambda x: x[0])
        sfx_parts = [p for _, p in all_effect_pairs]

        progress(0.87, desc="Ghep SFX + xuat file...")
        sfx_full = os.path.join(work, "sfx_full.wav")
        if len(sfx_parts) == 1:
            sfx_full = sfx_parts[0]
        else:
            concat_wavs(sfx_parts, sfx_full)

        base = os.path.splitext(os.path.basename(raw_path))[0]
        os.makedirs(out_dir, exist_ok=True)
        out_sfx = os.path.join(out_dir, f"{base}_SFX.wav")
        shutil.copyfile(sfx_full, out_sfx)

        if is_audio:
            _success = True
            progress(1.0, desc="Xong!")
            return (
                None, out_sfx,
                f"✅ Xong! (input audio - chi xuat WAV)\nSFX: {out_sfx}\n"
                f"[Luu y: audio input KHONG nhanh hon video voi BandIt - "
                f"toc do phu thuoc luong du lieu audio, khong phu thuoc container]"
            )

        out_mp4 = os.path.join(out_dir, f"{base}_SFX.mp4")
        _ffmpeg([
            "-i", raw_path, "-i", out_sfx,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            out_mp4,
        ])
        _success = True
        progress(1.0, desc="Xong!")
        return out_mp4, out_sfx, f"✅ Xong!\nMP4: {out_mp4}\nSFX: {out_sfx}"

    finally:
        # Xoa temp Colab (neu output da o Drive)
        if out_dir != work:
            shutil.rmtree(work, ignore_errors=True)
        # Xoa progress dir tren Drive chi khi THANH CONG
        # Khi loi -> giu lai de lan sau resume tiep tuc
        if _success and progress_dir and os.path.isdir(progress_dir):
            shutil.rmtree(progress_dir, ignore_errors=True)
            print(f"[cleanup] Xoa progress dir (da xong): {progress_dir}", flush=True)


def process(drive_file, upload_path, progress=gr.Progress()):
    import traceback
    try:
        return _process_impl(drive_file, upload_path, progress)
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        return None, None, f"❌ LOI:\n{exc}\n\n--- chi tiet ---\n{tb[-3000:]}"


# ====== GRADIO UI ======

with gr.Blocks(title="keepsfx - Giu lai SFX") as demo:
    gr.Markdown(
        "# 🎬 keepsfx → giu lai HIEU UNG (SFX)\n"
        "Upload **video hoac audio** → bo giong + nhac, giu lai tieng dong/hieu ung "
        "→ MP4 (audio = SFX) hoac WAV de long tieng Viet.\n"
        "*Model: BandIt (SOTA, CC BY-NC). "
        "**Backup tu dong**: sau moi chunk xong → luu Drive → co the tiep tuc neu Colab het han.*"
    )
    with gr.Row():
        with gr.Column():
            drive_dd = gr.Dropdown(
                choices=list_input_files(), value=None,
                label="📁 Chon file tu Drive (keepsfx_input) - NEN dung cho file lon",
            )
            refresh_btn = gr.Button("🔄 Lam moi danh sach Drive", size="sm")
            vin = gr.File(
                label="… hoac Upload truc tiep (video hoac audio, file nho)",
                file_types=[
                    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts",
                    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus",
                ],
            )
            btn = gr.Button("▶ Tach & giu SFX", variant="primary")
        with gr.Column():
            vout = gr.File(label="📥 MP4 (audio = SFX) — chi co neu input la video")
            aout = gr.File(label="📥 sfx.wav")
            log  = gr.Textbox(label="Log / Trang thai", lines=14)
    refresh_btn.click(fn=lambda: gr.update(choices=list_input_files()), outputs=drive_dd)
    btn.click(fn=process, inputs=[drive_dd, vin], outputs=[vout, aout, log])

if __name__ == "__main__":
    share   = os.environ.get("KEEPSFX_SHARE", "1") != "0"
    allowed = [p for p in [os.environ.get("KEEPSFX_OUTPUT", ""), "/content/drive/MyDrive"] if p]
    demo.queue().launch(share=share, inbrowser=not share, debug=True, allowed_paths=allowed)
