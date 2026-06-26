# -*- coding: utf-8 -*-
"""
keepsfx - Tach giong + nhac ra khoi video/audio, CHI GIU HIEU UNG (SFX).
Dung BandIt (SOTA cinematic audio separation: speech/music/effects).

Tinh nang backup/restore:
  - Sau moi chunk xong, effect WAV duoc sao chep vao Drive (keepsfx_progress/).
  - Neu session Colab het han, lan sau mo lai -> tu dong tiep tuc tu doan chua xu ly.

Input audio (wav/mp3/...): toc do BandIt KHONG thay doi, chi tiet kiem buoc mux video.
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
MODELS_DIR   = os.environ.get("KEEPSFX_MODELS_DIR", os.path.dirname(os.path.dirname(CKPT_PATH)))
INPUT_DIR    = os.environ.get("KEEPSFX_INPUT",    "/content/drive/MyDrive/keepsfx_input")
PROGRESS_DIR = os.environ.get("KEEPSFX_PROGRESS", "/content/drive/MyDrive/keepsfx_progress")

FS         = 44100
CHUNK_SEC  = int(os.environ.get("KEEPSFX_CHUNK_SEC", "60"))  # 60s: an toan voi T4
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus")

KNOWN_MODELS = {
    "dnr-3s-erb48-l1snr":  "🚀 Nhanh nhất — erb48 (~2× nhanh hơn bark48, RTF ≈ 0.9×, T4: 1h → ~55p)",
    "dnr-3s-mus64-l1snr":  "⚡ Nhanh — mus64 (~1.5× nhanh hơn bark48, RTF ≈ 1.2×, T4: 1h → ~1h15p)",
    "dnr-3s-bark48-l1snr": "⚖️ Cân bằng — bark48 (RTF ≈ 1.7×, T4: 1h → ~1h45p)  ← mặc định",
    "dnr-3s-bark64-l1snr": "🎯 Chất lượng — bark64 (RTF ~4×+, T4: rất chậm, không khuyến dùng)",
}
_default_stem = os.path.splitext(os.path.basename(CKPT_PATH))[0]
DEFAULT_MODEL = _default_stem if _default_stem in KNOWN_MODELS else "dnr-3s-bark48-l1snr"

print(f"[*] BANDIT_DIR   = {BANDIT_DIR}")
print(f"[*] MODELS_DIR   = {MODELS_DIR}")
print(f"[*] CKPT_PATH    = {CKPT_PATH} (ton tai: {os.path.isfile(CKPT_PATH)})")
print(f"[*] DEFAULT_MODEL = {DEFAULT_MODEL}")
print(f"[*] PROGRESS_DIR = {PROGRESS_DIR}")
print(f"[*] CHUNK_SEC    = {CHUNK_SEC}s")

if os.path.isdir(os.path.dirname(INPUT_DIR)):
    os.makedirs(INPUT_DIR,    exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)


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
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "chunk_%04d.wav")
    _ffmpeg(["-i", src_path, "-vn", "-ac", "2", "-ar", str(FS),
             "-f", "segment", "-segment_time", str(CHUNK_SEC), pattern])
    return sorted(glob.glob(os.path.join(out_dir, "chunk_*.wav")))


def concat_wavs(wav_list, out_wav):
    list_file = out_wav + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for w in wav_list:
            escaped = w.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_wav])


# ====== BACKUP / RESTORE ======

def _job_id(file_path):
    size_mb = os.path.getsize(file_path) // (1024 * 1024)
    stem    = re.sub(r"[^\w-]", "_", os.path.splitext(os.path.basename(file_path))[0])[:40]
    return f"{stem}_{size_mb}MB"


def _load_cached_indices(progress_dir):
    if not os.path.isdir(progress_dir):
        return set()
    cached = set()
    for f in glob.glob(os.path.join(progress_dir, "chunk_*_effects.wav")):
        m = re.search(r"chunk_(\d+)_effects", os.path.basename(f))
        if m:
            cached.add(int(m.group(1)))
    return cached


def _backup_new_effects(sep_dir, progress_dir, already_backed, log_fn=None):
    """Copy effect WAV moi -> Drive. Tra ve set da backed (cap nhat)."""
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
            kb  = os.path.getsize(dst) // 1024
            msg = f"[backup] chunk {idx:04d} -> Drive ({kb}KB)"
            print(msg, flush=True)
            if log_fn:
                log_fn(msg)
        except Exception as e:
            msg = f"[backup] WARN chunk {idx:04d}: {e}"
            print(msg, flush=True)
            if log_fn:
                log_fn(msg)
    return new_backed


# ====== MODEL MANAGEMENT ======

def _ensure_model(stem, log_fn=None):
    """Dam bao checkpoint va hparams.yaml cho model stem ton tai. Tra ve (ckpt_path, hparams_dst)."""
    import urllib.request

    def _log(msg):
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)

    if not os.path.isdir(MODELS_DIR):
        raise RuntimeError(
            f"Thu muc model chua ton tai: {MODELS_DIR}\n"
            "Hay chay notebook de mount Drive va tao thu muc truoc."
        )

    ckpt_dir    = os.path.join(MODELS_DIR, "ckpt")
    ckpt_path   = os.path.join(ckpt_dir, f"{stem}.ckpt")
    hparams_src = os.path.join(BANDIT_DIR, "expt", f"{stem}.yaml")
    hparams_dst = os.path.join(MODELS_DIR, "hparams.yaml")

    os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.isfile(ckpt_path):
        url = f"https://zenodo.org/records/10160698/files/{stem}.ckpt?download=1"
        _log(f"[model] Chua co {stem}.ckpt → tai ve (~775MB, luu Drive de lan sau khoi tai lai)...")
        pct_logged = [0]

        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(int(count * block_size / total_size * 100), 100)
                if pct >= pct_logged[0] + 10:
                    pct_logged[0] = (pct // 10) * 10
                    _log(f"[model] Download {stem}: {pct}%")

        urllib.request.urlretrieve(url, ckpt_path, reporthook=_progress)
        _log(f"[model] Tai xong: {os.path.getsize(ckpt_path) // (1024 * 1024)}MB → {ckpt_path}")
    else:
        _log(f"[model] Dung model: {stem}")

    if os.path.isfile(hparams_src):
        shutil.copyfile(hparams_src, hparams_dst)
    elif not os.path.isfile(hparams_dst):
        raise RuntimeError(
            f"Thieu hparams.yaml: {hparams_dst}\nVa khong tim thay {hparams_src}. Hay chay notebook truoc."
        )

    return ckpt_path, hparams_dst


# ====== BANDIT INFERENCE ======

def run_bandit_multi(file_glob, out_dir, ckpt_path, total=0, backup_dir=None, log_fn=None):
    def _log(msg):
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)

    if not os.path.isfile(ckpt_path):
        raise gr.Error(f"Khong thay checkpoint: {ckpt_path}")
    hparams = os.path.join(MODELS_DIR, "hparams.yaml")
    if not os.path.isfile(hparams):
        raise gr.Error(f"Thieu hparams.yaml o {hparams}")

    cmd = [
        "python", "inference.py", "inference_multiple",
        f"--ckpt_path={ckpt_path}",
        f"--file_glob={file_glob}",
        "--model_name=keepsfx",
        f"--output_dir={out_dir}",
        "--include_track_name=True",
        "--get_residual=False",
        "--get_no_vox_combinations=False",
    ]
    _log("[*] BandIt: " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", BANDIT_DIR)
    # expandable_segments: giam fragment VRAM
    # garbage_collection_threshold: giai phong VRAM chu dong khi dat 60% capacity
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                   "expandable_segments:True,garbage_collection_threshold:0.6")

    def _done_count():
        return sum(
            1 for w in glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
            if os.path.splitext(os.path.basename(w))[0].lower() in ("effects", "effect", "sfx")
        )

    output_lines = []

    def _drain(pipe):
        for line in pipe:
            output_lines.append(line)

    proc = subprocess.Popen(cmd, cwd=BANDIT_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)
    drain_thread = threading.Thread(target=_drain, args=(proc.stdout,), daemon=True)
    drain_thread.start()

    def _fmt(sec):
        sec = int(sec)
        if sec >= 3600:
            return f"{sec//3600}h{(sec%3600)//60}p{sec%60:02d}s"
        return f"{sec//60}p{sec%60:02d}s"

    backed_up      = set()
    t0             = time.time()
    prev_done      = 0
    t_chunk1_done  = None   # thoi diem chunk DAU TIEN xong (sau khi model load)

    while proc.poll() is None:
        done = _done_count()
        if backup_dir:
            backed_up = _backup_new_effects(out_dir, backup_dir, backed_up, log_fn)

        el = time.time() - t0
        if done == 0:
            # Model dang load — log moi 10 giay de khong spam
            if int(el) % 10 < 2:
                _log(f"[BandIt] Load model... {_fmt(el)}")
        elif done > prev_done:
            # Co chunk moi vua xong
            if t_chunk1_done is None:
                t_chunk1_done = time.time()
            if done >= 2 and t_chunk1_done:
                # RTF do tu chunk thu 2 tro di (loai tru thoi gian load model)
                el_since = time.time() - t_chunk1_done
                time_per_chunk = el_since / (done - 1)
                rtf = time_per_chunk / CHUNK_SEC
                eta = (total - done) * time_per_chunk
                _log(f"[BandIt] {done}/{total} doan xong | RTF {rtf:.2f}x | ETA ~{_fmt(eta)}")
            else:
                _log(f"[BandIt] {done}/{total} doan xong | dang do RTF...")
            prev_done = done
        time.sleep(2)

    drain_thread.join()
    if backup_dir:
        _backup_new_effects(out_dir, backup_dir, backed_up, log_fn)

    bandit_out = "".join(output_lines)
    print(bandit_out[-3000:])
    if proc.returncode != 0:
        if proc.returncode == -9:
            raise RuntimeError(
                f"BandIt bi kill boi OOM killer (exit -9). GPU het VRAM sau khi xu ly.\n"
                f"-> Giam chunk nho hon: dat KEEPSFX_CHUNK_SEC=30 trong notebook roi chay lai.\n"
                f"Output cuoi:\n{bandit_out[-1500:]}"
            )
        raise RuntimeError(f"BandIt inference loi (exit {proc.returncode}):\n{bandit_out[-2000:]}")


def find_effects_list(out_dir):
    eff = [
        w for w in glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
        if os.path.splitext(os.path.basename(w))[0].lower() in ("effects", "effect", "sfx")
    ]
    eff.sort(key=lambda p: int(re.findall(r"chunk_(\d+)", p)[-1]) if re.findall(r"chunk_(\d+)", p) else 0)
    return eff


# ====== XU LY CHINH ======

def _process_impl(drive_file, upload_path, log_fn, model_stem=None):
    def _log(msg):
        print(msg, flush=True)
        log_fn(msg)

    raw_path = upload_path or (os.path.join(INPUT_DIR, drive_file) if drive_file else None)
    if not raw_path or not os.path.isfile(raw_path):
        raise RuntimeError("Chua co file. Chon tu Drive HOAC upload.")

    stem = model_stem or DEFAULT_MODEL
    ckpt_path, _ = _ensure_model(stem, log_fn=_log)

    is_audio     = raw_path.lower().endswith(AUDIO_EXTS)
    job_id       = _job_id(raw_path)
    drive_ok     = os.path.isdir(PROGRESS_DIR)
    progress_dir = os.path.join(PROGRESS_DIR, job_id) if drive_ok else None

    cached_indices = _load_cached_indices(progress_dir) if progress_dir else set()
    if cached_indices:
        _log(f"[resume] Tim thay {len(cached_indices)} doan da xu ly tren Drive (job: {job_id})")

    work     = tempfile.mkdtemp(prefix="keepsfx_")
    out_dir  = os.environ.get("KEEPSFX_OUTPUT", work)
    _success = False

    try:
        _log(f"[*] Trich audio tung {CHUNK_SEC}s chunk tu: {os.path.basename(raw_path)}")
        chunks_dir = os.path.join(work, "chunks")
        all_chunks = split_to_chunks(raw_path, chunks_dir)
        if not all_chunks:
            raise RuntimeError("Khong trich duoc audio.")

        todo_idx   = [i for i in range(len(all_chunks)) if i not in cached_indices]
        cached_idx = [i for i in range(len(all_chunks)) if i in cached_indices]
        _log(f"[*] Tong {len(all_chunks)} doan x {CHUNK_SEC}s | "
             f"can xu ly: {len(todo_idx)} | da co Drive: {len(cached_idx)}")

        new_effects = []
        if todo_idx:
            todo_dir = os.path.join(work, "todo")
            os.makedirs(todo_dir, exist_ok=True)
            for i in todo_idx:
                shutil.copy2(all_chunks[i], os.path.join(todo_dir, os.path.basename(all_chunks[i])))

            sep_dir = os.path.join(work, "sep")
            run_bandit_multi(
                os.path.join(todo_dir, "chunk_*.wav"), sep_dir,
                ckpt_path,
                len(todo_idx),
                backup_dir=progress_dir,
                log_fn=log_fn,
            )
            new_effects = find_effects_list(sep_dir)
            if not new_effects:
                all_wavs = glob.glob(os.path.join(sep_dir, "**", "*.wav"), recursive=True)
                raise RuntimeError(
                    "Khong tim thay stem 'effects'. Files tach duoc: "
                    + ", ".join(os.path.basename(w) for w in all_wavs[:20])
                )
        else:
            _log("[resume] Tat ca chunk da xu ly. Bo qua BandIt.")

        if todo_idx:
            for w in glob.glob(os.path.join(work, "sep", "**", "*.wav"), recursive=True):
                if os.path.splitext(os.path.basename(w))[0].lower() not in ("effects", "effect", "sfx"):
                    try:
                        os.remove(w)
                    except Exception:
                        pass
            shutil.rmtree(os.path.join(work, "todo"), ignore_errors=True)
        shutil.rmtree(chunks_dir, ignore_errors=True)

        _log("[*] Khoi phuc SFX tu Drive + ghep...")
        all_effect_pairs = []

        if progress_dir and cached_idx:
            for idx in cached_idx:
                src = os.path.join(progress_dir, f"chunk_{idx:04d}_effects.wav")
                if not os.path.isfile(src):
                    raise RuntimeError(
                        f"Thieu chunk_{idx:04d}_effects.wav tren Drive. "
                        f"Xoa thu muc keepsfx_progress/{job_id} va chay lai tu dau."
                    )
                dst = os.path.join(work, f"cached_{idx:04d}.wav")
                shutil.copy2(src, dst)
                all_effect_pairs.append((idx, dst))

        for path in new_effects:
            m = re.findall(r"chunk_(\d+)", path)
            if m:
                all_effect_pairs.append((int(m[-1]), path))

        all_effect_pairs.sort(key=lambda x: x[0])
        sfx_parts = [p for _, p in all_effect_pairs]

        _log("[*] Ghep SFX + xuat file...")
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
            _log(f"✅ Xong! SFX: {out_sfx}")
            return None, out_sfx

        out_mp4 = os.path.join(out_dir, f"{base}_SFX.mp4")
        _ffmpeg([
            "-i", raw_path, "-i", out_sfx,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            out_mp4,
        ])
        _success = True
        _log(f"✅ Xong!\nMP4: {out_mp4}\nSFX: {out_sfx}")
        return out_mp4, out_sfx

    finally:
        if out_dir != work:
            shutil.rmtree(work, ignore_errors=True)
        if _success and progress_dir and os.path.isdir(progress_dir):
            shutil.rmtree(progress_dir, ignore_errors=True)
            print(f"[cleanup] Xoa progress dir: {progress_dir}", flush=True)


def process(drive_file, upload_path, model_stem=None):
    """Generator: cap nhat log textbox moi 2 giay trong khi BandIt chay."""
    log_lines  = []
    final      = [None]
    error_msg  = [None]
    done_event = threading.Event()

    def _log(msg):
        log_lines.append(msg)

    def _bg():
        import traceback as _tb
        try:
            final[0] = _process_impl(drive_file, upload_path, _log, model_stem)
        except Exception as exc:
            tb = _tb.format_exc()
            print(tb)
            error_msg[0] = f"❌ LOI:\n{exc}\n\n--- chi tiet ---\n{tb[-3000:]}"
            log_lines.append(error_msg[0])
        finally:
            done_event.set()

    threading.Thread(target=_bg, daemon=True).start()

    # Cap nhat log moi 2 giay cho den khi xong
    while not done_event.wait(timeout=2):
        yield None, None, "\n".join(log_lines[-100:])

    # Ket qua cuoi
    log_text = "\n".join(log_lines)
    if error_msg[0]:
        yield None, None, log_text
    else:
        mp4, wav = final[0] or (None, None)
        yield mp4, wav, log_text


# ====== GRADIO UI ======

with gr.Blocks(title="keepsfx - Giu lai SFX") as demo:
    gr.Markdown(
        "# 🎬 keepsfx → giu lai HIEU UNG (SFX)\n"
        "Upload **video hoac audio** → bo giong + nhac, giu tieng dong/hieu ung "
        "→ MP4 + WAV de long tieng Viet.\n"
        "*Model: BandIt (SOTA, CC BY-NC). "
        "**Backup tu dong**: sau moi chunk → luu Drive → tiep tuc duoc neu Colab het han.*"
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
            model_radio = gr.Radio(
                choices=[(label, stem) for stem, label in KNOWN_MODELS.items()],
                value=DEFAULT_MODEL,
                label="⚙️ Chon model (chat luong / toc do)",
            )
            gr.Markdown(
                "_Model chua co tren Drive se tu dong tai (~775MB). "
                "RTF la uoc tinh tren T4 voi chunk 30s._"
            )
            btn = gr.Button("▶ Tach & giu SFX", variant="primary")
        with gr.Column():
            vout = gr.File(label="📥 MP4 (audio = SFX) — chi co neu input la video")
            aout = gr.File(label="📥 sfx.wav")
            log  = gr.Textbox(label="Log / Trang thai (cap nhat moi 2 giay)", lines=14)
    refresh_btn.click(fn=lambda: gr.update(choices=list_input_files()), outputs=drive_dd)
    btn.click(fn=process, inputs=[drive_dd, vin, model_radio], outputs=[vout, aout, log])

if __name__ == "__main__":
    share   = os.environ.get("KEEPSFX_SHARE", "1") != "0"
    allowed = [p for p in [os.environ.get("KEEPSFX_OUTPUT", ""), "/content/drive/MyDrive"] if p]
    demo.queue().launch(share=share, inbrowser=not share, debug=True, allowed_paths=allowed)
