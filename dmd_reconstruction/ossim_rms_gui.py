#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OS-SIM (three-phase, RMS / in-phase demod) reconstruction tool (GUI + CLI)
- GUI: manually select three frames, set parameters, one-click reconstruction
- CLI: batch or headless usage

Dependencies: numpy pillow tifffile scikit-image opencv-python
Install: pip install numpy pillow tifffile scikit-image opencv-python
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile as tiff
from skimage import registration, exposure, restoration, img_as_float32
import cv2
# NEW / improved: optional DnD backends
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES as _DND_FILES_CONST
    _DND_AVAILABLE = True
    _USE_TKINTERDND2 = True
except ImportError:
    _DND_AVAILABLE = False
    _USE_TKINTERDND2 = False

# ---------------------- Basic image I/O and processing ----------------------
def read_any_image(path: Path):
    ext = path.suffix.lower()
    if ext in [".tif", ".tiff"]:
        arr = tiff.imread(str(path))
    else:
        arr = np.array(Image.open(path))
    return arr

def to_float(img):
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    else:
        return img_as_float32(img)

def write_image_uint(path: Path, img: np.ndarray, as_uint16: bool = True):
    img = np.nan_to_num(img, nan=0.0, posinf=np.max(img[np.isfinite(img)]) if np.any(np.isfinite(img)) else 0, neginf=0.0)
    img = img - np.min(img)
    vmax = np.max(img) if np.max(img) > 0 else 1.0
    img = img / vmax
    out = (img * (65535.0 if as_uint16 else 255.0) + 0.5).astype(np.uint16 if as_uint16 else np.uint8)
    tiff.imwrite(str(path), out)

def register_to_ref(moving, ref):
    # Phase correlation translation (subpixel)
    shift, _, _ = registration.phase_cross_correlation(ref, moving, upsample_factor=20)
    M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
    moved = cv2.warpAffine(moving, M, (moving.shape[1], moving.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return moved, (float(shift[0]), float(shift[1]))  # (row, col)

def ossim_rms(d1, d2, d3, normalize_by_sqrt3=False):
    out = np.sqrt((d1 - d2)**2 + (d1 - d3)**2 + (d2 - d3)**2)
    if normalize_by_sqrt3:
        out = out / np.sqrt(3.0)
    return out

def background_subtract(img, mode="gaussian", sigma=20, morph_radius=30):
    if mode == "none":
        return img
    if mode == "gaussian":
        k = int(6 * sigma) | 1
        bg = cv2.GaussianBlur(img, (k, k), sigma)
        out = np.clip(img - bg, 0, None)
        return out
    if mode == "morph":
        # May be slower; use for smaller images or when needing rolling-ball-like result
        from skimage import morphology
        selem = morphology.disk(int(morph_radius))
        bg = morphology.opening(img, selem)
        out = np.clip(img - bg, 0, None)
        return out
    return img

def clahe_enhance(img, clip=0.01, tile=8):
    out = exposure.equalize_adapthist(img, clip_limit=clip, nbins=256, kernel_size=tile)
    return out

def run_pipeline(path1, path2, path3, outdir, register=True, bleach_norm=False,
                 bg_mode="gaussian", bg_sigma=20, bg_morph_radius=30,
                 do_clahe=True, clahe_clip=0.01, clahe_tile=8,
                 save_uint16=True, normalize_sqrt3=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read and normalize to float
    d1 = to_float(read_any_image(Path(path1)))
    d2 = to_float(read_any_image(Path(path2)))
    d3 = to_float(read_any_image(Path(path3)))

    # Optional bleach normalization
    if bleach_norm:
        means = [np.mean(d1) + 1e-8, np.mean(d2) + 1e-8, np.mean(d3) + 1e-8]
        target = np.mean(means)
        d1, d2, d3 = [arr * (target / m) for arr, m in zip([d1, d2, d3], means)]

    # Registration (first frame as reference)
    shifts = [(0.0, 0.0)]
    if register:
        d2, s2 = register_to_ref(d2, d1)
        d3, s3 = register_to_ref(d3, d1)
        shifts = [(0.0, 0.0), s2, s3]

    # NEW: wide-field (average) BEFORE background suppression & CLAHE (after optional bleach + registration)
    wf_mean = np.mean(np.stack([d1, d2, d3], axis=0), axis=0)
    
    # Background suppression (before RMS)
    if bg_mode != "none":
        d1 = background_subtract(d1, mode=bg_mode, sigma=bg_sigma, morph_radius=bg_morph_radius)
        d2 = background_subtract(d2, mode=bg_mode, sigma=bg_sigma, morph_radius=bg_morph_radius)
        d3 = background_subtract(d3, mode=bg_mode, sigma=bg_sigma, morph_radius=bg_morph_radius)

    # RMS
    dos = ossim_rms(d1, d2, d3, normalize_by_sqrt3=normalize_sqrt3)

    # Optional CLAHE
    if do_clahe:
        dos = clahe_enhance(dos, clip=clahe_clip, tile=clahe_tile)

    # Filenames
    tag = Path(path1).parent.name + "_" + Path(path1).stem
    rms_path = Path(outdir) / f"{tag}_OSSIM_RMS.tif"
    wf_path = Path(outdir) / f"{tag}_WIDEFIELD.tif"

    # Save wide-field (raw average) and RMS result
    write_image_uint(wf_path, wf_mean, as_uint16=save_uint16)
    write_image_uint(rms_path, dos, as_uint16=save_uint16)

    # REMOVE: previous mean/modulation outputs (only two outputs now)

    # Simple modulation stats (using wide-field mask)
    th = np.percentile(wf_mean, 60)
    roi = wf_mean > th
    # Approximate modulation using per-pixel std / mean (no extra files)
    stack_for_stat = np.stack([d1, d2, d3], axis=0)
    std_map = np.std(stack_for_stat, axis=0)
    mean_map = np.mean(stack_for_stat, axis=0) + 1e-8
    mod_map_roi = (std_map / mean_map)[roi]
    global_mod_mean = float(np.mean(mod_map_roi))
    global_mod_median = float(np.median(mod_map_roi))

    return {
        "output_rms": str(rms_path),
        "output_widefield": str(wf_path),
        "shifts_row_col": shifts,
        "global_mod_mean_top40pct": global_mod_mean,
        "global_mod_median_top40pct": global_mod_median,
    }

# ---------------------- CLI interface ----------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="OS-SIM three-phase RMS (GUI + CLI)")
    ap.add_argument("--in", dest="inputs", nargs=3, help="Three frame paths: I1 I2 I3 (if provided run in CLI mode)")
    ap.add_argument("--out", dest="outdir", type=str, default="./ossim_out", help="Output directory")
    ap.add_argument("--no-register", dest="register", action="store_false", help="Disable subpixel registration (enabled by default)")
    ap.add_argument("--bleach-norm", action="store_true", help="Enable bleach normalization across frames")
    ap.add_argument("--bg", dest="bg_mode", choices=["none", "gaussian", "morph"], default="gaussian", help="Background suppression mode")
    ap.add_argument("--bg-sigma", type=float, default=20.0, help="Gaussian background sigma (pixels)")
    ap.add_argument("--bg-morph-radius", type=int, default=30, help="Morphological opening radius (pixels)")
    ap.add_argument("--no-clahe", dest="clahe", action="store_false", help="Disable CLAHE enhancement (enabled by default)")
    ap.add_argument("--clahe-clip", type=float, default=0.01, help="CLAHE clip limit")
    ap.add_argument("--clahe-tile", type=int, default=8, help="CLAHE tile size")
    ap.add_argument("--uint8", dest="uint16", action="store_false", help="Save as 8-bit (default 16-bit)")
    ap.add_argument("--normalize-sqrt3", action="store_true", help="Normalize RMS result by sqrt(3)")
    return ap

# ---------------------- GUI ----------------------
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    # Try native tkdnd if tkinterdnd2 not available
    _native_tkdnd_loaded = False
    if not _DND_AVAILABLE:
        try:
            _tmp_root = tk.Tk()
            try:
                _tmp_root.tk.eval('package require tkdnd')
                _native_tkdnd_loaded = True
            except tk.TclError:
                _native_tkdnd_loaded = False
            _tmp_root.destroy()
        except Exception:
            _native_tkdnd_loaded = False

    # Decide root class
    if _DND_AVAILABLE and _USE_TKINTERDND2:
        root = TkinterDnD.Tk()
        dnd_mode = "tkinterdnd2"
    else:
        root = tk.Tk()
        if _native_tkdnd_loaded:
            dnd_mode = "native_tkdnd"
        else:
            dnd_mode = "none"

    root.title("OS-SIM RMS (three-phase) Reconstruction")
    root.geometry("620x380")

    paths = [tk.StringVar(), tk.StringVar(), tk.StringVar()]
    outdir = tk.StringVar(value=str(Path.cwd() / "ossim_out"))
    bg_mode = tk.StringVar(value="gaussian")
    bg_sigma = tk.DoubleVar(value=20.0)
    bg_morph_radius = tk.IntVar(value=30)
    register = tk.BooleanVar(value=True)
    bleach_norm = tk.BooleanVar(value=False)
    clahe = tk.BooleanVar(value=True)
    clahe_clip = tk.DoubleVar(value=0.01)
    clahe_tile = tk.IntVar(value=8)
    uint16 = tk.BooleanVar(value=True)
    normalize_sqrt3 = tk.BooleanVar(value=False)

    # REPLACE old set_dropped / handle_drop with improved versions
    def _parse_drop_data(data):
        # Handles formats like: {C:/a b/c.png} {D:/d.png}
        if not data:
            return []
        try:
            return [p for p in root.tk.splitlist(data) if p]
        except Exception:
            # Fallback naive split
            return [s.strip("{}") for s in data.split() if s]

    def set_dropped(file_list):
        imgs = []
        for f in file_list:
            lf = f.lower()
            if lf.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                imgs.append(f)
        if not imgs:
            status.set("No image files detected in drop")
            return
        for i, p in enumerate(imgs[:3]):
            paths[i].set(p)
        extra = len(imgs) - 3
        status.set(f"Dropped {len(imgs[:3])} image(s)" + (f" (+{extra} ignored)" if extra > 0 else ""))

    def handle_drop(event):
        try:
            files = _parse_drop_data(event.data)
            if not files:
                status.set("Drop received but empty")
                return
            set_dropped(files)
        except Exception as e:
            status.set(f"Drop error: {e}")

    # Optional highlight effects
    def handle_enter(event):
        drop_label.configure(relief="solid")
    def handle_leave(event):
        drop_label.configure(relief="ridge")

    def browse(idx):
        p = filedialog.askopenfilename(title=f"Select frame {idx+1}", filetypes=[("Images","*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.gif")])
        if p:
            paths[idx].set(p)

    def browse_out():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            outdir.set(d)

    def run_now():
        try:
            p1, p2, p3 = [Path(v.get()) for v in paths]
            if not all(p.exists() for p in [p1, p2, p3]):
                messagebox.showerror("Error", "Please select three valid image files.")
                return
            od = Path(outdir.get())
            od.mkdir(parents=True, exist_ok=True)

            btn_run.config(state="disabled")
            status.set("Processing... please wait")

            root.update_idletasks()
            info = run_pipeline(
                p1, p2, p3, od,
                register=register.get(),
                bleach_norm=bleach_norm.get(),
                bg_mode=bg_mode.get(),
                bg_sigma=bg_sigma.get(),
                bg_morph_radius=bg_morph_radius.get(),
                do_clahe=clahe.get(),
                clahe_clip=clahe_clip.get(),
                clahe_tile=clahe_tile.get(),
                save_uint16=uint16.get(),
                normalize_sqrt3=normalize_sqrt3.get(),
            )
            status.set(f"Done: RMS -> {Path(info['output_rms']).name}, Wide-field -> {Path(info['output_widefield']).name}")
            messagebox.showinfo(
                "Done",
                f"RMS reconstruction:\n{info['output_rms']}\n\nWide-field (average):\n{info['output_widefield']}\n\nModulation mean (bright 40%): {info['global_mod_mean_top40pct']:.3f}"
            )

        except Exception as e:
            messagebox.showerror("Exception", str(e))
        finally:
            btn_run.config(state="normal")

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    # NEW: drag-and-drop zone (full-width)
    drop_style = {"relief": "ridge", "padding": 6}
    drop_frame = ttk.Frame(frm)
    drop_frame.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 8))

    if dnd_mode == "tkinterdnd2":
        dnd_text = "Drag & Drop 3 images here (tkinterdnd2 active)"
        dnd_color = "#0a0"
    elif dnd_mode == "native_tkdnd":
        dnd_text = "Drag & Drop 3 images here (native tkdnd)"
        dnd_color = "#0a0"
    else:
        dnd_text = "Drag & Drop not available (install: pip install tkinterdnd2)"
        dnd_color = "#a55"

    drop_label = ttk.Label(drop_frame, text=dnd_text, foreground=dnd_color, relief="ridge", padding=4)
    drop_label.pack(fill="x")

    # Register drop targets depending on backend
    if dnd_mode == "tkinterdnd2":
        try:
            drop_label.drop_target_register(_DND_FILES_CONST)
            drop_label.dnd_bind("<<Drop>>", handle_drop)
            drop_label.dnd_bind("<<DragEnter>>", handle_enter)
            drop_label.dnd_bind("<<DragLeave>>", handle_leave)
            # Root also accepts (optional)
            root.drop_target_register(_DND_FILES_CONST)
            root.dnd_bind("<<Drop>>", handle_drop)
        except Exception as e:
            status = tk.StringVar(value=f"DnD init failed: {e}")
    elif dnd_mode == "native_tkdnd":
        # Native tkdnd naming uses 'DND_Files'
        try:
            drop_label.drop_target_register('DND_Files')
            drop_label.dnd_bind('<<Drop>>', handle_drop)
            drop_label.dnd_bind("<<DragEnter>>", handle_enter)
            drop_label.dnd_bind("<<DragLeave>>", handle_leave)
            root.drop_target_register('DND_Files')
            root.dnd_bind('<<Drop>>', handle_drop)
        except Exception as e:
            pass  # silent fallback

    # SHIFT existing rows down by +1
    base_row = 1
    for i in range(3):
        ttk.Label(frm, text=f"Frame {i+1} path").grid(row=base_row + i, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=paths[i], width=60).grid(row=base_row + i, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Browse...", command=lambda idx=i: browse(idx)).grid(row=base_row + i, column=2)

    ttk.Label(frm, text="Output directory").grid(row=base_row + 3, column=0, sticky="w", pady=4)
    ttk.Entry(frm, textvariable=outdir, width=60).grid(row=base_row + 3, column=1, sticky="we", padx=6)
    ttk.Button(frm, text="Select...", command=browse_out).grid(row=base_row + 3, column=2)

    # NEW: Clear button
    def clear_paths():
        for v in paths: v.set("")
        status.set("Cleared paths")
    ttk.Button(frm, text="Clear Paths", command=clear_paths).grid(row=base_row + 4, column=2, sticky="e", pady=(0,4))

    # Options
    row = base_row + 4
    ttk.Separator(frm).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
    row += 1

    ttk.Checkbutton(frm, text="Subpixel registration", variable=register).grid(row=row, column=0, sticky="w"); row += 1
    ttk.Checkbutton(frm, text="Bleach normalization (match mean intensity)", variable=bleach_norm).grid(row=row, column=0, sticky="w"); row += 1
    ttk.Checkbutton(frm, text="CLAHE enhancement", variable=clahe).grid(row=row, column=0, sticky="w"); row += 1
    ttk.Checkbutton(frm, text="Save as 16-bit", variable=uint16).grid(row=row, column=0, sticky="w"); row += 1
    ttk.Checkbutton(frm, text="Normalize RMS by sqrt(3)", variable=normalize_sqrt3).grid(row=row, column=0, sticky="w"); row += 1

    ttk.Label(frm, text="Background suppression mode").grid(row=row, column=0, sticky="w")
    ttk.Combobox(frm, textvariable=bg_mode, values=["none","gaussian","morph"], width=10, state="readonly").grid(row=row, column=1, sticky="w"); row += 1

    ttk.Label(frm, text="Gaussian σ (px)").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=bg_sigma, width=10).grid(row=row, column=1, sticky="w"); row += 1

    ttk.Label(frm, text="Morph radius (px)").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=bg_morph_radius, width=10).grid(row=row, column=1, sticky="w"); row += 1

    ttk.Label(frm, text="CLAHE clip").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=clahe_clip, width=10).grid(row=row, column=1, sticky="w"); row += 1

    ttk.Label(frm, text="CLAHE tile").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=clahe_tile, width=10).grid(row=row, column=1, sticky="w"); row += 1

    status = tk.StringVar(value="Ready")
    ttk.Label(frm, textvariable=status, foreground="#0a0").grid(row=row, column=0, columnspan=3, sticky="w", pady=8)

    global btn_run
    btn_run = ttk.Button(frm, text="Run reconstruction", command=run_now)
    btn_run.grid(row=row+1, column=0, sticky="w", pady=4)

    root.mainloop()

# ---------------------- Entry point ----------------------
def main():
    ap = build_argparser()
    args = ap.parse_args()

    # CLI mode
    if args.inputs:
        p1, p2, p3 = args.inputs
        info = run_pipeline(
            p1, p2, p3, args.outdir,
            register=args.register,
            bleach_norm=args.bleach_norm,
            bg_mode=args.bg_mode,
            bg_sigma=float(args.bg_sigma),
            bg_morph_radius=int(args.bg_morph_radius),
            do_clahe=args.clahe,
            clahe_clip=float(args.clahe_clip),
            clahe_tile=int(args.clahe_tile),
            save_uint16=args.uint16,
            normalize_sqrt3=args.normalize_sqrt3,
        )
        print("[OK] RMS:", info["output_rms"])
        print("[OK] Wide-field:", info["output_widefield"])
        print("[Diag] shifts(row,col)=", info["shifts_row_col"])
        print("[Diag] modulation mean (top40%)=", info["global_mod_mean_top40pct"])
        return

    # GUI mode
    try:
        run_gui()
    except Exception as e:
        print("GUI launch failed:", e, file=sys.stderr)
        print("You can use CLI mode, e.g.:", file=sys.stderr)
        print('python ossim_rms_gui.py --in "I1.tif" "I2.tif" "I3.tif" --out ./out', file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
