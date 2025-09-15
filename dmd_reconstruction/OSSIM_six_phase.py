"""
Six-Phase OSSIM (Optical Sectioning Structured Illumination Microscopy)

基于等间隔六相(0°, 60°, 120°, 180°, 240°, 300°)条纹的光学切片重建。

核心思想：对六相序列做一阶谐波解调（类似相位移解调/DFT），
得到AC幅值|C1|作为调制振幅；将其除以DC（平均值）进行归一化，
得到抑制阴影的光学切片图。

公式（N=6）：
  DC = (1/N) * Σ_k I_k
  S_c = Σ_k I_k * cos(2πk/N)
  S_s = Σ_k I_k * sin(2πk/N)
  A = (2/N) * sqrt(S_c^2 + S_s^2)          # 一阶谐波幅值
  OS = A / (DC + eps)                       # 归一化光学切片
  φ = atan2(-S_s, S_c)                      # 可选：相位图（与初相位相关）

依赖：numpy, Pillow（tkinter 为可选GUI）

Author:
Date: 2025-09-15
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path
import glob
import os

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False


class SixPhaseOSSIM:
    def __init__(self):
        self.images: list[np.ndarray] = []

    def load_images(self, image_paths: list[str | os.PathLike]):
        if len(image_paths) != 6:
            raise ValueError("Exactly six phase-shifted images are required (0..300°, step 60°)")
        imgs: list[np.ndarray] = []
        for p in image_paths:
            with Image.open(p) as im:
                if im.mode != 'L':
                    im = im.convert('L')
                arr = np.array(im, dtype=np.float32) / 255.0
            imgs.append(arr)
        h, w = imgs[0].shape
        for i, a in enumerate(imgs[1:], 1):
            if a.shape != (h, w):
                raise ValueError(f"Image {i+1} dimension mismatch: {a.shape} vs {(h, w)}")
        self.images = imgs
        return True

    def reconstruct(self, return_phase: bool = False, mode: str = "demod"):
        if len(self.images) != 6:
            raise ValueError("Six phase-shifted images are required")
        I = self.images  # I[0..5]
        N = 6
        # DC term (widefield)
        widefield = (I[0] + I[1] + I[2] + I[3] + I[4] + I[5]) / float(N)

        mode_lc = (mode or "demod").lower()

        if mode_lc == "ios":
            # Multi-image IOS analogous to 3-image IOS: sqrt(sum_{i<j} (Ii - Ij)^2) with a normalization factor.
            # For 3-phase we had: IOS = sqrt(((I1-I2)^2 + (I1-I3)^2 + (I2-I3)^2)/2)
            # For 6-phase, there are C(6,2)=15 pairs. We choose the same scaling so that
            # amplitude is comparable to 3-phase up to a constant factor. Here we divide by 2 as in simple IOS.
            ios_sum = (
                (I[0]-I[1])**2 + (I[0]-I[2])**2 + (I[0]-I[3])**2 + (I[0]-I[4])**2 + (I[0]-I[5])**2 +
                (I[1]-I[2])**2 + (I[1]-I[3])**2 + (I[1]-I[4])**2 + (I[1]-I[5])**2 +
                (I[2]-I[3])**2 + (I[2]-I[4])**2 + (I[2]-I[5])**2 +
                (I[3]-I[4])**2 + (I[3]-I[5])**2 +
                (I[4]-I[5])**2
            )
            optical_section = np.sqrt(ios_sum / 2.0)
            if return_phase:
                # IOS模式不返回相位
                return widefield, optical_section, None
            return widefield, optical_section
        else:
            # First-harmonic demodulation via cosine/sine weights (default)
            # φ_k = 2πk/N, k=0..5
            k = np.arange(N, dtype=np.float32)
            phi = 2.0 * np.pi * k / float(N)
            cosw = np.cos(phi).astype(np.float32)
            sinw = np.sin(phi).astype(np.float32)

            Sc = np.zeros_like(I[0], dtype=np.float32)
            Ss = np.zeros_like(I[0], dtype=np.float32)
            for idx in range(N):
                Sc += I[idx] * cosw[idx]
                Ss += I[idx] * sinw[idx]

            amplitude = (2.0 / float(N)) * np.sqrt(Sc * Sc + Ss * Ss)
            eps = 1e-8
            optical_section = amplitude / (widefield + eps)
            optical_section = np.clip(optical_section, 0.0, 1.0)

            if return_phase:
                phase = np.arctan2(-Ss, Sc)
                return widefield, optical_section, phase
            return widefield, optical_section

    @staticmethod
    def _normalize_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype.kind != 'f':
            img = img.astype(np.float32)
        vmin = float(img.min())
        vmax = float(img.max())
        if vmax <= vmin:
            return np.full(img.shape, 128, dtype=np.uint8)
        out = (img - vmin) / (vmax - vmin)
        return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)

    def save_results(self, output_dir: str | os.PathLike, widefield: np.ndarray, optical_section: np.ndarray, phase: np.ndarray | None = None):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._normalize_u8(widefield), mode='L').save(str(out / 'widefield.png'))
        Image.fromarray(self._normalize_u8(optical_section), mode='L').save(str(out / 'optical_section.png'))
        if phase is not None:
            # Map phase [-pi, pi] to [0,255]
            ph_norm = (phase + np.pi) / (2.0 * np.pi)
            ph_u8 = np.clip(np.round(ph_norm * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(ph_u8, mode='L').save(str(out / 'phase_map.png'))
        return str(out)


def run_gui():
    if not _TK_AVAILABLE:
        raise RuntimeError("tkinter is not available in this environment")

    def select_and_process():
        folder = filedialog.askdirectory(title="选择包含六相图像的文件夹 (6 images)")
        if not folder:
            return
        exts = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))
        files = sorted(list(set(files)))
        if len(files) != 6:
            messagebox.showerror("错误", f"文件夹内需要且仅需要6张图像，当前找到 {len(files)} 张。")
            return

        outdir = filedialog.askdirectory(title="选择保存结果的文件夹")
        if not outdir:
            return

        try:
            status_var.set("处理中...")
            root.update()
            recon = SixPhaseOSSIM()
            recon.load_images(files)
            mode = 'ios' if ios_mode_var.get() else 'demod'
            wf, os_img, phase = recon.reconstruct(return_phase=(mode=='demod'), mode=mode)
            saved = recon.save_results(outdir, wf, os_img, phase)
            status_var.set("完成！")
            messagebox.showinfo("成功", f"六相OSSIM重建完成，结果保存在:\n{saved}\n\n输出文件:\n- widefield.png\n- optical_section.png\n- phase_map.png")
        except Exception as e:
            status_var.set("失败")
            messagebox.showerror("错误", f"处理失败：\n{e}")

    root = tk.Tk()
    root.title("六相 OSSIM 光学切片重建")
    root.geometry("520x220")
    root.resizable(False, False)

    tk.Label(root, text="六相 OSSIM 光学切片重建\n\n请选择包含 6 张等间隔相移图像的文件夹", font=("Arial", 12), pady=20).pack()
    # Mode selection
    ios_mode_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="使用IOS模式（与OSSIM_simple一致，不归一化）", variable=ios_mode_var).pack()

    tk.Button(root, text="选择图像文件夹并处理", font=("Arial", 11), command=select_and_process, width=22, height=2, bg="#4CAF50", fg="white").pack(pady=10)
    status_var = tk.StringVar(value="等待选择...")
    tk.Label(root, textvariable=status_var, font=("Arial", 10), fg="gray").pack(pady=10)
    tk.Label(root, text="支持格式: PNG, JPG, TIF, BMP", font=("Arial", 9), fg="blue").pack(pady=10)
    root.mainloop()


if __name__ == "__main__":
    print("Starting Six-Phase OSSIM GUI...")
    run_gui()
