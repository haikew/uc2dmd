"""
Six-Phase OSSIM (Optical Sectioning Structured Illumination Microscopy)

基于等间隔六相(0°, 60°, 120°, 180°, 240°, 300°)条纹的光学切片重建。
基于IOS = [(I1 - I2)² + (I1 - I3)² + (I2 - I3)²]^(1/2)公式的6相位版本

依赖：numpy, Pillow（tkinter 为可选GUI）

Author:
Date: 2024-06-01
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
        # 新增：记录亮度增益，便于调试或复用
        self.brightness_gains: list[float] | None = None

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

    def level_brightness(self, method: str = "median") -> None:
        """
        将六张图的整体亮度归一化到同一水平（默认基于中位数）。
        过程：
        - 计算每张图的统计量（均值或中位数）
        - 以所有统计量的中位数为参考，计算每张图的增益
        - 应用增益后，如整体最大值>1，则对全部图像再乘以公共缩放系数避免饱和
        """
        if not self.images or len(self.images) != 6:
            raise ValueError("Images must be loaded before brightness leveling")
        stats = []
        for img in self.images:
            if method == "mean":
                stats.append(float(np.mean(img)))
            else:
                stats.append(float(np.median(img)))
        # 参考值使用统计量的中位数，鲁棒性更好
        ref = float(np.median(np.array(stats, dtype=np.float32)))
        gains: list[float] = []
        leveled: list[np.ndarray] = []
        eps = 1e-8
        for img, s in zip(self.images, stats):
            g = ref / (s + eps) if s > eps else 1.0
            gains.append(float(g))
            leveled.append((img * g).astype(np.float32))
        # 避免溢出：统一按全局最大值做一次公共缩放（不破坏相对归一化）
        global_max = max(float(x.max()) for x in leveled)
        if global_max > 1.0 + 1e-6:
            scale = 1.0 / global_max
            leveled = [(x * scale).astype(np.float32) for x in leveled]
            gains = [g * scale for g in gains]
        self.images = leveled
        self.brightness_gains = gains

    def reconstruct(self, return_phase: bool = False, mode: str = "demod"):
        if len(self.images) != 6:
            raise ValueError("Six phase-shifted images are required")
        I = self.images  # I[0..5]
        N = 6
        # DC term (widefield)
        widefield = (I[0] + I[1] + I[2] + I[3] + I[4] + I[5]) / float(N)

        mode_lc = (mode or "demod").lower()

        if mode_lc == "ios":
            # Multi-image IOS analogous to 3-image IOS
            ios_sum = (
                (I[0]-I[1])**2 + (I[0]-I[2])**2 + (I[0]-I[3])**2 + (I[0]-I[4])**2 + (I[0]-I[5])**2 +
                (I[1]-I[2])**2 + (I[1]-I[3])**2 + (I[1]-I[4])**2 + (I[1]-I[5])**2 +
                (I[2]-I[3])**2 + (I[2]-I[4])**2 + (I[2]-I[5])**2 +
                (I[3]-I[4])**2 + (I[3]-I[5])**2 +
                (I[4]-I[5])**2
            )
            optical_section = np.sqrt(ios_sum / 2.0)
            phase = None
            return widefield, optical_section, phase
        else:
            # First-harmonic demodulation via cosine/sine weights (default)
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

            phase = np.arctan2(-Ss, Sc) if return_phase else None
            return widefield, optical_section, phase

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
            if brightness_norm_var.get():
                recon.level_brightness(method='median')
            mode = 'ios' if ios_mode_var.get() else 'demod'
            wf, os_img, phase = recon.reconstruct(return_phase=(mode=='demod'), mode=mode)
            saved = recon.save_results(outdir, wf, os_img, phase)
            status_var.set("完成！")
            outputs = "- widefield.png\n- optical_section.png"
            if phase is not None:
                outputs += "\n- phase_map.png"
            messagebox.showinfo("成功", f"六相OSSIM重建完成，结果保存在:\n{saved}\n\n输出文件:\n{outputs}")
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
    # 亮度归一化（输入级）
    brightness_norm_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="归一化每张输入图的整体亮度到同一水平", variable=brightness_norm_var).pack()

    tk.Button(root, text="选择图像文件夹并处理", font=("Arial", 11), command=select_and_process, width=22, height=2, bg="#4CAF50", fg="white").pack(pady=10)
    status_var = tk.StringVar(value="等待选择...")
    tk.Label(root, textvariable=status_var, font=("Arial", 10), fg="gray").pack(pady=10)
    tk.Label(root, text="支持格式: PNG, JPG, TIF, BMP", font=("Arial", 9), fg="blue").pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    print("Starting Six-Phase OSSIM GUI...")
    if _TK_AVAILABLE:
        run_gui()
    else:
        print("tkinter is not available in this environment")
