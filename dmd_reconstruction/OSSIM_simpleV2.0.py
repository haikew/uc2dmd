"""
IOS Reconstruction Tool for SIM Images
Simple IOS-based reconstruction from three phase-shifted images
IOS = [(I1 - I2)² + (I1 - I3)² + (I2 - I3)²]^(1/2)

Author: 
Date: 2025-08-29
"""

import numpy as np
from PIL import Image, ImageFilter
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import glob
import os


class IOSReconstructor:
    """IOS-based SIM Reconstruction Class"""

    def __init__(self):
        self.images = []

    def load_images(self, image_paths):
        """Load three phase-shifted images"""
        if len(image_paths) != 3:
            raise ValueError("Exactly three phase-shifted images are required")
        self.images = []
        for path in image_paths:
            with Image.open(path) as img:
                if img.mode != 'L':
                    img = img.convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
            self.images.append(img_array)
        # Check if image dimensions are consistent
        reference_shape = self.images[0].shape
        for img in self.images[1:]:
            if img.shape != reference_shape:
                raise ValueError("Image dimension mismatch")
        return True

    def _gaussian_blur(self, img, sigma):
        """Apply Gaussian blur to a float image. Prefer SciPy, fallback to Pillow."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(img, sigma=float(sigma))
        except Exception:
            arr8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr8, mode='L')
            pil_blur = pil.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
            return np.array(pil_blur, dtype=np.float32) / 255.0

    def ios_reconstruction(self, apply_blur=False, sigma=1.0, apply_pre_blur=False, pre_sigma=1.0):
        """Execute reconstruction using the basic IOS formula"""
        if len(self.images) != 3:
            raise ValueError("Three phase-shifted images are required")
        I1, I2, I3 = self.images

        # Optional: Gaussian pre-filtering of input phase images to reduce noise/stripe artifacts propagation
        if apply_pre_blur and pre_sigma and float(pre_sigma) > 0:
            I1b = self._gaussian_blur(I1, pre_sigma)
            I2b = self._gaussian_blur(I2, pre_sigma)
            I3b = self._gaussian_blur(I3, pre_sigma)
        else:
            I1b, I2b, I3b = I1, I2, I3

        widefield = (I1b + I2b + I3b) / 3.0
        # Pure IOS formula
        ios = np.sqrt((I1b - I2b) ** 2 + (I1b - I3b) ** 2 + (I2b - I3b) ** 2)

        # Optional: Gaussian blur on IOS result for further artifact suppression
        if apply_blur and sigma and float(sigma) > 0:
            ios = self._gaussian_blur(ios, sigma)
        return widefield, ios

    def save_results(self, output_dir, widefield, ios):
        """Save reconstruction results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def normalize_for_save(img):
            if img.max() == img.min():
                return np.full(img.shape, 128, dtype=np.uint8)
            img_norm = (img - img.min()) / (img.max() - img.min())
            return (img_norm * 255).astype(np.uint8)

        widefield_img = Image.fromarray(normalize_for_save(widefield), mode='L')
        widefield_img.save(str(output_path / "widefield.png"))
        ios_img = Image.fromarray(normalize_for_save(ios), mode='L')
        ios_img.save(str(output_path / "ios_reconstruction.png"))
        return str(output_path)


def run_ios_gui():
    """Run IOS reconstruction tool with GUI"""

    def select_input_folder():
        folder_path = filedialog.askdirectory(title="Select folder containing three phase-shifted images")
        if not folder_path:
            return
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        image_files = list(set(image_files))
        if len(image_files) != 3:
            messagebox.showerror("Error", f"Folder must contain exactly 3 image files!\nFound {len(image_files)} images.")
            return
        image_files.sort()
        try:
            reconstructor = IOSReconstructor()
            reconstructor.load_images(image_files)
            output_folder = filedialog.askdirectory(title="Select folder to save reconstruction results")
            if not output_folder:
                return
            status_label.config(text="Processing...")
            root.update()
            widefield, ios = reconstructor.ios_reconstruction(
                apply_blur=apply_blur_var.get(),
                sigma=blur_sigma_var.get(),
                apply_pre_blur=apply_pre_blur_var.get(),
                pre_sigma=pre_blur_sigma_var.get(),
            )
            saved_path = reconstructor.save_results(output_folder, widefield, ios)
            status_label.config(text="Processing completed!")
            messagebox.showinfo(
                "Success",
                f"IOS reconstruction completed!\nResults saved to:\n{saved_path}\n\nOutput files:\n- widefield.png (average image)\n- ios_reconstruction.png (IOS reconstruction)",
            )
        except Exception as e:
            status_label.config(text="Processing failed")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")

    root = tk.Tk()
    root.title("IOS SIM Reconstruction Tool")
    root.geometry("480x380")
    root.resizable(False, False)

    info_label = tk.Label(
        root,
        text="IOS SIM Reconstruction Tool\n\nSelect a folder containing exactly 3 phase-shifted images",
        font=("Arial", 12),
        pady=20,
    )
    info_label.pack()

    select_button = tk.Button(
        root,
        text="Select Image Folder",
        font=("Arial", 11),
        command=select_input_folder,
        width=20,
        height=2,
        bg="#4CAF50",
        fg="white",
    )
    select_button.pack(pady=10)

    # Gaussian pre-blur controls (input images)
    apply_pre_blur_var = tk.BooleanVar(value=False)
    pre_blur_sigma_var = tk.DoubleVar(value=0.8)

    def toggle_pre_sigma_state():
        state = tk.NORMAL if apply_pre_blur_var.get() else tk.DISABLED
        pre_sigma_scale.config(state=state)

    apply_pre_blur_check = tk.Checkbutton(
        root, text="Apply Gaussian pre-filtering to input phase images", variable=apply_pre_blur_var, command=toggle_pre_sigma_state
    )
    apply_pre_blur_check.pack()
    pre_sigma_scale = tk.Scale(
        root,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        label="Pre-filter sigma",
        variable=pre_blur_sigma_var,
        length=300,
    )
    pre_sigma_scale.pack()
    toggle_pre_sigma_state()

    # Gaussian blur controls (post-process IOS)
    apply_blur_var = tk.BooleanVar(value=True)
    blur_sigma_var = tk.DoubleVar(value=1.0)

    def toggle_sigma_state():
        state = tk.NORMAL if apply_blur_var.get() else tk.DISABLED
        sigma_scale.config(state=state)

    apply_blur_check = tk.Checkbutton(
        root, text="Apply Gaussian blur to IOS result", variable=apply_blur_var, command=toggle_sigma_state
    )
    apply_blur_check.pack()
    sigma_scale = tk.Scale(
        root,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        label="Post-process sigma",
        variable=blur_sigma_var,
        length=300,
    )
    sigma_scale.pack()
    toggle_sigma_state()

    status_label = tk.Label(root, text="Waiting for folder selection...", font=("Arial", 10), fg="gray")
    status_label.pack(pady=10)
    help_text = tk.Label(root, text="Supported formats: PNG, JPG, TIF, BMP", font=("Arial", 9), fg="blue")
    help_text.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    print("Starting IOS SIM Reconstruction Tool...")
    run_ios_gui()
