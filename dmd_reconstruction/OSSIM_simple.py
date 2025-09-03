"""
IOS Reconstruction Tool for SIM Images
Simple IOS-based reconstruction from three phase-shifted images
IOS = [(I1 - I2)² + (I1 - I3)² + (I2 - I3)²]^(1/2)

Author: 
Date: 2025-08-29
"""

import numpy as np
from PIL import Image
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
        
    def ios_reconstruction(self):
        """Execute reconstruction using the basic IOS formula"""
        if len(self.images) != 3:
            raise ValueError("Three phase-shifted images are required")
        I1, I2, I3 = self.images
        widefield = (I1 + I2 + I3) / 3.0
        # 纯IOS公式
        ios = np.sqrt((I1 - I2)**2 + (I1 - I3)**2 + (I2 - I3)**2)
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
            widefield, ios = reconstructor.ios_reconstruction()
            saved_path = reconstructor.save_results(output_folder, widefield, ios)
            status_label.config(text="Processing completed!")
            messagebox.showinfo("Success", f"IOS reconstruction completed!\nResults saved to:\n{saved_path}\n\nOutput files:\n- widefield.png (average image)\n- ios_reconstruction.png (IOS reconstruction)")
        except Exception as e:
            status_label.config(text="Processing failed")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
    root = tk.Tk()
    root.title("IOS SIM Reconstruction Tool")
    root.geometry("480x200")
    root.resizable(False, False)
    info_label = tk.Label(root, text="IOS SIM Reconstruction Tool\n\nSelect a folder containing exactly 3 phase-shifted images", 
                         font=("Arial", 12), pady=20)
    info_label.pack()
    select_button = tk.Button(root, text="Select Image Folder", font=("Arial", 11), 
                             command=select_input_folder, width=20, height=2,
                             bg="#4CAF50", fg="white")
    select_button.pack(pady=10)
    status_label = tk.Label(root, text="Waiting for folder selection...", font=("Arial", 10), fg="gray")
    status_label.pack(pady=10)
    help_text = tk.Label(root, text="Supported formats: PNG, JPG, TIF, BMP", 
                        font=("Arial", 9), fg="blue")
    help_text.pack(pady=10)
    root.mainloop()


if __name__ == "__main__":
    print("Starting IOS SIM Reconstruction Tool...")
    run_ios_gui()
