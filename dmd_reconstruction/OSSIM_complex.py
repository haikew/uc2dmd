"""
Optical Sectioning Tool using IOS (In-focus/Out-of-focus Sectioning) Algorithm
For processing three phase-shifted projection images for optical sectioning reconstruction

This tool uses a simplified IOS algorithm with basic image preprocessing to extract
optical section information and reduce out-of-focus light interference.

Author: 
Date: 2025-09-05
"""

import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import glob
import os
import threading


class OpticalSectioningIOS:
    """Optical Sectioning using IOS (In-focus/Out-of-focus Sectioning) Algorithm"""
    
    def __init__(self):
        """Initialize the reconstructor"""
        self.images = []
        
    def load_images(self, image_paths):
        """Load three phase-shifted images"""
        if len(image_paths) != 3:
            raise ValueError("Exactly three phase-shifted images are required")
            
        self.images = []
        
        for i, path in enumerate(image_paths):
            try:
                # Load image using PIL
                with Image.open(path) as img:
                    # Convert to grayscale
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    # Convert to numpy array
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                self.images.append(img_array)
                
            except Exception as e:
                raise Exception(f"Failed to load image {path}: {e}")
                
        # Check if image dimensions are consistent
        reference_shape = self.images[0].shape
        for i, img in enumerate(self.images[1:], 1):
            if img.shape != reference_shape:
                raise ValueError(f"Image {i+1} dimension mismatch: {img.shape} vs {reference_shape}")
                
        return True
        
    def optical_sectioning_reconstruction(self):
        """Execute IOS (In-focus/Out-of-focus Sectioning) algorithm with minimal, recommended preprocessing"""
        if len(self.images) != 3:
            raise ValueError("Three phase-shifted images are required")
            
        I0, I1, I2 = self.images

        # Minimal preprocessing (recommended): a single light 3x3 Gaussian available if needed
        def gaussian_filter(img):
            """Fast vectorized 3x3 Gaussian blur with reflect padding"""
            p = np.pad(img, ((1, 1), (1, 1)), mode='reflect')
            return (
                p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:] +
                2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:] +
                p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
            ) / 16.0

        # DC term (widefield) directly from raw inputs to preserve contrast
        widefield = (I0 + I1 + I2) / 3.0

        # 3-phase demodulation (AC amplitude), preserves in-focus modulation
        # I_k = DC + AC * cos(phi + k*2pi/3). Demod amplitude:
        ac = np.sqrt((2.0 / 3.0) * (((I0 - I1) ** 2) + ((I1 - I2) ** 2) + ((I2 - I0) ** 2)))

        # Normalize by DC to reduce shading; clip to [0,1] for saving
        optical_section_norm = ac / (widefield + 1e-8)

        # Optional gentle smoothing (single pass) to suppress pixel noise
        optical_section_final = gaussian_filter(optical_section_norm)

        optical_section_final = np.clip(optical_section_final, 0, 1)
        return widefield, optical_section_final
        
    def save_results(self, output_dir, widefield, optical_section):
        """Save reconstruction results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Print statistics for debugging
        print(f"Widefield: min={widefield.min():.4f}, max={widefield.max():.4f}, mean={widefield.mean():.4f}")
        print(f"Optical section: min={optical_section.min():.4f}, max={optical_section.max():.4f}, mean={optical_section.mean():.4f}")
        
        # Simplified normalize function to reduce artifacts
        def normalize_for_save(img, name=""):
            if img.max() == img.min():
                print(f"Warning: {name} image has no dynamic range (constant value: {img.max():.4f})")
                return np.full(img.shape, 128, dtype=np.uint8)  # Gray image
            
            # Use simple linear normalization to avoid artifacts
            img_norm = (img - img.min()) / (img.max() - img.min())
            result = (img_norm * 255).astype(np.uint8)
            
            print(f"{name}: min={img.min():.4f}, max={img.max():.4f} -> normalized")
            return result
        
        # Save images with improved normalization
        widefield_img = Image.fromarray(normalize_for_save(widefield, "Widefield"), mode='L')
        widefield_img.save(str(output_path / "widefield.png"))
        
        optical_section_img = Image.fromarray(normalize_for_save(optical_section, "Optical section"), mode='L')
        optical_section_img.save(str(output_path / "optical_section.png"))
        
        return str(output_path)


def run_ios_gui():
    """Run IOS reconstruction tool with GUI"""
    
    def select_input_folder():
        """Select folder containing three images"""
        folder_path = filedialog.askdirectory(title="Select folder containing three phase-shifted images")
        if not folder_path:
            return
            
        # Find image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        # Remove duplicates (in case files match multiple patterns)
        image_files = list(set(image_files))
        
        # Check number of images
        if len(image_files) != 3:
            messagebox.showerror("Error", f"Folder must contain exactly 3 image files!\nFound {len(image_files)} images.")
            return
            
        # Sort by filename to ensure consistent order
        image_files.sort()
        
        try:
            # Select output folder first (blocking dialog in UI thread)
            output_folder = filedialog.askdirectory(title="Select folder to save reconstruction results")
            if not output_folder:
                return
            
            # Disable UI and run all heavy work (load + reconstruct + save) in a background thread
            status_label.config(text="Processing...")
            select_button.config(state=tk.DISABLED)

            def worker():
                try:
                    sim = OpticalSectioningIOS()
                    sim.load_images(image_files)
                    widefield, optical_section = sim.optical_sectioning_reconstruction()
                    saved_path = sim.save_results(output_folder, widefield, optical_section)

                    def on_success():
                        status_label.config(text="Processing completed!")
                        select_button.config(state=tk.NORMAL)
                        messagebox.showinfo("Success", f"Reconstruction completed!\nResults saved to:\n{saved_path}\n\nOutput files:\n- widefield.png (widefield image)\n- optical_section.png (optical section)")

                    root.after(0, on_success)
                except Exception as e:
                    def on_error():
                        status_label.config(text="Processing failed")
                        select_button.config(state=tk.NORMAL)
                        messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
                    root.after(0, on_error)

            threading.Thread(target=worker, daemon=True).start()
            
        except Exception as e:
            status_label.config(text="Processing failed")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")

    # Create main window
    root = tk.Tk()
    root.title("Optical Sectioning IOS Reconstruction Tool")
    root.geometry("480x200")
    root.resizable(False, False)
    
    # Add description text
    info_label = tk.Label(root, text="Optical Sectioning IOS Reconstruction Tool\n\nPlease select a folder containing exactly 3 phase-shifted images", 
                         font=("Arial", 12), pady=20)
    info_label.pack()
    
    # Add selection button
    select_button = tk.Button(root, text="Select Image Folder", font=("Arial", 11), 
                             command=select_input_folder, width=20, height=2,
                             bg="#4CAF50", fg="white")
    select_button.pack(pady=10)
    
    # Status label
    status_label = tk.Label(root, text="Waiting for folder selection...", font=("Arial", 10), fg="gray")
    status_label.pack(pady=10)
    
    # Add help text
    help_text = tk.Label(root, text="Note: Folder must contain exactly 3 image files\nSupported formats: PNG, JPG, TIF, BMP", 
                        font=("Arial", 9), fg="blue")
    help_text.pack(pady=10)
    
    root.mainloop()


if __name__ == "__main__":
    print("Starting Optical Sectioning IOS Reconstruction Tool...")
    run_ios_gui()
