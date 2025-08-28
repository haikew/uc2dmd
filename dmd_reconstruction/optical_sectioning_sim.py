"""
Optical Sectioning Structured Illumination Microscopy Reconstruction Tool
For processing three phase-shifted projection images for optical sectioning reconstruction

Based on the optical sectioning principle of structured illumination microscopy, 
this tool extracts optical section information from three phase-structured illumination images
and removes out-of-focus light and scattered light.

Author: 
Date: 2025-08-28
"""

import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import glob
import os


class OpticalSectioningSIM:
    """Optical Sectioning Structured Illumination Microscopy Reconstruction Class"""
    
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
        """Execute optical sectioning reconstruction"""
        if len(self.images) != 3:
            raise ValueError("Three phase-shifted images are required")
            
        I0, I1, I2 = self.images
        
        # Calculate widefield image (average image)
        widefield = (I0 + I1 + I2) / 3.0
        
        # Calculate modulation components (based on 120-degree phase shift)
        diff0 = I0 - widefield
        diff1 = I1 - widefield  
        diff2 = I2 - widefield
        
        # Calculate first-order diffraction components
        cos_component = 2.0 * (diff0 - 0.5 * diff1 - 0.5 * diff2) / 3.0
        sin_component = (diff1 - diff2) / np.sqrt(3.0)
        
        # Modulation amplitude
        modulation_amplitude = np.sqrt(cos_component**2 + sin_component**2)
        
        # Normalized modulation depth
        safe_widefield = np.maximum(widefield, 1e-6)
        modulation_depth = modulation_amplitude / safe_widefield
        
        # Optical sectioning reconstruction (simplified version)
        modulation_threshold = 0.1  # Fixed threshold
        modulation_mask = modulation_depth > modulation_threshold
        
        # Weighted reconstruction based on modulation depth
        optical_section = widefield * (1 + modulation_depth)
        optical_section = optical_section * modulation_mask + widefield * (1 - modulation_mask) * 0.1
        
        return widefield, optical_section, modulation_depth
        
    def save_results(self, output_dir, widefield, optical_section, modulation_depth):
        """Save reconstruction results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Normalize to 0-255 range for saving
        def normalize_for_save(img):
            img_norm = (img - img.min()) / (img.max() - img.min())
            return (img_norm * 255).astype(np.uint8)
        
        # Save images (using PIL)
        widefield_img = Image.fromarray(normalize_for_save(widefield), mode='L')
        widefield_img.save(str(output_path / "widefield.png"))
        
        optical_section_img = Image.fromarray(normalize_for_save(optical_section), mode='L')
        optical_section_img.save(str(output_path / "optical_section.png"))
        
        modulation_depth_img = Image.fromarray(normalize_for_save(modulation_depth), mode='L')
        modulation_depth_img.save(str(output_path / "modulation_depth.png"))
        
        return str(output_path)


def run_sim_gui():
    """Run SIM reconstruction tool with GUI"""
    
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
        
        # Check number of images
        if len(image_files) != 3:
            messagebox.showerror("Error", f"Folder must contain exactly 3 image files!\nFound {len(image_files)} images.")
            return
            
        # Sort by filename to ensure consistent order
        image_files.sort()
        
        try:
            # Create reconstructor and load images
            sim = OpticalSectioningSIM()
            sim.load_images(image_files)
            
            # Select output folder
            output_folder = filedialog.askdirectory(title="Select folder to save reconstruction results")
            if not output_folder:
                return
                
            # Execute reconstruction
            status_label.config(text="Processing...")
            root.update()
            
            widefield, optical_section, modulation_depth = sim.optical_sectioning_reconstruction()
            
            # Save results
            saved_path = sim.save_results(output_folder, widefield, optical_section, modulation_depth)
            
            status_label.config(text="Processing completed!")
            messagebox.showinfo("Success", f"Reconstruction completed!\nResults saved to:\n{saved_path}\n\nOutput files:\n- widefield.png (widefield image)\n- optical_section.png (optical section)\n- modulation_depth.png (modulation depth)")
            
        except Exception as e:
            status_label.config(text="Processing failed")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
    
    # Create main window
    root = tk.Tk()
    root.title("Optical Sectioning Structured Illumination Microscopy Reconstruction Tool")
    root.geometry("480x200")
    root.resizable(False, False)
    
    # Add description text
    info_label = tk.Label(root, text="Optical Sectioning SIM Reconstruction Tool\n\nPlease select a folder containing exactly 3 phase-shifted images", 
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
    print("Starting Optical Sectioning Structured Illumination Microscopy Reconstruction Tool...")
    run_sim_gui()
