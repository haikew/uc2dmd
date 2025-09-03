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
        """Execute optical sectioning reconstruction with salt-and-pepper noise reduction"""
        if len(self.images) != 3:
            raise ValueError("Three phase-shifted images are required")
            
        I0, I1, I2 = self.images
        
        # Step 1: Pre-process images to reduce salt-and-pepper noise
        def median_filter(img, kernel_size=3):
            """Median filter to remove salt-and-pepper noise"""
            pad = kernel_size // 2
            padded = np.pad(img, pad, mode='reflect')
            filtered = np.zeros_like(img)
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    filtered[i, j] = np.median(window)
            return filtered
        
        # Apply median filter to each image to remove salt-and-pepper noise
        I0_filtered = median_filter(I0, kernel_size=3)
        I1_filtered = median_filter(I1, kernel_size=3)  
        I2_filtered = median_filter(I2, kernel_size=3)
        
        # Calculate widefield image (average image)
        widefield = (I0_filtered + I1_filtered + I2_filtered) / 3.0
        
        # Calculate modulation components (based on 120-degree phase shift)
        diff0 = I0_filtered - widefield
        diff1 = I1_filtered - widefield  
        diff2 = I2_filtered - widefield
        
        # Calculate first-order diffraction components for 3-phase SIM
        cos_component = 2.0/3.0 * (2.0*diff0 - diff1 - diff2)
        sin_component = 2.0/np.sqrt(3.0) * (diff1 - diff2)
        
        # Modulation amplitude (visibility)
        modulation_amplitude = np.sqrt(cos_component**2 + sin_component**2)
        
        # Apply additional median filter to modulation amplitude to remove residual noise
        modulation_amplitude_clean = median_filter(modulation_amplitude, kernel_size=3)
        
        # Normalized modulation depth (visibility)
        safe_widefield = np.maximum(widefield, 1e-6)
        modulation_depth = modulation_amplitude_clean / safe_widefield
        
        # Apply median filter to modulation depth as well
        modulation_depth_clean = median_filter(modulation_depth, kernel_size=3)
        
        # Simple optical sectioning reconstruction - reduce artifacts
        modulation_threshold = 0.02  # Very low threshold
        
        # Conservative enhancement
        enhanced_signal = modulation_amplitude_clean * 0.3  # Further reduce to minimize noise amplification
        
        # Conservative reconstruction
        optical_section = widefield + enhanced_signal
        
        # Apply one more median filter to the final result to ensure noise removal
        optical_section_clean = median_filter(optical_section, kernel_size=3)
        
        # Gentle thresholding
        mask = modulation_depth_clean > modulation_threshold
        optical_section_final = optical_section_clean * mask + widefield * (1 - mask) * 0.9
        
        # Ensure values stay reasonable
        optical_section_final = np.clip(optical_section_final, 0, 1)
        
        return widefield, optical_section_final, modulation_depth_clean
        
    def save_results(self, output_dir, widefield, optical_section, modulation_depth):
        """Save reconstruction results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Print statistics for debugging
        print(f"Widefield: min={widefield.min():.4f}, max={widefield.max():.4f}, mean={widefield.mean():.4f}")
        print(f"Optical section: min={optical_section.min():.4f}, max={optical_section.max():.4f}, mean={optical_section.mean():.4f}")
        print(f"Modulation depth: min={modulation_depth.min():.4f}, max={modulation_depth.max():.4f}, mean={modulation_depth.mean():.4f}")
        
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
        
        modulation_depth_img = Image.fromarray(normalize_for_save(modulation_depth, "Modulation depth"), mode='L')
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
        
        # Remove duplicates (in case files match multiple patterns)
        image_files = list(set(image_files))
        
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
