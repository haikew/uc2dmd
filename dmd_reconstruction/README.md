# Optical Sectioning Structured Illumination Microscopy Reconstruction Tool

This is an easy-to-use optical sectioning SIM reconstruction tool with a graphical interface that processes three phase-shifted structured illumination images for optical sectioning reconstruction.

## Features

- **Simple GUI interface**: Select folders and save locations through dialog boxes
- **Lightweight dependencies**: Only requires numpy and PIL, no complex image processing libraries
- **Automatic file detection**: Automatically detects image files in folders
- **Strict input validation**: Ensures exactly 3 image files are present
- **One-click processing**: Automatically completes optical sectioning reconstruction
- **Multi-format support**: Supports PNG, JPG, TIF, BMP formats

## Installation

Only requires two simple libraries:

```bash
pip install Pillow numpy
```

Or use the requirements.txt from the project root directory:

```bash
pip install -r ../requirements.txt
```

## Usage

### GUI Mode (Recommended)

Run the program directly:
```bash
python optical_sectioning_sim.py
```

Operation steps:
1. Click "Select Image Folder" button
2. Select a folder containing **exactly 3** phase-shifted images
3. The program will automatically check file count and report errors if requirements are not met
4. If validation passes, a dialog will appear to select the save location
5. Processing completes automatically!

### Programmatic Usage

```python
from optical_sectioning_sim import OpticalSectioningSIM

# Create reconstructor
sim = OpticalSectioningSIM()

# Load three images (in phase order: 0°, 120°, 240°)
image_paths = ['phase1.png', 'phase2.png', 'phase3.png']
sim.load_images(image_paths)

# Execute reconstruction
widefield, optical_section, modulation_depth = sim.optical_sectioning_reconstruction()

# Save results
sim.save_results('output_folder', widefield, optical_section, modulation_depth)
```

## Output Files

After processing, 3 files will be generated in the specified directory:

1. **widefield.png** - Widefield image (average of three images)
2. **optical_section.png** - Optical sectioning reconstruction result
3. **modulation_depth.png** - Modulation depth map

## Important Notes

### Input Requirements
- Folder must contain **exactly 3** image files
- Images must be structured illumination images with 120° phase difference
- Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP
- Image dimensions must be identical

### Image Order
The program processes images in alphabetical order by filename. Ensure filenames correctly reflect phase order (0°, 120°, 240°).

### Processing Parameters
- Modulation threshold fixed at 0.1 (suitable for most cases)
- Automatic image normalization
- Output images in 8-bit PNG format

## Dependencies

**Uses only lightweight libraries**:
- `numpy`: Numerical computing
- `PIL (Pillow)`: Image reading/writing
- `tkinter`: GUI interface (Python standard library)

**Does NOT need**:
- OpenCV (too complex)
- matplotlib (for visualization)
- scipy (scientific computing library)

## Common Issues

**Q: Error about wrong number of images?**
A: Ensure the folder contains only 3 image files, remove other image files.

**Q: Processing failed?**
A: Check if image files are corrupted or have inconsistent dimensions.

**Q: Poor results?**
A: Ensure input images are correct phase-shifted sequences with sufficient fringe contrast.

**Q: Dependency installation issues?**
A: Only need `pip install Pillow numpy`, no other complex libraries required.

## Algorithm Principle

Based on three-phase structured illumination optical sectioning algorithm:
1. Calculate widefield image (average of three images)
2. Calculate modulation components for each phase
3. Extract modulation amplitude and depth
4. Perform optical sectioning reconstruction based on modulation depth

Suitable for removing out-of-focus light and scattered light, improving imaging contrast and axial resolution.

---

*Version: 2.0 (English Simplified)*  
*Date: 2025-08-28*  
*Dependencies: numpy + PIL only*
