from pathlib import Path
from OSSIM_six_phase import SixPhaseOSSIM


def main():
	# Example assumes images are named sorted in phase order.
	folder = Path("./example_six_phase")  # change to your folder
	if not folder.exists():
		print(f"Example folder not found: {folder.resolve()}")
		return
	# Collect exactly six images
	import glob, os
	exts = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
	files = []
	for ext in exts:
		files.extend(glob.glob(os.path.join(folder.as_posix(), ext)))
		files.extend(glob.glob(os.path.join(folder.as_posix(), ext.upper())))
	files = sorted(list(set(files)))
	if len(files) != 6:
		print(f"Need exactly 6 images, found {len(files)}")
		return

	recon = SixPhaseOSSIM()
	recon.load_images(files)
	# Use IOS mode to match OSSIM_simple behavior (no DC normalization)
	widefield, optical_section, phase = recon.reconstruct(return_phase=False, mode='ios')
	outdir = folder / "recon_output"
	saved = recon.save_results(outdir, widefield, optical_section, phase)
	print("Saved results to:", saved)


if __name__ == "__main__":
	main()
