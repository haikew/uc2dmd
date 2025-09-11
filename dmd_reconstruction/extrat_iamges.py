import sys
from pathlib import Path
from PIL import Image

def extract_tiff_stack(input_tiff: str, output_dir: str):
    """Extract 3 frames from a TIFF stack (must contain exactly 3) and save as separate single-frame TIFF files.
    Args:
        input_tiff: Path to multi-frame TIFF (stack) containing 3 images.
        output_dir: Directory to write frame_1.tif, frame_2.tif, frame_3.tif.
    """
    in_path = Path(input_tiff)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as img:
        frames = []
        try:
            i = 0
            while True:
                img.seek(i)
                frames.append(img.copy())
                i += 1
        except EOFError:
            pass  # reached end

    if len(frames) != 3:
        raise ValueError(f"TIFF stack must contain exactly 3 frames, found {len(frames)}")

    saved_files = []
    for idx, frame in enumerate(frames, start=1):
        # Ensure saved as single-frame grayscale or keep mode
        if frame.mode not in ("L", "I;16", "RGB"):
            frame = frame.convert("L")
        out_file = out_path / f"frame_{idx}.tif"
        frame.save(out_file, compression="none")
        saved_files.append(out_file)
        print(f"Saved: {out_file}")

    return [str(p) for p in saved_files]


def main():
    if len(sys.argv) != 3:
        print("Usage: python extrat_iamges.py <input_stack.tif> <output_dir>")
        sys.exit(1)
    input_tiff = sys.argv[1]
    output_dir = sys.argv[2]
    try:
        extract_tiff_stack(input_tiff, output_dir)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
