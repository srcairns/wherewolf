# --- START OF FILE extract_portraits.py ---

import os
import glob
from PIL import Image

# === CONFIG ===
input_dir = "cards_output"      # Directory with the full card images (e.g., Witch.png)
output_dir = "portraits_output" # Where to save the cropped portrait images

# Define the crop box (left, top, right, bottom) in pixels.
# This assumes your input card images are consistently sized (e.g., 1080x2340).
# Example: For a card of 1080x2340, this crops the top section.
# Adjust these values based on your card dimensions and desired portrait area.

#PORTRAIT_BOX = (45, 45, 1035, 1150)  # (left, upper, right, lower)
PORTRAIT_BOX = (65, 75, 1025, 1140)  # (left, upper, right, lower)

# === END CONFIG ===

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Portrait crop box: {PORTRAIT_BOX}")

# Get all image files (e.g., png, jpg) from the input directory
image_files = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"):
    image_files.extend(glob.glob(os.path.join(input_dir, ext)))

if not image_files:
    print(f"No image files found in '{input_dir}'.")
    exit()

print(f"\nFound {len(image_files)} images to process.")

processed_count = 0
error_count = 0

for image_path in image_files:
    filename = os.path.basename(image_path)
    print(f"Processing '{filename}'...")

    try:
        with Image.open(image_path) as img:
            # Crop the image using the defined box
            portrait_img = img.crop(PORTRAIT_BOX)

            # Construct the output path
            output_filepath = os.path.join(output_dir, filename) # Keep original filename

            # Save the cropped portrait
            portrait_img.save(output_filepath)
            processed_count += 1

    except FileNotFoundError:
        print(f"  Error: File not found at '{image_path}'. Skipping.")
        error_count += 1
    except Exception as e:
        print(f"  Error processing '{filename}': {e}")
        error_count += 1

print("\n--- Portrait Extraction Summary ---")
print(f"Successfully processed and saved: {processed_count}")
print(f"Errors: {error_count}")
print("---------------------------------")

# --- END OF FILE extract_portraits.py ---
