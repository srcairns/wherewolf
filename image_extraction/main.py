# --- START OF MODIFIED FILE main.py ---

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os
import glob

# === CONFIG ===
input_dir = "pdfs"
output_dir = "cards_output"
debug_dir = "debug_crops"
save_debug_crops = False # Set to True to check initial crops again

dpi = 400
min_cut_gap = 20
whitespace_ratio_thresh = 0.75
threshold_level = 150
resize_to = (1080, 2340)

# --- Cropping Strictness Controls ---
use_iterative_trim = True
dark_edge_threshold = 75        # For main iterative shrink (sides/bottom mostly)
edge_scan_density = 0.75        # For main iterative shrink
max_shrink_iterations = 100
# --- New setting for refined top edge check ---
refine_top_avg_threshold = 200  # <<< Avg pixel value threshold (0-255). Lower = darker row needed. Try 180, 220 etc.
refine_top_edge_scan_limit = 100 # <<< Increase scan limit if needed (large top whitespace)
# --- End new settings ---
crop_inset = 0
trim_threshold = 220

# --- END CONFIG ---

# (Make directories, clean_cut_positions, find_cut_positions_by_whitespace - same)
os.makedirs(output_dir, exist_ok=True)
if save_debug_crops:
    os.makedirs(debug_dir, exist_ok=True)

def clean_cut_positions(cuts, min_distance=20):
    """Filters cut positions, ensuring a minimum distance between them."""
    if not cuts: return []
    cleaned = []
    sorted_cuts = sorted(list(set(cuts)))
    prev = -min_distance
    for c in sorted_cuts:
        if c - prev >= min_distance:
             cleaned.append(c); prev = c
    return cleaned

def find_cut_positions_by_whitespace(binary_img, axis='horizontal', ratio_thresh=0.85, min_gap=10):
    """Finds potential cut positions based on the START of continuous whitespace."""
    h, w = binary_img.shape
    if axis == 'horizontal':
        pixel_counts = np.mean(binary_img == 255, axis=0); dimension = w; scan_axis = 'cols'
    else:
        pixel_counts = np.mean(binary_img == 255, axis=1); dimension = h; scan_axis = 'rows'
    cuts = []; in_gap = False
    if dimension > 0 and pixel_counts[0] >= ratio_thresh: in_gap = True
    for i in range(1, dimension):
        is_white = pixel_counts[i] >= ratio_thresh; was_white = pixel_counts[i-1] >= ratio_thresh
        if is_white and not was_white: cuts.append(i); in_gap = True
        elif not is_white and was_white: in_gap = False
    cleaned_cuts = clean_cut_positions(cuts, min_distance=min_gap)
    print(f"    Found {len(cleaned_cuts)} cleaned cuts ({scan_axis}): {cleaned_cuts}")
    return cleaned_cuts


def is_edge_dark(edge_pixels, darkness_thresh, density_thresh):
    """Check if a significant portion of edge pixels are dark."""
    if edge_pixels is None or edge_pixels.size == 0: return False
    dark_pixels = edge_pixels[edge_pixels < darkness_thresh]
    return (dark_pixels.size / edge_pixels.size) >= density_thresh

def trim_card_iterative(pil_img, initial_threshold=220,
                       dark_thresh=75, density=0.75, max_iter=100,
                       top_scan_limit=100, top_avg_thresh=200, inset=0): # Added top_avg_thresh param
    """
    Trims image iteratively, finds dark edges, refines top edge using AVG intensity, applies inset.
    """
    cv_img = np.array(pil_img); gray = None # Define gray later
    if len(cv_img.shape) == 2: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

    # Only convert to gray if we need it (which we do)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray.shape

    # 1. Get initial bounding box (same as before)
    _, thresh = cv2.threshold(gray, initial_threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, img_w, img_h
    if contours:
        # Using largest contour might be better than combining if there's noise
        largest_contour = max(contours, key=cv2.contourArea)
        x_i, y_i, w_i, h_i = cv2.boundingRect(largest_contour)
        # all_points = np.concatenate([cnt for cnt in contours])
        # x_i, y_i, w_i, h_i = cv2.boundingRect(all_points)
        if w_i > 0 and h_i > 0: x, y, w, h = x_i, y_i, w_i, h_i
        else: print("⚠️ Initial contour box invalid.")
    else: print("⚠️ No initial contours found.")

    # 2. Iteratively shrink box (same as before, using is_edge_dark)
    x1, y1, x2, y2 = x, y, x + w, y + h
    last_y1 = y1 # Store y1 before loop for refinement check later

    for iteration in range(max_iter):
        if w <= 1 or h <= 1: print("⚠️ Box shrunk too small."); break
        current_y1=min(y1,img_h-1); current_y2=min(y2,img_h); current_x1=min(x1,img_w-1); current_x2=min(x2,img_w)
        top_edge = gray[current_y1, current_x1:current_x2] if current_y1 < current_y2 and current_x1 < current_x2 else None
        bottom_edge = gray[current_y2-1, current_x1:current_x2] if current_y2 > current_y1 and current_y2-1 < img_h and current_x1 < current_x2 else None
        left_edge = gray[current_y1:current_y2, current_x1] if current_x1 < current_x2 and current_y1 < current_y2 else None
        right_edge = gray[current_y1:current_y2, current_x2-1] if current_x2 > current_x1 and current_x2-1 < img_w and current_y1 < current_y2 else None
        top_is_dark=is_edge_dark(top_edge,dark_thresh,density); bottom_is_dark=is_edge_dark(bottom_edge,dark_thresh,density)
        left_is_dark=is_edge_dark(left_edge,dark_thresh,density); right_is_dark=is_edge_dark(right_edge,dark_thresh,density)
        if top_is_dark and bottom_is_dark and left_is_dark and right_is_dark: break
        adjusted = False
        if not top_is_dark and y1 < y2 - 1: y1 += 1; h -= 1; adjusted = True
        if not bottom_is_dark and y2 > y1 + 1: y2 -= 1; h -= 1; adjusted = True
        if not left_is_dark and x1 < x2 - 1: x1 += 1; w -= 1; adjusted = True
        if not right_is_dark and x2 > x1 + 1: x2 -= 1; w -= 1; adjusted = True
        if not adjusted and iteration > 0: break
        if iteration == max_iter - 1: print(f"⚠️ Iterative trim reached max iterations.")

    # --- 3. Refine Top Edge using AVERAGE INTENSITY ---
    y_refined = y1 # Start refinement from where the loop stopped
    # print(f"    DEBUG Refining top edge starting from y={y1}")
    for y_scan in range(y1, min(y1 + top_scan_limit, y2)): # Scan downwards limited distance
        if y_scan >= img_h: break # Stop if scan goes beyond image height
        top_edge_scan = gray[y_scan, x1:x2] if x1 < x2 else np.array([])

        if top_edge_scan.size > 0:
            avg_intensity = np.mean(top_edge_scan)
            # print(f"    DEBUG Top Scan y={y_scan}, Avg Intensity={avg_intensity:.1f} (Threshold={top_avg_thresh})") # Debug print
            # Check if average intensity is below the threshold
            if avg_intensity < top_avg_thresh:
                # Found the first row that's significantly darker than white on average
                y_refined = y_scan
                # print(f"    DEBUG Top edge refinement found avg intensity < threshold at y={y_scan}")
                break # Stop scanning
        # else: # Optional debug if scan line is empty
        #     print(f"    DEBUG Top Scan y={y_scan}, Scan line empty (x1={x1}, x2={x2})")

    if y_refined > y1:
        print(f"    Refined top edge from y={y1} to y={y_refined} based on avg intensity.")
        h -= (y_refined - y1) # Adjust height
        y1 = y_refined        # Update y1

    # 4. Apply final inset (same as before)
    if inset > 0:
        x_in, y_in = x1 + inset, y1 + inset; w_in, h_in = w-(2*inset), h-(2*inset)
        if w_in > 0 and h_in > 0: x1, y1, w, h = x_in, y_in, w_in, h_in
        else: print(f"⚠️ Inset {inset} too large. Skipping.")

    # 5. Final Crop (same as before)
    x1_final=max(0, x1); y1_final=max(0, y1); x2_final=min(img_w, x1+w); y2_final=min(img_h, y1+h)
    if x2_final > x1_final and y2_final > y1_final: cropped = cv_img[y1_final:y2_final, x1_final:x2_final]
    else: print("❌ ERROR: Final box invalid. Returning original."); cropped = cv_img
    if cropped is None or cropped.size == 0: print("❌ ERROR: Cropped empty! Returning original PIL."); return pil_img
    return Image.fromarray(cropped)


# === MAIN PROCESSING === (Loop structure is the same)
pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
if not pdf_files: print(f"No PDFs found in: {input_dir}")

card_counter = 0
for pdf_path in pdf_files:
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n📄 Processing {base_name}...")
    try: pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e: print(f"❌ Failed read/convert {pdf_path}: {e}"); continue

    for page_num, page in enumerate(pages, start=1):
        print(f"  -- Processing Page {page_num} --")
        # (Setup cv_img_bgr, gray, binary, img_h, img_w - same as before)
        cv_img_bgr = np.array(page)[:, :, ::-1]; gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY)
        img_h, img_w = gray.shape

        # (Find cuts, define rows/cols - same as before)
        row_cuts=find_cut_positions_by_whitespace(binary, axis='vertical', ratio_thresh=whitespace_ratio_thresh, min_gap=min_cut_gap)
        col_cuts=find_cut_positions_by_whitespace(binary, axis='horizontal', ratio_thresh=whitespace_ratio_thresh, min_gap=min_cut_gap)
        valid_row_cuts=sorted([c for c in row_cuts if 0 < c < img_h]) if row_cuts else []
        rows=[0]+valid_row_cuts+[img_h] if valid_row_cuts else [0,img_h//2,img_h]
        if not valid_row_cuts and row_cuts is not None: print(f"    ⚠️ WARNING: No valid row cuts. Splitting.")
        valid_col_cuts=sorted([c for c in col_cuts if 0 < c < img_w]) if col_cuts else []
        cols=[0]+valid_col_cuts+[img_w] if valid_col_cuts else [0,img_w//2,img_w]
        if not valid_col_cuts and col_cuts is not None: print(f"    ⚠️ WARNING: No valid col cuts. Splitting.")
        rows=sorted(list(set(rows))); cols=sorted(list(set(cols)))
        print(f"    Row boundaries: {rows}"); print(f"    Col boundaries: {cols}")

        num_rows_to_process = min(2, len(rows) - 1)
        print(f"    Processing first {num_rows_to_process} row(s)...")

        for i in range(num_rows_to_process): # Loop limited to 0, 1
            j = 1 # Right column index
            if j < len(cols) - 1:
                top=rows[i]; bottom=rows[i+1]; left=cols[j]; right=cols[j+1]
                print(f"    Attempting Box: Row {i}, Col {j} -> T:{top}, B:{bottom}, L:{left}, R:{right}")

                if right > left and bottom > top:
                    card_initial_crop = page.crop((left, top, right, bottom))
                    if save_debug_crops:
                        # (Save debug crop logic)
                        debug_filename = f"{base_name}_page{page_num}_card{i+1}_INITIAL_CROP.png"
                        debug_save_path = os.path.join(debug_dir, debug_filename)
                        try: card_initial_crop.save(debug_save_path); print(f"      💾 Saved initial crop: {debug_filename}")
                        except Exception as e: print(f"      ❌ FAILED save debug crop {debug_filename}: {e}")


                    if use_iterative_trim:
                         processed_card = trim_card_iterative(
                             card_initial_crop, initial_threshold=trim_threshold,
                             dark_thresh=dark_edge_threshold, density=edge_scan_density,
                             max_iter=max_shrink_iterations,
                             top_scan_limit=refine_top_edge_scan_limit, # Pass new limit
                             top_avg_thresh=refine_top_avg_threshold,   # Pass new threshold
                             inset=crop_inset
                         )
                    else:
                         # (Fallback logic - same as before)
                         print("    (Iterative trim disabled - using basic thresholding/contouring)")
                         processed_card = trim_card_iterative(
                             card_initial_crop, initial_threshold=trim_threshold,
                             dark_thresh=255, density=0.0, max_iter=1, inset=crop_inset,
                             top_avg_thresh=256 # Effectively disable avg threshold check
                         )


                    if processed_card is None or np.array(processed_card).size == 0:
                        print(f"    ❌ ERROR: Trimming empty for card {i+1}. Skipping."); continue

                    if resize_to:
                        if isinstance(processed_card, Image.Image):
                            try: processed_card = processed_card.resize(resize_to, Image.Resampling.LANCZOS)
                            except AttributeError: processed_card = processed_card.resize(resize_to, Image.LANCZOS)
                        else: print(f"    ❌ ERROR: Not PIL image before resize."); continue

                    card_counter += 1
                    filename = f"{base_name}_page{page_num}_card{i+1}.png"
                    save_path = os.path.join(output_dir, filename)
                    try: processed_card.save(save_path); print(f"      ✅ Saved: {filename}")
                    except Exception as e: print(f"      ❌ FAILED save {filename}: {e}")
                else:
                    # (Skip invalid box logic - same as before)
                    reason=[];
                    if not right > left: reason.append(f"width invalid (L{left}>=R{right})")
                    if not bottom > top: reason.append(f"height invalid (T{top}>=B{bottom})")
                    print(f"    ⚠️ Skipping box: {' and '.join(reason)}")
            else: print(f"    ⚠️ Skipping col index {j}: Not enough col boundaries ({cols}).")

print(f"\n🎉 Processing complete. {card_counter} cards saved to {output_dir}")
if save_debug_crops: print(f"ℹ️ Debug crops in: {debug_dir}")

# --- END OF MODIFIED FILE ---