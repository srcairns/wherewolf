# --- START OF FILE rename_cards.py ---

import os
import glob
import re

# === CONFIG ===
target_dir = "cards_output"  # Directory containing the images to rename
# === END CONFIG ===

print(f"Scanning directory: {target_dir}")

if not os.path.isdir(target_dir):
    print(f"Error: Directory not found: {target_dir}")
    exit()

# Get all PNG files - sorting helps ensure some consistency if multiple pages produce same names
files_to_process = sorted(glob.glob(os.path.join(target_dir, "*.png")))

if not files_to_process:
    print("No PNG files found in the directory.")
    exit()

print(f"Found {len(files_to_process)} PNG files. Starting renaming process...")

renamed_count = 0
skipped_no_match_count = 0
error_count = 0

for old_filepath in files_to_process:
    old_filename = os.path.basename(old_filepath)

    # Parse the filename
    match = re.match(r"^(.*?) - (.*?)_page\d+_card([12])\.png$", old_filename)

    if match:
        name1 = match.group(1).strip()
        name2 = match.group(2).strip()
        card_index = int(match.group(3))

        if card_index == 1:
            base_name = name1
        elif card_index == 2:
            base_name = name2
        else:
            print(f"⚠️ Skipping '{old_filename}': Invalid card index '{card_index}'.")
            skipped_no_match_count += 1
            continue

        # Clean the base name slightly (optional: remove potentially problematic characters)
        # base_name = re.sub(r'[\\/*?:"<>|]', '', base_name) # Example: Remove invalid filename chars

        # --- Check for existing files and find available name ---
        counter = 1
        new_filename = f"{base_name}.png"
        new_filepath = os.path.join(target_dir, new_filename)

        # Check if the *intended* path is the same as the *current* path
        # OR if the intended path already exists (and isn't the current file)
        while os.path.exists(new_filepath) and new_filepath != old_filepath:
            counter += 1
            new_filename = f"{base_name}{counter}.png"
            new_filepath = os.path.join(target_dir, new_filename)
            # Safety break - avoid excessively long loops if something is wrong
            if counter > 999:
                print(f"⚠️ Skipping '{old_filename}': Could not find an available name for '{base_name}' after 999 attempts.")
                error_count += 1 # Treat as an error/skip
                new_filepath = None # Signal to skip renaming
                break

        if new_filepath is None: # Check if loop was broken due to too many attempts
            continue

        # --- Perform Rename ---
        # Check if renaming is actually needed (target path might be same as original)
        if old_filepath == new_filepath:
            # print(f"ℹ️ Skipping '{old_filename}': Name '{new_filename}' is already correct.")
            # No need to count this as skipped, it's already done.
            continue

        try:
            os.rename(old_filepath, new_filepath)
            print(f"✅ Renamed '{old_filename}' -> '{new_filename}'")
            renamed_count += 1
        except OSError as e:
            print(f"❌ Error renaming '{old_filename}' to '{new_filename}': {e}")
            error_count += 1

    else:
        print(f"ℹ️ Skipping '{old_filename}': Does not match pattern.")
        skipped_no_match_count += 1


print("\n--- Renaming Summary ---")
print(f"Successfully renamed: {renamed_count}")
print(f"Skipped (no match): {skipped_no_match_count}")
print(f"Errors/Skipped (naming conflict): {error_count}")
print("------------------------")

# --- END OF FILE rename_cards.py ---