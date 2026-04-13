import pandas as pd
import os

# --- CONFIG ---
CSV_PATH = 'data/processed/clinical_data.csv'
# We check both likely locations for your images
POSSIBLE_IMG_DIRS = ['data/raw/images', 'data/raw']

def verify():
    if not os.path.exists(CSV_PATH):
        print("❌ CSV not found. Run process_real_data.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Checking {len(df)} entries in CSV...")

    # 1. Determine the correct column name for images
    if 'Image_ID' in df.columns:
        col_name = 'Image_ID'
    elif 'image_path' in df.columns:
        col_name = 'image_path'
    else:
        print(f"❌ Error: Could not find image column. Found columns: {df.columns.tolist()}")
        return

    print(f"Using column '{col_name}' for image filenames.")

    # 2. Determine where the images actually are
    valid_img_dir = None
    for d in POSSIBLE_IMG_DIRS:
        if os.path.exists(d):
            # Check if this folder contains the first image from the CSV
            first_img = df.iloc[0][col_name]
            if os.path.exists(os.path.join(d, first_img)):
                valid_img_dir = d
                break
    
    if not valid_img_dir:
        print("❌ Could not locate the folder containing the images.")
        print(f"Checked: {POSSIBLE_IMG_DIRS}")
        print("Please check where you put your .jpg files.")
        return

    print(f"✅ Found images in: {valid_img_dir}")

    # 3. Verify all files
    missing_files = []
    for index, row in df.iterrows():
        img_name = row[col_name]
        full_path = os.path.join(valid_img_dir, img_name)

        if not os.path.exists(full_path):
            missing_files.append(img_name)

    if missing_files:
        print(f"⚠️ Warning: {len(missing_files)} files in CSV were not found on disk.")
    else:
        print(f"✅ All {len(df)} images verified successfully!")
        
    # 4. IMPORTANT: Print instructions if path changed
    if valid_img_dir == 'data/raw':
        print("\n⚠️ NOTICE: Your images are in 'data/raw'.")
        print("You MUST update IMG_DIR in 'src/train.py' and 'src/evaluate_model.py' to 'data/raw'.")
    elif valid_img_dir == 'data/raw/images':
        print("\n✅ Images are in the standard 'data/raw/images' folder. No code changes needed.")

if __name__ == "__main__":
    verify()