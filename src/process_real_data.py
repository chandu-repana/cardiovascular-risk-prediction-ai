import pandas as pd
import numpy as np
import os
import random

# CONFIGURATION
# We check both locations just in case the move command failed earlier
POSSIBLE_PATHS = ['data/raw/data.xlsx', 'data.xlsx']
OUTPUT_CSV = 'data/processed/clinical_data.csv'
IMAGE_DIR = 'data/raw'

def get_excel_path():
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            return path
    return None

def calculate_risk(age, sbp, diabetes, bmi):
    """
    Assigns a Cardiovascular Risk Label (0, 1, 2) based on vitals.
    """
    score = 0
    if age > 60: score += 1
    if sbp > 140: score += 1
    if diabetes == 1: score += 1
    if bmi > 30: score += 1
    
    noise = random.uniform(-0.3, 0.3)
    final_score = score + noise
    
    if final_score < 1.5: return 0  # Low Risk
    elif final_score < 2.5: return 1  # Medium Risk
    else: return 2  # High Risk

def process_metadata():
    excel_path = get_excel_path()
    if not excel_path:
        print(f"Error: Could not find 'data.xlsx' in data/raw/ or the root folder.")
        print("Please ensure you have downloaded the ODIR-5K excel file.")
        return

    print(f"Reading Excel file from: {excel_path}...")
    try:
        # Load specific columns usually found in ODIR-5K
        df_raw = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        print("Make sure you installed openpyxl: pip install openpyxl")
        return

    processed_data = []
    
    # Get list of actual images in folder
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory {IMAGE_DIR} not found.")
        return

    available_images = set(os.listdir(IMAGE_DIR))
    print(f"Found {len(available_images)} images in folder.")

    print("Processing rows...")
    for index, row in df_raw.iterrows():
        # 1. Extract Real Demographics
        # Adjust column names if your specific excel version differs
        age = row.get('Age', 50) 
        gender_str = row.get('Sex', 'Male') 
        gender = 'Male' if 'Male' in str(gender_str) else 'Female'
        
        # 2. Simulate Missing Vitals (BP, BMI, Diabetes) based on Age
        age_factor = (age - 30) / 50.0 
        
        sbp = int(np.random.normal(120 + (20 * age_factor), 15))
        sbp = np.clip(sbp, 110, 180)
        
        dbp = int(np.random.normal(80 + (10 * age_factor), 10))
        dbp = np.clip(dbp, 70, 110)
        
        bmi = np.random.normal(25 + (5 * age_factor), 4)
        bmi = np.clip(bmi, 18.5, 35.0)
        
        diabetes_prob = 0.2 + (0.3 * age_factor)
        diabetes = 1 if np.random.random() < diabetes_prob else 0
        
        risk_label = calculate_risk(age, sbp, diabetes, bmi)
        
        # 3. Map to Images
        left_img = row.get('Left-Fundus', None)
        right_img = row.get('Right-Fundus', None)
        
        if left_img and left_img in available_images:
            processed_data.append([left_img, age, gender, sbp, dbp, bmi, diabetes, risk_label])
            
        if right_img and right_img in available_images:
            processed_data.append([right_img, age, gender, sbp, dbp, bmi, diabetes, risk_label])

    # Save
    columns = ['Image_ID', 'Age', 'Gender', 'SBP', 'DBP', 'BMI', 'Diabetes', 'Risk_Label']
    final_df = pd.DataFrame(processed_data, columns=columns)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Success! Processed {len(final_df)} images.")
    print(f"Saved training data to: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_metadata()