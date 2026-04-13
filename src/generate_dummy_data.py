import os
import pandas as pd
import numpy as np
import random

# CONFIG
IMAGE_DIR = 'data/raw/images'
OUTPUT_CSV = 'data/processed/clinical_data.csv'

def calculate_risk(age, sbp, diabetes, bmi):
    """
    Simple rule-based logic to assign Ground Truth labels 
    so the model actually has a pattern to learn.
    0: Low, 1: Medium, 2: High
    """
    score = 0
    if age > 60: score += 1
    if sbp > 140: score += 1
    if diabetes == 1: score += 1
    if bmi > 30: score += 1
    
    # Add randomness to simulate real-world noise
    noise = random.uniform(-0.5, 0.5)
    final_score = score + noise
    
    if final_score < 1.5: return 0 # Low
    elif final_score < 2.5: return 1 # Medium
    else: return 2 # High

def generate_data():
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory {IMAGE_DIR} not found. Please create it and add images.")
        return

    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) == 0:
        print("No images found! Please add retinal images to data/raw/images/")
        return

    print(f"Found {len(images)} images. Generating clinical data...")

    data = []
    for img_name in images:
        # Simulate realistic features
        age = np.random.randint(40, 85)
        gender = np.random.choice(['Male', 'Female'])
        sbp = np.random.randint(110, 180) # Systolic BP
        dbp = np.random.randint(70, 110)  # Diastolic BP
        bmi = np.random.uniform(18.5, 35.0)
        diabetes = np.random.choice([0, 1], p=[0.7, 0.3]) # 0: No, 1: Yes
        
        # Assign label based on features (semi-deterministic)
        risk_label = calculate_risk(age, sbp, diabetes, bmi)
        
        data.append([img_name, age, gender, sbp, dbp, bmi, diabetes, risk_label])

    df = pd.DataFrame(data, columns=['Image_ID', 'Age', 'Gender', 'SBP', 'DBP', 'BMI', 'Diabetes', 'Risk_Label'])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully generated {OUTPUT_CSV} with {len(df)} records.")

if __name__ == "__main__":
    generate_data()