import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import MultiModalDataLoader

# --- CONFIGURATION ---
MODEL_PATH = 'models/cvd_multimodal_model.h5'
CSV_PATH = 'data/processed/clinical_data.csv'
IMG_DIR = 'data/raw/images'  # <--- CORRECT PATH
PLOT_SAVE_PATH = 'outputs/plots/confusion_matrix.png'

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    print("Loading model and test data...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    loader = MultiModalDataLoader(CSV_PATH, IMG_DIR, batch_size=32)
    _, val_ds, _ = loader.get_dataset()
    
    y_true = []
    y_pred = []

    print("Generating predictions on the test set...")
    for features, labels in val_ds:
        preds = model.predict(features, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title('Confusion Matrix: Cardiovascular Risk Classification', fontsize=16, pad=20)
    plt.ylabel('Actual Patient Status', fontsize=14, labelpad=10)
    plt.xlabel('AI Predicted Risk Level', fontsize=14, labelpad=10)
    
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
    plt.savefig(PLOT_SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion Matrix saved to: {PLOT_SAVE_PATH}")
    
    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=classes, zero_division=0))
    
    plt.show()

if __name__ == "__main__":
    evaluate()