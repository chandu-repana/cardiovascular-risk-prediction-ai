import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import MultiModalDataLoader

# --- CONFIGURATION ---
HISTORY_PATH = 'models/training_history.pkl'
MODEL_PATH = 'models/cvd_multimodal_model.h5'
CSV_PATH = 'data/processed/clinical_data.csv'
IMG_DIR = 'data/raw/images'  # <--- CORRECT PATH
SAVE_DIR = 'outputs/plots'

def calculate_f1(precision, recall):
    p = np.array(precision)
    r = np.array(recall)
    return 2 * (p * r) / (p + r + 1e-7)

def generate_separate_plots():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. LOAD HISTORY
    if not os.path.exists(HISTORY_PATH):
        print(f"❌ Error: {HISTORY_PATH} not found. Run src/train.py first.")
        return
        
    with open(HISTORY_PATH, 'rb') as f:
        H = pickle.load(f)
    
    epochs = range(1, len(H['accuracy']) + 1)
    
    # GRAPH 1: TRAINING PERFORMANCE
    print("Generating Graph 1: Training Performance...")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, H['accuracy'], label='Training Acc', linewidth=2)
    plt.plot(epochs, H['val_accuracy'], label='Validation Acc', linestyle='--', linewidth=2)
    plt.title('Model Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, H['loss'], label='Training Loss', color='red', linewidth=2)
    plt.plot(epochs, H['val_loss'], label='Validation Loss', color='green', linestyle='--', linewidth=2)
    plt.title('Model Loss over Epochs', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path_1 = os.path.join(SAVE_DIR, '1_training_performance.png')
    plt.tight_layout()
    plt.savefig(save_path_1, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path_1}")

    # GRAPH 2: VALIDATION METRICS
    print("Generating Graph 2: Detailed Metrics...")
    val_f1 = calculate_f1(H['val_precision'], H['val_recall'])
    train_f1 = calculate_f1(H['precision'], H['recall'])

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, H['precision'], label='Train Precision', color='purple')
    plt.plot(epochs, H['val_precision'], label='Val Precision', linestyle='--')
    plt.title('Precision', fontsize=14)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, H['recall'], label='Train Recall', color='orange')
    plt.plot(epochs, H['val_recall'], label='Val Recall', linestyle='--')
    plt.title('Recall', fontsize=14)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1, label='Train F1', color='teal')
    plt.plot(epochs, val_f1, label='Val F1', linestyle='--')
    plt.title('F1-Score', fontsize=14)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    save_path_2 = os.path.join(SAVE_DIR, '2_validation_metrics.png')
    plt.tight_layout()
    plt.savefig(save_path_2, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path_2}")

    # GRAPH 3: CONFUSION MATRIX
    print("Generating Graph 3: Confusion Matrix...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    loader = MultiModalDataLoader(CSV_PATH, IMG_DIR, batch_size=32)
    _, val_ds, _ = loader.get_dataset()
    
    y_true = []
    y_pred = []
    
    for features, labels in val_ds:
        preds = model.predict(features, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    classes = ['Low Risk', 'Medium Risk', 'High Risk']
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title(f'Confusion Matrix (Test Set n={len(y_true)})', fontsize=16, pad=20)
    plt.ylabel('Actual Patient Status', fontsize=14)
    plt.xlabel('AI Predicted Risk Level', fontsize=14)

    save_path_3 = os.path.join(SAVE_DIR, '3_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(save_path_3, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path_3}")
    
    print("\n--- All graphs generated successfully! ---")

if __name__ == "__main__":
    generate_separate_plots()