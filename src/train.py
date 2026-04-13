import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_loader import MultiModalDataLoader
from model_builder import build_multimodal_model

# --- CONFIGURATION ---
CSV_PATH = 'data/processed/clinical_data.csv'
IMG_DIR = 'data/raw/images'  # <--- CORRECT PATH
MODEL_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'cvd_multimodal_model.h5')
HISTORY_SAVE_PATH = os.path.join(MODEL_DIR, 'training_history.pkl')
EPOCHS = 30
BATCH_SIZE = 32

def main():
    # 1. Setup Directories
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 2. Load Data
    print("Loading Data...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run process_real_data.py first.")
        return

    loader = MultiModalDataLoader(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE)
    train_ds, val_ds, clinical_dim = loader.get_dataset()

    # 3. Build Model
    print("Building Multimodal Model...")
    model = build_multimodal_model(num_clinical_features=clinical_dim)
    
    model.summary()

    # 4. Define Callbacks
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    # 5. Train
    print("Starting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 6. Save Training History
    with open(HISTORY_SAVE_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✅ Training history saved to {HISTORY_SAVE_PATH}")

    # 7. Final Save
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Final Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()