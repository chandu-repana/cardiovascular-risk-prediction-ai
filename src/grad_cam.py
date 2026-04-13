import os

# --- 1. SUPPRESS TENSORFLOW LOGS & ONE-DNN MESSAGES ---
# This must be done BEFORE importing tensorflow to take effect.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress floating-point round-off warnings

import tensorflow as tf
import numpy as np
import cv2

# --- STANDALONE GRAD-CAM SCRIPT ---

def make_gradcam_heatmap(img_array, clinical_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        # Cast inputs to ensure they are tracked by the tape
        inputs = [tf.cast(img_array, tf.float32), tf.cast(clinical_array, tf.float32)]
        last_conv_layer_output, preds = grad_model(inputs)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4, output_path='outputs/gradcam/result.jpg'):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, superimposed_img)
    print(f"Grad-CAM saved to {output_path}")

def run_explanation(model_path, img_path, clinical_data):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train.py first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # 1. Prepare Image Tensor
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img_array = tf.expand_dims(img / 255.0, axis=0)
    
    # 2. Prepare Clinical Tensor (FIXED: Convert directly to Tensor, not Numpy)
    clinical_array = tf.convert_to_tensor([clinical_data], dtype=tf.float32)
    
    print("Predicting...")
    # Passing tensors directly is safer
    preds = model.predict([img_array, clinical_array])
    pred_label = np.argmax(preds[0])
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    print(f"Prediction: {labels[pred_label]} (Confidence: {preds[0][pred_label]:.2f})")
    
    try:
        heatmap = make_gradcam_heatmap(img_array, clinical_array, model, 'mixed10', pred_label)
        save_and_display_gradcam(img_path, heatmap)
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        print("Tip: If 'mixed10' layer not found, check model.summary()")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    MODEL_FILE = 'models/cvd_multimodal_model.h5'
    
    # 1. FIND A VALID IMAGE
    image_dir = 'data/raw/images'
    if os.path.exists(image_dir):
        files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        if files:
            TEST_IMG = os.path.join(image_dir, files[0])
            print(f"Using image: {TEST_IMG}")
        else:
            print("ERROR: No images found in data/raw/images/")
            exit()
    else:
        print("ERROR: data/raw/images folder does not exist")
        exit()

    # 2. DUMMY CLINICAL DATA (Age, Gender, SBP, DBP, BMI, Diabetes)
    TEST_CLINICAL = [0.8, 0, 0.7, 0.6, 0.5, 1] 
    
    run_explanation(MODEL_FILE, TEST_IMG, TEST_CLINICAL)