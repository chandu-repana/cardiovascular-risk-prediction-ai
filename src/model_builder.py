import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

def build_multimodal_model(num_clinical_features, img_shape=(224, 224, 3), num_classes=3):
    # --- BRANCH 1: IMAGE (InceptionV3) ---
    image_input = Input(shape=img_shape, name='image_input')
    
    # Load Pretrained InceptionV3 (exclude top classification layers)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=image_input)
    
    # Freeze base model layers initially
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    image_features = Dropout(0.3)(x)

    # --- BRANCH 2: CLINICAL (MLP) ---
    clinical_input = Input(shape=(num_clinical_features,), name='clinical_input')
    
    y = Dense(32, activation='relu')(clinical_input)
    y = Dense(16, activation='relu')(y)
    clinical_features = Dropout(0.2)(y)

    # --- FUSION ---
    combined = Concatenate()([image_features, clinical_features])
    
    z = Dense(64, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax', name='risk_output')(z)

    # --- CREATE MODEL ---
    model = Model(inputs=[image_input, clinical_input], outputs=output)
    
    # --- COMPILE WITH METRICS FOR GRAPHS ---
    # We added Precision and Recall here so they appear in the history for plotting
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    
    return model