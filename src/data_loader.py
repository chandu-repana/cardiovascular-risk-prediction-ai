import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class MultiModalDataLoader:
    def __init__(self, csv_path, image_dir, img_size=(224, 224), batch_size=32):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.df = pd.read_csv(csv_path)
        self.preprocess_clinical_data()
        
    def preprocess_clinical_data(self):
        # 1. Encode Gender
        self.df['Gender'] = self.df['Gender'].map({'Male': 0, 'Female': 1})
        
        # 2. Normalize Continuous Variables
        scaler = MinMaxScaler()
        cols_to_norm = ['Age', 'SBP', 'DBP', 'BMI']
        self.df[cols_to_norm] = scaler.fit_transform(self.df[cols_to_norm])
        
        self.clinical_cols = ['Age', 'Gender', 'SBP', 'DBP', 'BMI', 'Diabetes']
        self.num_clinical_features = len(self.clinical_cols)

    def load_image(self, img_name):
        img_path = tf.strings.join([self.image_dir, '/', img_name])
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        img = img / 255.0
        return img

    def augment_image(self, img):
        """Applies random transformations to artificially increase dataset size"""
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img

    def get_dataset(self):
        train_df, val_df = train_test_split(self.df, test_size=0.2, random_state=42, stratify=self.df['Risk_Label'])
        
        def create_tf_dataset(dataframe, is_training=False):
            image_paths = dataframe['Image_ID'].values
            clinical_data = dataframe[self.clinical_cols].values.astype(np.float32)
            labels = dataframe['Risk_Label'].values.astype(np.int32)
            labels = tf.one_hot(labels, depth=3)

            dataset = tf.data.Dataset.from_tensor_slices((image_paths, clinical_data, labels))
            
            def map_func(img_path, clin_data, label):
                img = self.load_image(img_path)
                if is_training:
                    img = self.augment_image(img) # Apply Augmentation ONLY on training
                return {'image_input': img, 'clinical_input': clin_data}, label

            dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
            
            if is_training:
                dataset = dataset.shuffle(buffer_size=1000)
                
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return dataset

        # Enable augmentation for Training set, disable for Validation set
        train_ds = create_tf_dataset(train_df, is_training=True)
        val_ds = create_tf_dataset(val_df, is_training=False)
        
        return train_ds, val_ds, self.num_clinical_features