import pandas as pd
import argparse
import tensorflow as tf
import numpy as np
import joblib

IMG_HEIGHT, IMG_WIDTH = 128, 128
CLASS_NAMES = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

def load_feature_extractor(model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model

def decode_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize
    return img

def get_test_data(df):
    images = []
    labels = []
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    for _, row in df.iterrows():
        img = decode_img(row['image_path'])
        label = label_to_index[row['label']]
        images.append(img)
        labels.append(label)
    return tf.stack(images), np.array(labels)

def extract_features(model, images, batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        feats = model.predict(batch)
        features.append(feats)
    return np.vstack(features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_model', type=str, default='feature_extractor.keras')
    parser.add_argument('--model', type=str, default='svm_classifier.pkl')
    parser.add_argument('--test_csv', type=str, default='mushrooms_test.csv')
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    test_images, test_labels = get_test_data(test_df)

    # Load models
    feature_model = load_feature_extractor(args.feature_model)
    svm = joblib.load(args.model)

    # Extract features
    X_test = extract_features(feature_model, test_images)

    # Predict using SVM
    y_pred = svm.predict(X_test)

    acc = np.mean(y_pred == test_labels)
    print(f"SVM + Feature Extractor Accuracy: {acc*100:.2f}%")
