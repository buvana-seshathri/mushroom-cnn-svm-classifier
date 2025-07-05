import pandas as pd
import argparse
import tensorflow as tf
import os

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
CLASS_NAMES = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model

def decode_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0, 1]
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
    return tf.stack(images), tf.keras.utils.to_categorical(labels, num_classes=len(CLASS_NAMES))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='trial_model.keras')
    parser.add_argument('--test_csv', type=str, default='mushrooms_test.csv')
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    test_images, test_labels = get_test_data(test_df)

    model = load_model(args.model)
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test Accuracy: {acc*100:.2f}%")
