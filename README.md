Deep Learning and Support Vector Machine Project
Project Overview
This project implements two models for classifying mushroom images into 9 species:

CNN Classification Model: A Convolutional Neural Network (CNN) that classifies images into 9 categories using a softmax output layer.

Feature Extraction + SVM Model: A CNN-based feature extractor followed by an SVM classifier for classification.

Dataset
The dataset contains images of mushrooms across 9 species:

Classes: Agaricus, Amanita, Boletus, Cortinarius, Entoloma, Hygrocybe, Lactarius, Russula, Suillus

Folder Structure for Training:

nginx
Copy
Edit
Mushrooms
├── Agaricus
├── Amanita
├── Boletus
├── Cortinarius
├── Entoloma
├── Hygrocybe
├── Lactarius
├── Russula
└── Suillus
Test Data: CSV with image paths and labels.

Saved Models
CNN Model: classifier.keras

Feature Extractor + SVM: svm_classifier.pkl

Model Overview
1. CNN Classification Model
Input: Images resized to 128x128x3

Layers: Conv2D, MaxPooling, GlobalAveragePooling, Dense (with softmax for 9 classes)

Loss: categorical_crossentropy, Metric: Accuracy

2. Feature Extraction + SVM Model
Purpose: Extracts features with CNN and classifies using an SVM.

Process: CNN output → feature vectors → SVM classifier.

SVM: Trained using scikit-learn’s SVC(kernel='linear').

Approach
Data Preprocessing: Loaded and resized images using tf.keras.utils.image_dataset_from_directory. Split into 80% training and 20% validation.

Data Augmentation: Applied transformations (flipping, rotation, zoom) for better generalization.

Model Building:

Constructed CNN with Conv2D, MaxPooling, Dense layers (softmax output).

Extracted CNN features, trained an SVM classifier.

Training: Model trained for 30 epochs. Saved models as .keras and .pkl files.

Testing
Setup the Python environment using requirements.txt.

Run the test scripts:

For Model 1:

bash
Copy
Edit
python proj3_classification_test.py --model <model_path> --test_csv <test_data_csv_path>
For Model 2:

bash
Copy
Edit
python proj3_extractSVM_test.py --model <model_path> --test_csv <test_data_csv_path> --feature_model <feature_model_path>
Requirements
Python 3.11.0

tensorflow 2.19.0

numpy 1.24.3

pandas 2.2.3

scikit-learn 1.6.1

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Conclusion
This project demonstrates the use of CNNs for image classification and SVMs for classification with extracted CNN features, providing flexibility for different classification strategies.
