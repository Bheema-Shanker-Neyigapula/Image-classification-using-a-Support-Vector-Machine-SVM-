# Import necessary libraries
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import numpy as np

# Function to load images and labels from a specified directory
def load_images_and_labels(directory):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.flatten())
                labels.append(int(filename.split("_")[0]))  # Assuming the label is the first part of the filename

    return np.array(images), np.array(labels)

# Load images and labels from two different classes (e.g., class 0 and class 1)
class0_images, class0_labels = load_images_and_labels("path/to/class0")
class1_images, class1_labels = load_images_and_labels("path/to/class1")

# Concatenate the data from both classes
X = np.concatenate((class0_images, class1_images), axis=0)
y = np.concatenate((class0_labels, class1_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Support Vector Machine (SVM) classifier
svm_classifier = SVC()

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = svm_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Now, you can use this trained model for classifying new images.
# For example, if you have a new image 'new_image.jpg', you can use the following code:
new_image = cv2.imread("path/to/new_image.jpg", cv2.IMREAD_GRAYSCALE)
if new_image is not None:
    new_image_flattened = new_image.flatten()
    prediction = svm_classifier.predict([new_image_flattened])
    print(f"Prediction for new_image.jpg: {prediction}")
else:
    print("Failed to load new_image.jpg")
