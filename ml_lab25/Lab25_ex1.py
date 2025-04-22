#MULTICLASS CLASSIFICATION
#Use CIFAR10 dataset and develop a ML model for image classification using kNN

import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#DATASET PREPARATION
def load_images(path,image_size=(32,32)):

     #[When we have image clf problems where we work with folders as class labels we can use os.listdir(path)
    #os.listdir(path) returns the list of file and folder names in the directory
    #os.path.join() - this joins  multiple parts of a file into one valid path, using the correct separator for any os]

    images=[] #stores the image arrays
    labels=[] #stores the labels/class for each image
    class_names=sorted(os.listdir(path)) #keeps all the folders in alphabetical order

    for class_name in class_names: #for each folder like /airplane,/cat..
        class_path=os.path.join(path,class_name)  #building the full path to that class eg./train/frog or /test/cat
        if not os.path.isdir(class_path): #this prevents the loop frm breaking if a file/folder accidentally exists
            continue
        for fname in os.listdir(class_path): #loops through every png image
            if fname.endswith(".png"):
                image_path=os.path.join(class_path,fname) #building img path
                img=Image.open(image_path).resize(image_size) #opens the image and resizes it
                img_array=np.array(img)
                #converts image to a numpy array like shape:(32,32,3)- meaning 32 rows,32 cols,3 color channels (RGB)
                if img_array.shape != (32,32,3):
                    continue #ensures if the image is valid (not a greyscale)
                images.append(img_array)
                labels.append(class_name)
    return np.array(images), np.array(labels)

def KNNClassifier(X_train,y_train,n_neighbors=223): #k=sqrt(N) where N-training samples
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    return knn
def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
def show_predictions(images, true_labels, predicted_labels, label_encoder, num_images=10):
    #this function is for us to visualise random samples with it's true and predicted labels
    plt.figure(figsize=(15, 4))
    for i in range(num_images):
        idx = np.random.randint(0, len(images))
        img = images[idx]
        true_label = label_encoder.inverse_transform([true_labels[idx]])[0]
        pred_label = label_encoder.inverse_transform([predicted_labels[idx]])[0]
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f"T:{true_label}\nP:{pred_label}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_misclassified(images, true_labels, predicted_labels, label_encoder, num_images=10):
    misclassified_idxs = np.where(true_labels != predicted_labels)[0]
    if len(misclassified_idxs) == 0:
        print("No misclassified samples to show!")
        return
    num_images = min(num_images, len(misclassified_idxs))
    plt.figure(figsize=(15, 4))
    for i in range(num_images):
        idx = misclassified_idxs[i]
        img = images[idx]
        true_label = label_encoder.inverse_transform([true_labels[idx]])[0]
        pred_label = label_encoder.inverse_transform([predicted_labels[idx]])[0]
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
def main():
    train_folder = "cifar10/train"
    test_folder = "cifar10/test"

    print("Loading training data...")
    X_train, y_train = load_images(train_folder)

    print("Loading test data...")
    X_test, y_test = load_images(test_folder)

    print("Training KNN classifier...")
    # model = KNNClassifier(X_train, y_train, n_neighbors=3)
    # Flatten the images to 1D vectors (e.g., 32*32*3 = 3072)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1)) #w/o normalizing it takes time to run, because images of different pixels are compared

    # Encode class labels to integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    print("Training KNN classifier...")
    model = KNNClassifier(X_train, y_train, n_neighbors=4)

    print("Evaluating on test data...")
    accuracy = evaluate_model(model, X_test, y_test)

    print(f"Test Accuracy: {accuracy:.2f}")
    # #FOR DISPLAYING RANDOM IMAGES WITH ITS TRUE AND PREDICTED CLASS
    # # Reshape test images back for display
    # X_test_images, _ = load_images(test_folder)  # Re-load images in (32, 32, 3) format
    # y_pred = model.predict(X_test)
    # show_predictions(X_test_images, y_test, y_pred, le)

    #TO VIEW MISCLASSIFIED SAMPLES
    # # Get predictions
    # y_pred = model.predict(X_test)
    #
    # # Re-load test images in (32, 32, 3) format for visualization
    # X_test_images, _ = load_images(test_folder)
    #
    # # Show misclassified samples
    # show_misclassified(X_test_images, y_test, y_pred, le, num_images=10)
if __name__ == "__main__":
    main()