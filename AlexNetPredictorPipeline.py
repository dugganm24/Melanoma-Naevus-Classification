import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

#Load images function
def load_images(folder_path, num_images=70, target_size=(224, 224)):
    images = []
    image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    for filename in selected_images:
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        images.append(image)
    
    images_array = np.array(images, dtype='float32')
    return images_array

#Assign labels function
def assign_data_labels(melanoma_images, naevus_images):
    melanoma_labels = np.zeros(len(melanoma_images))
    naevus_labels = np.ones(len(naevus_images))
    
    all_data = np.concatenate([melanoma_images, naevus_images], axis=0)
    all_labels = np.concatenate([melanoma_labels, naevus_labels], axis=0)
    
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    
    return all_data, all_labels

#AlexNet model definition
def AlexNet(input_shape=(224, 224, 3), num_classes=2, dropout_rate=0.5):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = SGD(learning_rate=0.001, momentum=0.9)  
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    
    return model

def main():
    melanoma_path = os.path.join("complete_mednode_dataset", "melanoma")
    naevus_path = os.path.join("complete_mednode_dataset", "naevus")
        
    melanoma_images = load_images(melanoma_path, num_images=70)
    naevus_images = load_images(naevus_path, num_images=100)[:70]  #Ensure only 70 naevus images for balance

    train_images_m, test_images_m = train_test_split(melanoma_images, test_size=0.28, random_state=42) #.28 of 70 is 20
    train_images_n, test_images_n = train_test_split(naevus_images, test_size=0.28, random_state=42) #.28 of 70 is 20
    
    data_training, data_training_labels = assign_data_labels(train_images_m, train_images_n)
    data_test, data_test_labels = assign_data_labels(test_images_m, test_images_n)
    
    data_training /= 255.0
    data_test /= 255.0

    data_training_labels_onehot = to_categorical(data_training_labels, num_classes=2)
    data_test_labels_onehot = to_categorical(data_test_labels, num_classes=2)

    best_dropout = 0
    training_accuracy_best = 0
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #Dropout rates to test
    
    print("Table 1: Dropout Rate vs Image Classification Accuracy")
    accuracy_results = []
    
    for dropout_rate in dropout_rates:
        alexnet_model = AlexNet(dropout_rate=dropout_rate)
    
        batch_size = 32
        num_epochs = 20 
        alexnet_model.fit(data_training, data_training_labels_onehot, batch_size=batch_size, epochs=num_epochs, validation_split=0.1)
        
        training_loss, training_accuracy = alexnet_model.evaluate(data_training, data_training_labels_onehot)
        print(f"Dropout Rate: {dropout_rate}, Training Accuracy: {training_accuracy * 100:.2f}%")
        accuracy_results.append((dropout_rate, training_accuracy * 100))
        
        if training_accuracy > training_accuracy_best:
            best_dropout = dropout_rate
            training_accuracy_best = training_accuracy
    
    print(f"The best dropout rate from training data is {best_dropout} with an image classification accuracy of {training_accuracy_best * 100:.2f}%")
    
    for dropout_rate, accuracy in accuracy_results:
        print(f"Dropout Rate: {dropout_rate}, Accuracy: {accuracy:.2f}%")
    
    print("Evaluating model with best dropout rate on test data...")
    alexnet_model = AlexNet(dropout_rate=best_dropout)
    alexnet_model.fit(data_training, data_training_labels_onehot, batch_size=32, epochs=20, validation_split=0.1)
    test_loss, test_accuracy = alexnet_model.evaluate(data_test, data_test_labels_onehot)
    print(f"Best Dropout Rate: {best_dropout}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
