# Brain-tumor-VGG16


<h3>The above Trained Model is giving a predicted percentage of 95%.<h3>

A Convolutional Neural Network, also known as CNN or ConvNet, is a class of neural networks that specializes in processing data that has a grid-like topology, such as an image.

VGG16 is a convolutional neural network (CNN) architecture that was introduced by the Visual Graphics Group (VGG) at the University of Oxford. It is part of the VGGNet family of models, which includes variations like VGG16 and VGG19. VGG16 specifically refers to a model with 16 weight layers, including 13 convolutional layers and 3 fully connected layers. The "16" in VGG16 signifies the total number of weight layers in the network.

Key characteristics of VGG16:

Architecture:

Consists of 13 convolutional layers, each followed by a ReLU (Rectified Linear Unit) activation function.
Followed by 3 fully connected layers.
Uses 3x3 convolutional filters throughout the entire network.
Filter Size:

Employs small 3x3 convolutional filters, which is a distinctive feature of the VGG architecture. Larger receptive fields are achieved by stacking multiple 3x3 convolutional layers.
Pooling Layers:

Uses max-pooling layers with 2x2 pooling windows to downsample the spatial dimensions of the input.
Fully Connected Layers:

The final three layers are fully connected layers that are responsible for making predictions based on the learned features.
Number of Parameters:

VGG16 has a relatively large number of parameters, making it computationally expensive compared to some other architectures.
VGG16 achieved excellent performance on image classification tasks and became well-known for its simplicity and effectiveness. It was one of the top-performing models in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2014.

While VGG16 is not the most computationally efficient model, its architecture and principles have influenced the design of subsequent CNN architectures. Researchers often use VGG16 as a baseline or starting point for experiments in computer vision tasks. Moreover, pre-trained versions of VGG16 are commonly used as feature extractors or as the initial layers for transfer learning on various image-related tasks.

Explanation of the Code:-

This code is aimed at preparing a dataset for training a neural network model to classify brain tumor images into four categories: 'glioma', 'meningioma', 'notumor', and 'pituitary'. The dataset is split into training and testing sets. Here's a summary of the code:

Initialization: Initialize empty lists X_train and Y_train to store image data and corresponding labels, respectively. Also, set the image_size to 150 and define the four tumor categories in the labels list.

Training Data Loading: Iterate through each label in the labels list and for each label, read images from the corresponding training folder ('C:/Users/Ayan Kanti Das/OneDrive/Desktop/brain/archive/Training'). Resize each image to the specified image_size (150x150 pixels) using OpenCV (cv2). Append the resized images to the X_train list and their corresponding labels to the Y_train list.

Testing Data Loading: Similar to the training data loading process, iterate through each label and read images from the testing folder ('C:/Users/Ayan Kanti Das/OneDrive/Desktop/brain/archive/Testing'). Resize each image and append it to the X_train list along with its label in the Y_train list.

Conversion to Numpy Arrays: Convert the lists X_train and Y_train into NumPy arrays, resulting in the final training dataset.

In summary, this code constructs a dataset for brain tumor classification by loading and resizing images from training and testing folders for each tumor category. The data is stored in NumPy arrays X_train (containing image data) and Y_train (containing corresponding labels).


Shuffling: It shuffles the order of samples in the X_train and Y_train arrays using the shuffle function from scikit-learn. This is done to ensure a random distribution of samples during training. The random_state=101 parameter is set for reproducibility.


. Train-Test Split: The code uses the train_test_split function from scikit-learn to split the dataset into training and testing sets. The test_size is set to 0.1, indicating that 10% of the data will be used for testing, and the random_state=101 is set for reproducibility.

Label Indexing for Training Data: It converts the string labels in y_train to numerical indices using the labels.index(i) function and stores the result in y_train_new. This is done to prepare the labels for neural network training.

One-Hot Encoding for Training Data: It performs one-hot encoding on the numerical labels in y_train using tf.keras.utils.to_categorical. This step is crucial for training a neural network as it transforms categorical labels into a binary matrix, making them suitable for classification tasks.

Label Indexing and One-Hot Encoding for Testing Data: The same operations are performed on the testing set (y_test) to ensure consistency in label preprocessing.

In summary, this code segment prepares the dataset for training and testing by shuffling, splitting, and transforming the labels into a suitable format for neural network training using one-hot encoding.

This neural network model appears to be a convolutional neural network (CNN) inspired by the VGG16 architecture. It consists of multiple convolutional layers with varying filter sizes, activation functions (ReLU), and max-pooling layers. The model also includes dropout layers to prevent overfitting. After the convolutional layers, there is a flattening layer followed by dense (fully connected) layers with ReLU activations. The output layer has 4 units with a softmax activation function, indicating a multi-class classification task with four possible classes. The total number of parameters in the model can be determined from the summary output.

1. *Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3))*:
   - Convolutional layer with 32 filters of size (3,3), ReLU activation function, and input shape of (150,150,3).

2. *Conv2D(64, (3,3), activation='relu')*:
   - Convolutional layer with 64 filters of size (3,3) and ReLU activation.

3. *MaxPooling2D(2,2)*:
   - Max pooling layer with a pool size of (2,2).

4. *Dropout(0.3)*:
   - Dropout layer with a dropout rate of 0.3, which helps prevent overfitting during training.

5. *Flatten()*:
   - Flattening layer to convert the 3D output to a 1D vector before the fully connected layers.

6. *Dense(512, activation='relu')*:
   - Fully connected layer with 512 units and ReLU activation.

7. *Dense(4, activation='softmax')*:
   - Output layer with 4 units (assuming a multi-class classification task) and softmax activation, indicating probabilities for each class.

The repeated pattern of convolutional layers, max pooling, and dropout is a characteristic of VGG-style architectures. The model aims to capture hierarchical features from the input images, and the fully connected layers at the end facilitate the final classification. The softmax activation in the output layer is suitable for multi-class classification problems.


Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896       
                                                                 
 conv2d_1 (Conv2D)           (None, 146, 146, 64)      18496     
                                                                 
 max_pooling2d (MaxPooling2D  (None, 73, 73, 64)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 73, 73, 64)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 71, 71, 64)        36928     
                                                                 
 conv2d_3 (Conv2D)           (None, 69, 69, 64)        36928     
                                                                 
 dropout_1 (Dropout)         (None, 69, 69, 64)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 34, 34, 64)       0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 34, 34, 64)        0         
                                                                 
 conv2d_4 (Conv2D)           (None, 32, 32, 128)       73856     
                                                                 
 conv2d_5 (Conv2D)           (None, 30, 30, 128)       147584    
                                                                 
 conv2d_6 (Conv2D)           (None, 28, 28, 128)       147584    
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 14, 14, 128)      0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 14, 14, 128)       0         
                                                                 
 conv2d_7 (Conv2D)           (None, 12, 12, 128)       147584    
                                                                 
 conv2d_8 (Conv2D)           (None, 10, 10, 256)       295168    
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 256)        0         
 2D)                                                             
                                                                 
 dropout_4 (Dropout)         (None, 5, 5, 256)         0         
                                                                 
 flatten (Flatten)           (None, 6400)              0         
                                                                 
 dense (Dense)               (None, 512)               3277312   
                                                                 
 dense_1 (Dense)             (None, 512)               262656    
                                                                 
 dropout_5 (Dropout)         (None, 512)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 2052      
                                                                 
=================================================================
Total params: 4,447,044
Trainable params: 4,447,044
Non-trainable params: 0




1. Configuring the Model:

Loss Function: The model is set up to use 'categorical_crossentropy' as its loss function. This is a common choice for multi-class classification problems, where the model needs to predict a category from multiple possibilities.
Optimizer: The Adam optimizer is chosen to update the model's weights during training. Adam is a popular and efficient algorithm that often works well in practice.
Metrics: The model will track 'accuracy' as a metric to evaluate its performance. Accuracy measures how often the model's predictions match the true labels in the dataset.
2. Training the Model:

Training Data: The model is trained on the data stored in 'X_train' and 'y_train' variables. These likely contain the input features and corresponding labels for the training set.
Epochs: The model will undergo 20 training epochs. Each epoch involves a complete pass through the entire training dataset.
Validation Split: 10% of the training data will be held out as a validation set. This set is used to evaluate the model's performance during training and help prevent overfitting.
Key Points:

The code is likely part of a neural network training process for multi-class classification.
The model is being configured and trained to make accurate predictions on unseen data.
The specific details of the model's architecture and data are not provided in this snippet.
Additional Insights:

To assess the model's performance fully, it's essential to examine the accuracy scores on both the training and validation sets.
It's often helpful to visualize the training and validation accuracy curves over epochs to monitor progress and identify potential overfitting or underfitting issues.
Consider experimenting with different hyperparameters (e.g., learning rate, number of epochs) to potentially improve performance.


Epoch 1/20
178/178 [==============================] - 342s 2s/step - loss: 1.4155 - accuracy: 0.4522 - val_loss: 0.8919 - val_accuracy: 0.6218
Epoch 2/20
178/178 [==============================] - 341s 2s/step - loss: 0.7309 - accuracy: 0.6862 - val_loss: 0.8637 - val_accuracy: 0.6203
Epoch 3/20
178/178 [==============================] - 343s 2s/step - loss: 0.5696 - accuracy: 0.7632 - val_loss: 0.6537 - val_accuracy: 0.7152
Epoch 4/20
178/178 [==============================] - 339s 2s/step - loss: 0.4790 - accuracy: 0.8056 - val_loss: 0.6340 - val_accuracy: 0.7468
Epoch 5/20
178/178 [==============================] - 337s 2s/step - loss: 0.4096 - accuracy: 0.8330 - val_loss: 0.5782 - val_accuracy: 0.7611
Epoch 6/20
178/178 [==============================] - 337s 2s/step - loss: 0.3677 - accuracy: 0.8465 - val_loss: 0.7280 - val_accuracy: 0.7373
Epoch 7/20
178/178 [==============================] - 338s 2s/step - loss: 0.3055 - accuracy: 0.8759 - val_loss: 1.0015 - val_accuracy: 0.7136
Epoch 8/20
178/178 [==============================] - 333s 2s/step - loss: 0.2882 - accuracy: 0.8861 - val_loss: 0.5848 - val_accuracy: 0.7832
Epoch 9/20
178/178 [==============================] - 330s 2s/step - loss: 0.2704 - accuracy: 0.8914 - val_loss: 1.0490 - val_accuracy: 0.6472
Epoch 10/20
178/178 [==============================] - 329s 2s/step - loss: 0.2403 - accuracy: 0.9073 - val_loss: 0.5446 - val_accuracy: 0.7848
Epoch 11/20
178/178 [==============================] - 329s 2s/step - loss: 0.2396 - accuracy: 0.9073 - val_loss: 0.5545 - val_accuracy: 0.7769
Epoch 12/20
178/178 [==============================] - 326s 2s/step - loss: 0.1925 - accuracy: 0.9300 - val_loss: 0.7220 - val_accuracy: 0.7674
Epoch 13/20
178/178 [==============================] - 326s 2s/step - loss: 0.1810 - accuracy: 0.9284 - val_loss: 0.9082 - val_accuracy: 0.7168
Epoch 14/20
178/178 [==============================] - 329s 2s/step - loss: 0.1770 - accuracy: 0.9328 - val_loss: 0.3951 - val_accuracy: 0.8718
Epoch 15/20
178/178 [==============================] - 329s 2s/step - loss: 0.1648 - accuracy: 0.9353 - val_loss: 0.7029 - val_accuracy: 0.7943
Epoch 16/20
178/178 [==============================] - 325s 2s/step - loss: 0.1671 - accuracy: 0.9390 - val_loss: 0.4136 - val_accuracy: 0.8528
Epoch 17/20
178/178 [==============================] - 328s 2s/step - loss: 0.1295 - accuracy: 0.9504 - val_loss: 0.5548 - val_accuracy: 0.8228
Epoch 18/20
178/178 [==============================] - 336s 2s/step - loss: 0.1338 - accuracy: 0.9492 - val_loss: 0.4733 - val_accuracy: 0.8402
Epoch 19/20
178/178 [==============================] - 344s 2s/step - loss: 0.1168 - accuracy: 0.9564 - val_loss: 0.7498 - val_accuracy: 0.8022
Epoch 20/20
178/178 [==============================] - 337s 2s/step - loss: 0.1350 - accuracy: 0.9494 - val_loss: 0.5879 - val_accuracy: 0.8196



![image](https://github.com/AyanKantiDas/Brain-tumor-VGG16/assets/103057066/3fdd79f2-e137-4fa9-b7f2-70c4c790bf2b)



![image](https://github.com/AyanKantiDas/Brain-tumor-VGG16/assets/103057066/856adb74-7eb1-431a-88b5-f4f3ff205f31)





