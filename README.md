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





