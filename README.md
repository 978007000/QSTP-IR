# QSTP - Image Recognition
### A python script for building a Convolutional Neural Network using PyTorch

### Packages Required:
- Pytorch
- TorchVision
- Matplotlib
- Numpy

---

A Convolutional Neural Network (CNN) is a type of neural network that is mainly used in Image/Pattern recognition applications. 
It detects patterns by forming kernels (an NxN matrix) and mapping it to an area of the image (called the reception) by doing bitwise multiplication and then summation. The resulting matrix generated is called a feature map. We have many kernels applied to the original image to generate multiple feature maps. The feature maps are then compressed in size through a process called downsampling/pooling which is of 2 types:
- Max pooling
- Average pooling

We create the CNN through the following steps:
- ## Step 1: Loading the dataset FashionMNIST from Pytorch and converting it to a tensor.
- ## Step 2: Making the dataset iterable
- ## Step 3: Creating the model class
- ## Step 4: Instantiating the model class
- ## Step 5: Instantiating the loss function
- ## Step 6: Instantiating the optimizer variable
- ## Final Step: Training the model 


