# QSTP - Image Recognition
### A python script for building a Convolutional Neural Network using PyTorch

### Packages Required:
- Pytorch
- TorchVision
- Matplotlib
- Numpy

###### Visit [Pytorch](https://pytorch.org/) for details on installation and support.
---

A Convolutional Neural Network (CNN) is a type of neural network that is mainly used in Image/Pattern recognition applications. 
It detects patterns by forming kernels (an NxN matrix) and mapping it to an area of the image (called the reception) by doing bitwise multiplication and then summation. The resulting matrix generated is called a feature map. We have many kernels applied to the original image to generate multiple feature maps. The feature maps are then compressed in size through a process called downsampling/pooling which is of 2 types:
- Max pooling
- Average pooling

We will be using 2 convolutional layers for our network.

## We create the CNN through the following steps:
- ### Step 1: Loading the dataset FashionMNIST from Pytorch and converting it to a tensor.
- ### Step 2: Making the dataset iterable
We use the DataLoader utility in Pytorch
> We set the batch size to 100
> Our number of iterations will be 3000
> The total number of epochs can be calculated as:
>number_iterations/(len(training_data)/batch_size)
- ### Step 3: Creating the model class
>We will use a kernel of 5x5 for the 2 layers. Also we use **ReLU** activation function since the output is either 0 or the number itself depending upon the sign, and is much faster since it does not involve any complex operations.
**f(x)=max(0,x)**
> 
```python
class CNNModel(nn.Module):
    def __init__(self):
        ....
    
    def forward(self,data):
        ....
    
```
- ### Step 4: Instantiating the model class
> 
```python
model=CNNModel()
```
- ### Step 5: Instantiating the loss function
>We use the CrossEntropyLoss for our CNN. **Cross-entropy** loss increases as the predicted probability diverges from the actual label. It therfore acts as a reinforcement feedback to minimise the error.
>
```python
criterion=nn.CrossEntropyLoss()
```

- ### Step 6: Instantiating the optimizer variable
>We use a learning_rate of 0.01 as a baseline, and pass in the ```model.parameters()``` to the optimizer.
- ### Final Step: Training the model 
>Now we just have to actually train the created model and find its accuracy simultaneously on the test data set.
> ##### NOTE: The training step utilizes a 100% of the CPU and heats it up pretty quick. For that case, one can opt to use the Google Colab platform to run and test their scripts [Google Colab](https://colab.research.google.com)

### Accuracy Observed: ~85%




