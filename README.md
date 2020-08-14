## Traffic Sign Classifier(Convolutional Neural Networks)

Overview
---
In this project, the goal is to use deep neural networks and convolutional neural networks to classify traffic signs. By training and validating through TensorFlow and Keras, the model can accurately classify unseen traffic sign images. You will then try out your model on images of German traffic signs that you find on the website. 

### TensorFlow and Keras

TensorFlow is an open-source machine learning and deep learning library which gives users more control over their networks. TF is usually used while doing some research work or developing special structure of neural network. On the other hand, Keras is a high level neural network library that run on top of TensorFlow. It's easy to use and allows us to quickly put ideas into practice and prototype scalable network. 

### CNN

If we know some domain knowledge or structure about our data, how can we use the spatial structure in the input to inform the architecture of the network? This is the main purpose of convolutional neural network. It captures the spatial dependencies between pixels instead of viewing each pixel as single independent input. In addition, the amount of parameters can be drastically reduced by virtue of weight sharing across the image. With these advantages, the performance of the entire network can be improved. Note that CNN learns all of the features from the training set by its own. Therefore, the underlying properties of CNN is not intuitive and hard to visualize, sometimes we regard it as a black box. In this project, the network can be divided into two parts, namely CNN for feature learning and fully connected layers for classification. 

- Steps of CovNets
1. choose parameters: kernel(patch, filters) size, stride, padding, depth(K). 
2. initalize learnable weights and biases and share across the space to achieve statistical invariance. 
3. apply pooling layer to reduce spatial size of the representation.
4. incorporate non-linearity by using activation function(e.g. rectified linear unit).
5. train the network and make prediction by connecting the whole thing to few regular fully connected layers.
 
Notes:
* Statistical invariance(translation invariance): things that don't change on average across time or space  
* High level feature detection: identify key characteristics in each image category.
* CNN applciations - end-to-end robotic control, natural language processing(NLP), Fully Convolutional Networks(FCN) for semantic segmentation, semantic means the meaning or the label of image pixels
* Weight sharing: when you know that two inputs can contain the same kind of information. The sharing means that the weights and bias of a single kernel are shared across all the other kernels in the same channel, since they are supposed to predict the same pattern or feature.

# smaller patch == fewer amount of weights ==

### Dimensionality and size of the model

Pooling(max pooling or average pooling): downsample to deal with different spatial resolutions and achieve spatial invarince.

<img src="/image/equa1.png" alt="train" width="200" height="200"/>



### Dataset exploration

The pickled data contains resized images with the shape of (32,32,3). The distributions of the number of examples in each class are shown in the following bar charts.
German Traffic Sign Dataset

* number of training data: 34799
* number of validation data: 4410
* number of testing data: 12630
* number of classes: 43

<img src="/image/training_bar.png" alt="train" width="200" height="200"/>

<img src="/image/validation_bar.png" alt="valid" width="200" height="200"/>

<img src="/image/testing_bar.png" alt="test" width="200" height="200"/>

<img src="/image/more.png" alt="more" width="200" height="200"/>

<img src="/image/less.png" alt="less" width="200" height="200"/>


### Data augmentation

From the above bar charts, we know that the given training dataset is imbalanced = instability, meaning that the difference between the number of examples in each class is large. In order to increase the amount of data, several augmentation approaches can be applied. For image classification, few different processing methods are able to create augmented images, such as rotation, color shifting, mirroring. In this project some of these methods are implemented with a view to increasing the robustness of the model. This can also be achieved by `tf.nn.`.

Generalizability

Rotation: rotate the image within an angle range.
Color shifting: Add or subtract random values from each channel to change the color of the original image.
Mirroring: Horizontaly flip the image 



### Model Architecture

1. Preprocessing 

This step aims to prepare the input format for the CNN and hence speed up the process. First of all, the input could be normalized by taking zero means and equal variances. Normalization is usually used when the range of values(difference between maximum and minimum) between dimensions is considerably different. Secondly, turning RGB images into grayscale reduce the dimensionality. Take numbers or letters recognition as example. The color of a letter has nothing to do with the final prediction of this letter. Therefore, grayscale images contain sufficient information for training the network.


2. Layers  

- number of layers: 
- 32x32x3 to 

3. Weights and biases and batch normalization

We know that normalizing the inputs can make the contour of our problem more round instead of being elongated, thereby increasing the training speed. Similarly, we can also apply normalization to the hidden layers. This is called batch norm and it is governed by two learnable parameters(like weights and biases), gamma and beta. Since we don't want the hidden layers to always have `mean = 0` and `variance = 1`, batch norm allows us to 

4. Bacth size, numebr of epochs

Stochastic gradient descent is very noisy, might be good since it updates frequently but lose the speedup from vectorization 
Mini batch gradient descent = make progress and update after each batch
Batch gradient descent take too much time per iteration.

5. Flatten or vectorization

Deep learning is a iterative and highly empirical process where we need to observe and adjust it repeatedly. Therefore, it is important to make your model run faster. Vectorization makes good use of parallel computing and avoid using for loops explicitly. This speeds up your code and results in a significant decrease of time complexity.

6. Optimizer and learning rate decay

Adam(adaptive moment estimation) optimization algorithm: Gradient descent with momentum + RMS prop 
Slowly reduce your learning rate over time 

7. Regularization 

- Dropout

- L2 norm 

- Early stopping


8. Result



### Model Testing



### Relection and discussion




### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```


