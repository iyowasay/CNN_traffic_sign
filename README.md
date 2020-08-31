## Traffic Sign Classifier(Convolutional Neural Networks)

Overview
---
In this project, the goal is to use deep learning and convolutional neural networks to classify traffic signs. After training and validating through TensorFlow and Keras, the model can correctly classify unseen traffic sign images. You will then try out your model on images of German traffic signs that you find on the website. 

Here is a link to my [project code](https://github.com/iyowasay/CNN_traffic_sign/blob/master/Traffic_Sign_Classifier.ipynb), and my [HTML file](https://github.com/iyowasay/CNN_traffic_sign/blob/master/Traffic_Sign_Classifier.html).

### TensorFlow and Keras

TensorFlow is an open-source machine learning and deep learning library which gives users more control over their networks. TF is usually used while doing some research works or developing special structures of neural network. On the other hand, Keras is a high level neural network library that run on top of TensorFlow. It's easy to use and allows us to quickly put ideas into practice and prototype scalable networks. 

### CNN

If we know some domain knowledge or structure about our data, how can we use the spatial structure in the input to inform the architecture of the network? This is the main purpose of convolutional neural network. It captures the spatial dependencies between pixels instead of viewing each pixel as single independent input. In addition, the amount of parameters can be drastically reduced by virtue of weight sharing across the image. With these advantages, the performance of the entire network can be improved. Note that CNN learns all of the features from the training set by its own. Generally speaking, the underlying properties of CNN is not intuitive and hard to visualize, sometimes we regard it as a black box. In this project, the network can be divided into two parts, namely CNN for feature learning and fully connected layers for classification. 

- Steps of ConvNets
1. choose parameters: kernel(patch, filters) size, stride, padding, depth(K). 
2. initalize learnable weights and biases and share across the space to achieve statistical invariance.
3. decide the number of convolutional layers in the model.
4. apply pooling layer to reduce spatial size of the representation.
5. incorporate non-linearity by using activation function(e.g. rectified linear unit).
6. train the network and make prediction by connecting the whole thing to few regular fully connected layers.
 
Notes:
* Statistical invariance(translation invariance): things that don't change on average across time or space.  
* High level feature detection: identify key characteristics in each image category.
* CNN applications - end-to-end robotic control, natural language processing(NLP), Fully Convolutional Networks(FCN) for semantic segmentation, semantic means the meaning or the label of image pixels.
* Weight sharing: when you know that two inputs can contain the same kind of information. The sharing means that the weights and bias of a single kernel are shared across all the other kernels in the same channel or feature map, since they are supposed to predict the same pattern or feature.

<!-- # smaller patch == fewer amount of weights == -->

### Dimensionality and size of the model

The dimensionality depends on the input size and the value of hyperparameters you decide. The following equation shows the relation between the height, width of the output from certain layer and those parameters. In addition, the size of the model is also controlled by the input size and these values. For instance, the purpose of pooling layer(max pooling or average pooling) is to deal with different spatial resolutions and achieve spatial invarince by downsampling.

<img src="/image/equa1.png" alt="train" width="650" height="200"/>


### Dataset exploration 

The pickled data contains resized images with the shape of (32,32,3). The distributions of the number of examples in each class are shown in the following bar charts. The summary statistics of given German Traffic Sign Dataset is calculated using the pandas library.

* number of training data: 34799
* number of validation data: 4410
* number of testing data: 12630
* number of classes/labels: 43

<img src="/image/training_bar.png" alt="train" width="560" height="330"/>

<img src="/image/validation_bar.png" alt="valid" width="560" height="330"/>

<img src="/image/testing_bar.png" alt="test" width="560" height="330"/>

<!-- <img src="/image/more.png" alt="more" width="550" height="300"/> -->

<!-- <img src="/image/less.png" alt="less" width="550" height="300"/> -->


### Data augmentation

From the above bar charts, we know that the given training dataset is imbalanced, meaning that the difference between the number of examples in each class is large. This may lead to instability of the model. In order to generate additional data, several augmentation approaches can be applied. For image classification, few different processing methods are able to create augmented images, such as rotation, color shifting, mirroring. In this project some of these methods are implemented with a view to increasing the robustness and generalizability of the model. Here all the classes which have the number of data less than 1500 are augmented. The size of training set after augmentation is 67380.

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
data_aug = ImageDataGenerator(rotation_range=13, width_shift_range=0.14, height_shift_range=0.14,shear_range=9,
                              zoom_range=0.2,channel_shift_range=15)
```

* Rotation(`rotation_range`): rotate the image within an angle range.
* Translation(`width_shift_range`, `height_shift_range`): shift the image in x or y dircetion
* Color shifting(`channel_shift_range`): Add or subtract random values from each channel to change the color of the original image.
* Scaling(`zoom_range`): zoom in or out.
* Mirroring: Horizontaly flip the image. This is not included in this project since some of the traffic signs are directional and flipping them might cause problem.

<img src="/image/training_aug_bar.png" alt="aug" width="560" height="330"/>


### Model Architecture

1. Preprocessing techniques

This step aims to prepare the input format for the CNN and hence speed up the process. First of all, the input could be normalized by taking zero means and equal variances. Normalization is usually used when the range of values(difference between maximum and minimum) between dimensions is considerably different. Secondly, converting RGB images into grayscale reduce the dimensionality. Take numbers or letters recognition as example. The color of a letter has nothing to do with the final prediction of this letter. Therefore, grayscale images contain sufficient information for training the network.


2. Layers  

My final model, which achieves around 97% validation accuracy, consist of the following layers. One can also use TensorBoard to visualize your model architecture.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   			| 
| Convolution 1 5x5     | 1x1 stride, valid padding         	|
| RELU					| outputs - 28x28x6					    |
| Convolution 2 5x5     | 2x2 stride, valid padding         	|
| RELU					| outputs - 12x12x16					    |
| Convolution 3 5x5    	| 1x1 stride, valid padding         	|
| RELU					| outputs - 8x8x30					    |
| Max pooling	      	| 2x2 stride,  outputs - 4x4x30 			|
| Flatten	            | outputs - 480 (vectorization)   			|
| Dropout layer 1       | keep_prob = 0.6                       |
| Fully connected 1	+ RELU	| outputs - 250       				|
| Fully connected 2 + RELU	| outputs - 90			    		|
| Dropout layer 2       | keep_prob = 0.7                       |
| Fully connected 3		| outputs - 43 classes         			|


3. Batch normalization

We know that normalizing the inputs can make the contour of our problem more round instead of being elongated, thereby increasing the training speed. Similarly, we can also apply normalization to the hidden layers. This is called batch norm and it is governed by two learnable parameters(like weights and biases), gamma and beta. Since we don't want the hidden layers to always have `mean = 0` and `variance = 1`, batch norm allows us to select the targeting mean and variance. However, this is not implemented in this project. 

4. Hyperparameters(e.g. batch size, numebr of epochs)

* batch size = 128
* number of epochs = 25
* learning rate = 0.00087
* beta(L2 norm) = 0.0001


The batch size affects not only the memory usage but also the accuracy of the network. Small batch size requires less memory and trains faster, but it gives less accurate estimation. The two extreme cases of batch size are stochastic gradient descent(SGD) and batch gradient descent. SGD is very noisy, might be good since it updates frequently but lose the speedup from vectorization. On the other hand, batch gradient descent take too much time per iteration. Here, in this project, mini batch gradient descent is used. It combines the advantages of the two extreme cases and make progress(update)after each batch. This gives a better overall performance.

Learning rate might be the most important parameter to be tuned. Too large or too small value will both result in bad result, or even cause the netwrok not learning at all. A suggested initial value is 0.001. One can also apply learning rate decay, meaning that the model is able to slowly reduce the learning rate over time as approaching the best estimation.


5. Flatten(vectorization)

Deep learning is a iterative and highly empirical process where we need to observe and adjust it repeatedly. Therefore, it is important to make your model run faster. Vectorization makes good use of parallel computing and avoid using for loops explicitly. This speeds up your code and results in a significant decrease of time complexity.


6. Optimizer and regularization 

Here the Adam(adaptive moment estimation) optimization algorithm is used. It's a combination of gradient descent with momentum and RMS prop. 

In order to prevent overfitting and gain generalizability, few different regularization methods can be carried out.

- Dropout

Neurons or some outputs from a certain layer are randomly ignored or deactivated. This makes the training process more noisy and forces the network to learn without heavily fitting the training dataset. It looks like the model is trying out different structures and layer connectivities during every update. The dropped proportion of a layer can be determined by the value `keep_prob`.

- L2 norm 

Add a term to the loss function, thereby incorporating the model complexity and the error penalty. This term is usually controlled by parameter lambda or BETA in my code.

- Early stopping

Stop training when the performance of validation dataset starts to decrease.   


### Model Testing

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 98.6%
* test set accuracy of 95.9%

This result is obtained by trying out several combinations of hyperparameters. The model is based on the LeNet structure proposed by Yann LeCun. Then different layers are added into the network and values have been tested iteratively. The LeNet model was originally used for recognition of handwritten numbers. Since the traffic signs are more complicated in shape and color, it is reasonable to use more layers and larger number of neurons. The figure below shows the loss and the accuracy curve on validation set.

<img src="/image/learning_curve.png" alt="aug" width="690" height="340"/>


<!-- If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? -->
 
### New images from the website

Here are some of the German traffic signs that I found on the web. 

<img src="/image/new_test.png" alt="aug" width="525" height="250"/>

The quality of chosen images is good in general. But some of them might be difficult to classify since the brightness of image is low or the image is a bit tilted. Moreover, the accuracy also depends on the resolution of input image and the architechture of your network. If, for instance, a high resolution image is resized into 32x32 to fit the model input size, lots of pixel information are lost. Therefore, it is highly possible for the model to make a wrong prediction. 

Here are the results of the prediction. The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This is lower than the accuracy on the test set. The incorrect prediction is the "No entry" sign. This is because the quality of this image is relatively worse. 

<!-- Possible reasons might be that the image quality of new data is better than the test set(thus easier to recognize) and the number of new data is much smaller than test set.  -->

<img src="/image/match.png" alt="aug" width="550" height="205"/>


The bar chart of softmax probabilities is located in the 26th cell of the Ipython notebook. The probabilities of all correctly classified images are very close or equal to 1.0. Some of the bar charts are shown below.

<img src="/image/soft1.png" alt="aug" width="650" height="370"/>

<img src="/image/soft2.png" alt="aug" width="650" height="370"/>


### Future improvement

1. Currently the input size of the model is fixed due to the given format of pickled dataset(32x32). This may affect the performance of the model but allow it to run faster. In order to increase the accuracy of unseen images, the input size could be extended to higher resolution. 

2. The visualization of convolutional layers is implemented in this project. Add the visualizing part for fully connected layers. Look into the meaning of those feature maps and gain more insights about CNN.

<img src="/image/visual.png" alt="aug" width="650" height="370"/>

3. Look into the test set performance, the distribution and the quality of incorrectly classified images. 

4. Compare the performance of different optimizers and plot the learning curves. [article](https://ruder.io/optimizing-gradient-descent/index.html#adam)

5. Try out more combination of hyperparameters or different models, such as AlexNet, VGGNet. Find more traffic sign images from Google Streetview and evaluate the result.

6. Implement early termination. It is not necessary for the model to always finish all the epochs. Instead, you can define a max number of epochs and a stop value n. The model will stop training when there's no improvement, compare with previous iteration, for the last n epochs. Or you can simply keep and save the model which gives the highest validation accuracy. This can also prevent overtraining or overfitting.

