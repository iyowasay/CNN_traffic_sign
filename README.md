## Traffic Sign Classifier(Convolutional Neural Networks)

Overview
---
In this project, the goal is to use deep neural networks and convolutional neural networks to classify traffic signs. By training and validating through TensorFlow and Keras, the model can accurately classify unseen traffic sign images. You will then try out your model on images of German traffic signs that you find on the website. 

Here is a link to my [project code](https://github.com/iyowasay/CNN_traffic_sign/blob/master/Traffic_Sign_Classifier.ipynb)

### TensorFlow and Keras

TensorFlow is an open-source machine learning and deep learning library which gives users more control over their networks. TF is usually used while doing some research work or developing special structure of neural network. On the other hand, Keras is a high level neural network library that run on top of TensorFlow. It's easy to use and allows us to quickly put ideas into practice and prototype scalable network. 

### CNN

If we know some domain knowledge or structure about our data, how can we use the spatial structure in the input to inform the architecture of the network? This is the main purpose of convolutional neural network. It captures the spatial dependencies between pixels instead of viewing each pixel as single independent input. In addition, the amount of parameters can be drastically reduced by virtue of weight sharing across the image. With these advantages, the performance of the entire network can be improved. Note that CNN learns all of the features from the training set by its own. Therefore, the underlying properties of CNN is not intuitive and hard to visualize, sometimes we regard it as a black box. In this project, the network can be divided into two parts, namely CNN for feature learning and fully connected layers for classification. 

- Steps of ConvNets
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

<img src="/image/equa1.png" alt="train" width="650" height="200"/>


### Dataset exploration 

The pickled data contains resized images with the shape of (32,32,3). The distributions of the number of examples in each class are shown in the following bar charts. The summary statistics of given German Traffic Sign Dataset is calculated using the pandas library.

* number of training data: 34799
* number of validation data: 4410
* number of testing data: 12630
* number of classes/labels: 43

<img src="/image/training_bar.png" alt="train" width="450" height="450"/>

<img src="/image/validation_bar.png" alt="valid" width="450" height="450"/>

<img src="/image/testing_bar.png" alt="test" width="450" height="450"/>

<img src="/image/more.png" alt="more" width="550" height="300"/>

<img src="/image/less.png" alt="less" width="550" height="300"/>


### Data augmentation

From the above bar charts, we know that the given training dataset is imbalanced = instability, meaning that the difference between the number of examples in each class is large. In order to increase the amount of data, several augmentation approaches can be applied. For image classification, few different processing methods are able to create augmented images, such as rotation, color shifting, mirroring. In this project some of these methods are implemented with a view to increasing the robustness of the model. This can also be achieved by `tf.nn.`.

Generalizability

generate additional data

Rotation: rotate the image within an angle range.
Color shifting: Add or subtract random values from each channel to change the color of the original image.
Mirroring: Horizontaly flip the image 



### Model Architecture

1. Preprocessing techniques

This step aims to prepare the input format for the CNN and hence speed up the process. First of all, the input could be normalized by taking zero means and equal variances. Normalization is usually used when the range of values(difference between maximum and minimum) between dimensions is considerably different. Secondly, converting RGB images into grayscale reduce the dimensionality. Take numbers or letters recognition as example. The color of a letter has nothing to do with the final prediction of this letter. Therefore, grayscale images contain sufficient information for training the network.


2. Layers  

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    |       									|
| Fully connected		|         									|
| Softmax				|         									|
|						|												|
| 						|												|

- number of layers: 
- 32x32x3 to 

Connectivity, layer size

or use a diagram (google draw?)

3. Weights and biases and batch normalization

We know that normalizing the inputs can make the contour of our problem more round instead of being elongated, thereby increasing the training speed. Similarly, we can also apply normalization to the hidden layers. This is called batch norm and it is governed by two learnable parameters(like weights and biases), gamma and beta. Since we don't want the hidden layers to always have `mean = 0` and `variance = 1`, batch norm allows us to 

4. Hyperparameters(e.g. bacth size, numebr of epochs)

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

* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

### Relection and discussion




### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```


