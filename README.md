# Facial Keypoint Recognition
## Final Project - W207 - Applied Machine Learning
### Summer 2019
### T. Goter, C. Weyandt, and J. Yu

Repository for Final Project for Applied Machine Learning

### Facial Keypoint Detection
Link - https://www.kaggle.com/c/facial-keypoints-detection/overview

### Navigation of Files
There are several Jupyter Notebook files provided in this GitHub repository that have been used for various aspects of this project. This section discusses the purpose of each file. The **bolded** files contain more detailed discussion and are considered more essential to the overall project.

- **DataExploration.ipynb** - This file was used to explore and clean the training data. This file was also used to create serialized dataframes of the training images. These serialized dataframes were then read in as input to other notebooks for actual model training. The only model building included in this notebook is the "mean model" and the nearest neighbors models which are used as baselines.
- *tf_notebook_tpg.ipynb* - This notebook was used to start to build single and double layer neural nets. It was mostly used for learning how to use Keras and TensorFlow for constructing neural networks. This was also used to run a suite of sensitivity studies to begin to build user understanding of different hyperparameters.
- *cnn_notebook_tpg.ipynb* - Initial CNN models were built using this notebook. All of these studies were run on a CPU. Sensitivities included in this notebook are batch normalization, data augmentation, dropout, stride, et cetera.
- **gpu_fkd_cnn_filter.ipynb** - This notebook was created in the Google Colaboratory environment. It is set up to run with a linked Google Drive account from which the training data is read and to which the models are saved (not included in this GitHub repo). This notebook has the bulk of the AlexNet based studies, including the top performing specilist model construction. This notebook has additional discussion of the studies, results and conclusions.
- *neural_net_analysis.ipynb* - This notebook is used to visualize many of the results from many of the different results. It requires reading of the various output files that were generated during sensitivity analyses. It creates an integrated, serialized dataframe (i.e., OutputData/integrated_cnn_df.pkl) that contains the epoch data and metadata for many of the sensitivities run. It also generates a smaller aggregated data file (i.e., OutputData/aggregated_cnn_df.pkl) that contains just the minimum scores for each of the sensitivities. This data feeds the Bokeh Visualization App.
- **architecture.ipynb** - This notebook, created in the Google Colaboratory environment, is used to experiment with different CNN architectures. It is set up to run with a linked Google Drive account from which the training data is read and to which results are saved (not included in this GitHub repo). Architectures shown in this notebook inclues GoogleNet, ResNet, and MobileNet. Results are submitted to Kaggle for scoring so they can be compared with other models from other notebooks. 
- *submission_notebook.ipynb* - This notebook is used to reload the representative models from various studies, generate predictions on the Kaggle test data, and write a submission file.
- *BokehApp* - This directory contains a Python script and supporting files for running a BokehApp. The app can be run with the following command `bokeh serve --show BokehApp/` from the root directory of this Repository. The application is used to visulize the results of many of the sensitivities in an interactive manner. This was not considered to be essential to the project, but the team used this final project to also expand their visualization capabilities using Python.


#### Problem Abstract

The goal of our models are to detect facial keypoint locations on input images that are 96 by 96 pixels. In order to do this detection, the modelers choose to evaluate various Convolutional Neural Networks (CNNs) with various architectures, input transformation and hyperparameter settings. These explorations and evaluations result in a final model that is used to evaluate a fixed set of test data that has not been trained on or investigated whatsoever. Through this process the modelers develop a deeper understanding of machine learning (ML), ML applications, deep learning, data preprocessing, hyperparamter tuning and model optimization.

#### Data Source
The original training and test data is provided from Kaggle, as this was a competition in 2016. This data is explored in a separate Jupyter Notebook (DataExploration) which can be found in this [GitHub repository](https://github.com/tomgoter/w207_finalproject). Through this data exploration it was determined that many of the provided images only have labels for some of the keypoints. This means there is much data that is not useful for training. This data could be labeled manually, but that would be excessively time consuming and prone to error. Additionally, not all of the images even have all of the keypoints. It was determined that initial models would only use the training examples which have labels for all 15 facial keypoints. Unfortunately this reduced the size of the data set from >7000 to ~2100. Additional discussion of the data cleaning is included in the previously linked to Jupyter Notebook. This notebook was used to clean and pickle the data (kept in a Pandas dataframe object). This pkl file was then copied to the modeler's Google Drive account to enable easy reading with Colaboratory. This file was too large to keep in GitHub.

#### Scoring Metric
Model quality is determined through the Root Mean Squared Error (RMSE) between predictions and actuals. The  equation for this calculation is given below where the sum is over every prediction (i.e., test examples multiplied by keypoints per example.) This scoring metric will also be used as the loss function (actually just he mean squared error) for the neural network training.

$$RMSE = \sqrt{\frac{1}{n}\cdot\sum\limits_{i=1}^{n}(y-\hat{y})^2}$$

#### Baseline Models
The modeler chose to generate three baseline models from which to gage progress for more advanced models. The first of these baseline models was rudimentary. For any new test image, the mean x and y location for each keypoint was used as the keypoint location prediction. In this case the mean keypoint location was able to be determined through an average over all training data (minus those reserved as development data - ~15%) simply by ignoring the NaNs in the dataset. The resultant RMSE from this simple prediction method was 3.16. Thus, a baseline was established with plenty of room for improvement. For reference, the Public Leaderboard for the [Kaggle Competition](https://www.kaggle.com/c/facial-keypoints-detection/leaderboard) had a high score (or low in terms of RMSE) of 1.53 at the time the competition was completed.

In addition to this simplistic model, the modelers also constructed a k-nearest regressor and an r-nearest regressor model with tuned hyperparameters to determine how much better than the baseline model one could get with a simple, but slightly more advanced model than the baseline. The k-nearest regressor model was implemented, and a k-value of *five* was determined to be optimal based on testing on ~15% of the data reserved as a development set. This model merely identifies the five closest images in the training set to the target image. Image proximity is determined through the euclidean distance in pixel values for all 9,216 pixels. The keypoint locations of the target image are then estimated to be the average of the k (five in our case) nearest images. Using this model, the validation set RMSE was determined to be 2.49. Thus, a substantial improvement over the baseline model was achieved with relative ease. The r-nearest regressor model follows the same basic concept as the k-nearest regressor model, but instead of predetermining the number of images from which to average keypoint locations, instead a radial "distance" is specified. The target image keypoints are then predicted to be the average of all images found within the pre-specified distance. For the model in question, r was determined to be optimized at an approximate distance of 12. This resulted in a RMSE on the validation set of ~2.68 which is worse than the k-nearest regressor model. Due to this depreciation in accuracy and given the additional training time of this model, it was determined that a reasonable baseline+ model was the k-nearest regressor model.

#### Single Layer Neural Nets
On the modeling path toward the CNNs, the modelers made a pit stop at the single layer neural net. This was done to get more familiarity with different training options and simply building neural nets in tensor flow (with Keras as abstraction) and optimizing neural nets. Several sensitivities to the number of hidden units, optimizer and activation function were evaluated. All of these sensitivities were interesting, but in general the modelers found that the single layer models were underperforming and overfitting. Investigating the differences in gradient descent optimizers was as interesting outcome of this pit stop, and through these evaluations further emphasis was placed mostly on the stochastic gradient descent (SGD) optimizer with Nesterov (look-ahead) momentum and the ADAM optimizer which adapts training rates for each parameter in the model.

#### Double Layer Nerual Nets
If single layer neural nets was a pit stop, the modelers foray into double layer perceptrons was more of a rolling-stop past a stop sign. Fruitful results were not obtained; overfitting and underperforming continued. The modelers determined perhaps there is a reason people are using CNNs for image detection after all. Thus, emphasis was next placed on building deeper neural nets. This is discussed in the next section.

#### Convolutional Neural Nets
Convolutional neural nets typically make use of a deep network architecture with limited breadth. The basic principal for image processing is to start with your image pixel values, and then convolve the image through a smaller kernel matrix such as shown in the image below. Each kernel matrix cell is initialized with a random weight, and these weights are learned by training the model. This kernel slides over the input feature matrix as shown by the image below. As this process continues the features are transformed. This transformation continues in layers. In between convolution layers the feature set size is typically decreased through a pooling layer in which another kernel matrix is used (many times 2x2) to identify either the average feature value (average pooling) or the maximum feature (max pooling) from this small grid. This process results in a greatly reduced number of dimensions (e.g., if using a 2x2 pooling layer, the dimensions are halved). This process of convolving and reducing dimensions basically allows the neural net to slowly transform the starting feature space into higher level items it can use to classify (or in this case regress). After several convolutions and poolings, the remaining dimensions are flattened and passed to one or two fully connected layers. These output layers are then used to perform the regression by narrowing down to 30 outputs. Note that for classification it is typically to see a softmax (version of a sigmoid function) as the output activation function, but for this regression problem a floating point output is desired. As such, a linear (identity) activation function is used for the final output layer.


![alt text](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/padding_strides.gif)

The basic principles and conventions behind building a CNN can be summarized by the following bullets.



*   Small filters are typically used first (i.e., 2x2 to 5x5 or thereabouts) to gather the local information from the base image.
*   As we step deeper into the network, the filter width can be expanded. The theory behind this is that as we go deeper into the network we begin to explore more global features as we have already captured the local features.
* In terms of filter depth, start with a lowish number (it seems like 8 to 32 is fairly common), and increase as depth is added
* Layers are added until overfitting starts to occur. Overfitting can then be addressed through various methods. One can implement dropout rates or regularization fairly easily at different levels throughout the network to help address this.
* There are many existing architectures from which to start from, as discussed below.

**AlexNet Inspired Models**

The modelers chose to begin with a simple AlexNet-inspired architecture which [has it roots in the early 1980s](https://www.analyticsvidhya.com/blog/2017/08/10-advanced-deep-learning-architectures-data-scientists/).  This architecture, in the grand scheme of different CNN architectures, is quite simple and easy to understand. It was the natural starting point for this regression analysis and is quite quick to train on the Colaboratory GPUs. This architecture is outlined in the figure below. The basic principle of this architecture that was followed for this analysis is to simply alternate convolution and pooling layers. For the models developed herein, three convolution-pooling layers were used. During each convolution layer, kernel depth is increased. This basically means that several independent kernels, each with unique weights to learn, are used during the training. Through this process, as shown by the image below, the feature space is reduced in the original dimensions (let's call them height and width) but increased in the depth dimension. The original feature space is replaced by these learned transformations. At each layer the user can specify kernel size, filter depth, dropout rate, and a variety of other parameters that all help in the model tuning process. During the model building discussed herein sensitivities were run to many of these hyperparameters, and in the end an optimized AlexNet model was developed. This model scored an RMSE of ~1.25 on the validation data. Unfortunately when scored against the actual Kaggle test data, it only achieved a 2.87 RMSE which would be about 53rd on the leaderboard. This result led to development and testing of additional model architectures as discussed below.

![alt text](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/08131757/temp6.png)

**VGG Net Inspired Models**

As an alternative to starting with something that resembles an AlexNet architecture, the modelers also decided to construct a model inspired by the [VGG Net](https://www.analyticsvidhya.com/blog/2017/08/10-advanced-deep-learning-architectures-data-scientists/). As compared to the AlexNet, the VGG Net is much deeper. Its basic idea is to layer many convolutions prior to pooling and applying dimensionality reduction. This results in more parameters to train and longer model training times. A basic representation of a VGG Net is shown in the image below. Overall the results of the VGG Nets were comparable to the AlexNet-inspired models, but the additional runtime (~factor of 8) from the VGG-inspired nets was basically just wasted GPU resources. This path was investigated for a few different sensitivities, but after non-inspiring results, it was abandoned in favor of additional AlexNet studies. For reference, the modelers' best VGG-inspired net resulted in an RMSE of 2.92 on the Kaggle test data.

![alt text](https://tech.showmax.com/2017/10/convnet-architectures/image_0-8fa3b810.png)

**Specialist Models**

The AlexNet results both in terms of efficiency and accuracy led the modelers to continue building more specialized model. Instead of building one model that tries to accurately locate the x- and y-coordinates of all 15 facial keypoints, the decision was made to instead build 15 models. There are two significant benefits and one significant deleterious effect of this decision.  First, the good parts:

1. Specialized models will allow the modelers to use all the training data. As previously noted, there were only about 2100 examples with all locations labeled. However some keypoints had > 7000 labels. Thus, the dataset for the combined models was much smaller than it really could be if all data were used. There should be a benefit to using all of the training data for the creation of these specialized models.
2. Specialized models were expected to be more accurate, even without additional training data, because the point of the entire network is only to identify a single keypoint. Thus, all of the training and learning done by the model is hyper-focused on the individual output. This also allows the modeler to tune hyperparameters for different keypoints, as necessary.

But of course the downside must also be discussed.

The specialized models will increase our model training time by at least a factor of 15. In addition to this, if the modelers are to tune each model, the amount of hands-on time is greatly increased. This is not an insignificant challenge to overcome. The modelers tried to balance this downside by first running all 15 models, starting with the combined model weights and hyperparameters. This method by itself resulted in very significant improvements in generalization of the model. The predictions from this model were evaluated against the Kaggle test set and determined to have an RMSE of 2.18 - an improvement of ~0.7 from the combined AlexNet model!

The accuracies of the specialized models were then inspected and ranked and focused effort was placed on improving the worst performing models, as shown below. This ranking allowed the modelers to determine which keypoints to prioritize. From the list below, it appears that significant benefit could come just from improving the predictions of the first two keypoints (i.e., mouth_center_bottom_lip and nose_tip).

1.   mouth_center_bottom_lip 2.90
2.   nose_tip  2.65
3.   left_eye_center 2.04
4.   right_eye_center           1.94
5.   left_eyebrow_outer_end     1.86
6.   right_eyebrow_outer_end    1.72
7.   right_eyebrow_inner_end    1.64
8.   mouth_right_corner         1.54
9. left_eyebrow_inner_end     1.51
10. mouth_center_top_lip       1.48
11. mouth_left_corner          1.42
12. right_eye_outer_corner     1.40
13. left_eye_outer_corner      1.30
14. right_eye_inner_corner     1.10
15. left_eye_inner_corner      1.05

