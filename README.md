# Facial Keypoint Recognition
## Final Project - W207 - Applied Machine Learning
### Summer 2019
### T. Goter, C. Weyandt, and J. Yu

Repository for Final Project for Applied Machine Learning

### Facial Keypoint Detection
Link - https://www.kaggle.com/c/facial-keypoints-detection/overview

The point of this project is to use machine learning techniques in order to identify facial keypoint coordinates on images. The dataset comes from Kaggle at the link above, and the competition is over. Thus, there is a lot of information already online about different techniques that are effective for this application. However, for this project we will start at a basic level with simplistic techniques and build to a more complicated final model. We will report out metrics for our different techniques such as accuracy (as measured by the root mean squared error) and timing.

### The Data
The dataset provided by Kaggle for training contains 7049 images with labels. However, not all of the images have labels for every keypoint, of which there are 15. There are X and Y coordinates for each keypoint, so in total there are 30 values we are trying to predict for each image. For simplicity, we began by excluding all examples with a NaN for any label. This reduced our dataset to ~2000 images. We further identified three images with erroneous labels through exploratory data analysis; however, our EDA was not comprehensive so additional mislabelled images may be present in the residual training set.  Finally we separate our data into training data and development data at a rate of ~85%. The development data will be used to tune our hyperparameters and mitigate over-fitting to our training data.

### Baseline Model
Our first model simply took the  training data set of ~85% of the ~1800 images in our training set and calculated the mean X and Y coordinate for each of the 15 keypoints.  These average values were then used as our predictions for our development set.  With this simplistic approach the RMSE error was calculated to be ~3.2. This gives us basically a benchmark from which we should expect significant improvements. For reference the top score on the Kaggle leaderboards are ~1.3.

### Nearest Neighbors Models
Before jumping right into neura nets (as used by all of the top scorers), we decided to try a simplistic ML approach of using nearest neighbors regression, using both k-NR and r-NR methods. We used a Grid Search to tune hyperparameters in each case. With the k-NR methodology and a k-value of 5, we were able to achieve an RMSE of ~2.5 on our development set which is a significant improvement over the baseline methodology with ample room for improvement.












#### Helpful References
 - Neural networks and deep learning book - http://neuralnetworksanddeeplearning.com/chap1.html
 - Convolutional neural network tutorial - http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
 - Free GPU usage on Google Colab - https://hackernoon.com/begin-your-deep-learning-project-for-free-free-gpu-processing-free-storage-free-easy-upload-b4dba18abebc
 - Stanford Deep Learning Tutorial - http://deeplearning.stanford.edu/tutorial/
