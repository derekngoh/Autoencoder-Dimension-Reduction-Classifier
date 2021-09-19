Dimension Reduction Classifier


**Program and Goal**
This program experimented the extent to which an image could be dimensionally reduced through encoding with an autoencoder while still maintaining an acceptable level of accuracy in terms of recognition with the encoded images. Images were first fed into an autoencoder to achieve a well-trained encode-decode model. The encoder part of the autoencoder was then used to encode the dataset to reduce their dimensions, and they were used to train a softmax model for classification. The accuracy of classifying the dimensionally-reduced images was 94%, 4% lower than the standard classification model. The dimension was reduced from [28,28] to [5]. 


**The Data**
The data used in this program was the MNIST dataset loaded from tensorflow at tensorflow.keras.datasets.


**Data Overview**
The MNIST dataset comes with a set of 70,000 images of handwritten numbers normalised to 28 pixels by 28 pixels. The dataset is split into 60,000 training images and 10,000 test images on load. For more details about the dataset, you may visit https://en.wikipedia.org/wiki/MNIST_database.


**Structure & Approach**
There were 2 models created in the course of this project. Model 1, named "autoencoder", and Model 2, named "model". The function of the autoencoder was to encode the images and reduce their dimensions while still being able to reproduce the original image after decoding. As such, the "autoencoder" itself is made up of an encoder and a decoder sub-model. The other function of "model" was to measure the accuracy of recognition of the encoded images. The result was compared with the "Standard Model" created in the CNN Standard Number Identification project. A high level of accuracy meant that the encoded image could be used in place of the original image and still produce comparatively good result.

The program is divided into the following 7 sections:

Section 1 : Explore Data 
Section 2 : Create and Fit AutoEncoder
Section 3 : Autoencoder Prediction and Evaluation
Section 4 : Data Prepocessing for Fitting with Encoded Images
Section 5 : Model Creation and Fitting
Section 6 : Model Evaluation and Prediction
Section 7 : Results Analysis


**Result Analysis**
Model has a 4% lower accuracy of 94% when compared to the Standard model (98%). However, the Standard model used images of [28,28] elements while this Reduced model uses images of [5] elements. With reference to the heat map, the model mistaked a high number of 9s for number 4s. A review of some of the images of the decoded 9s showed that the top part of 9s were frequently diminished to such an extent that it resembled a number 4. In summary, this experiment attempted to test the performance of dimension reduction of a simple autoencoder. More can done to balance the performance by reducing the reduction to improve on the prediction accuracy.


