Dimension Reduction Classifier

**Program and Goal**
This program experimented the extent to which an image could be dimensionally reduced through encoding with an autoencoder while still maintaining an acceptable level of accuracy in terms of recognition with the encoded images. Images were first fed into an autoencoder to achieve a well-trained encode-decode model. The encoder part of the autoencoder was then used to encode the dataset to reduce their dimensions, and they were used to train a softmax model for classification. The accuracy of classifying the dimensionally-reduced images was 94%, 4% lower than the standard classification model. The dimension was reduced from [28,28] to [5].

**The Data**
The data used in this program was the MNIST dataset loaded from tensorflow at tensorflow.keras.datasets.


**Data Overview**
The MNIST dataset comes with a set of 70,000 images of handwritten numbers normalised to 28 pixels by 28 pixels. The dataset is split into 60,000 training images and 10,000 test images on load. For more details about the dataset, you may visit https://en.wikipedia.org/wiki/MNIST_database.


**How to Run**
Download the following 4 Python files and the 3 trained model h5 file:
1. Autoencoder.py
2. Classifier.py
3. Autoencoder_Model.py
4. Utils.py
5. my_autoencoder.h5
6. my_encoder.h5
7. my_decoder.h5
 
To train a new autoencoder, follow the steps below. If not training new autoencoder, train classifier directly with exisiting autoencoder.
1. In the terminal, cd into the directory where the downloaded files are. Make sure files are organised as shown.
2. Enter command "python Autoencoder.py [-h] [--patience PATIENCE] [--training_epochs TRAINING_EPOCHS] [--save_as_filename SAVE_AS_FILENAME]" 
3. Example: python Autoencoder.py --patience 10 --training_epochs 100 --save_as_filename my

To train the classifier, follow the steps below.
1. In the terminal, cd into the directory where the downloaded files are. Make sure files are organised as shown.
2. Enter command "python Classifier.py [-h] [--patience PATIENCE] [--training_epochs TRAINING_EPOCHS] [--load_coders [LOAD_CODERS ...]]" 
3. Example: python Classifier.py --patience 10 --training_epochs 100 --load_coders my_autoencoder.h5 my_encoder.h5 my_decoder.h5


**Result Analysis**
Model has a 4% lower accuracy of 94% when compared to the Standard model (98%). However, the Standard model used images of [28,28] elements while this Reduced model uses images of [5] elements. With reference to the heat map, the model mistaked a high number of 9s for number 4s. A review of some of the images of the decoded 9s showed that the top part of 9s were frequently diminished to such an extent that it resembled a number 4. In summary, this experiment attempted to test the performance of dimension reduction of a simple autoencoder. More can done to balance the performance by reducing the reduction to improve on the prediction accuracy.
