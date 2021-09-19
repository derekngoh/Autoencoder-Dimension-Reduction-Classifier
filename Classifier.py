import argparse, random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from Autoencoder_Model import Autoencoder_Model

cmdparser = argparse.ArgumentParser(description='train autoencoder')
cmdparser.add_argument('--patience', help='set number of patience for earlystop', default='10')
cmdparser.add_argument('--training_epochs', help='number of epochs to train for', default='100')
cmdparser.add_argument('--load_coders', help='name of saved autoencoder, encoder and decoder. e.g. "my_autoencoder.h5 my_encoder.h5 my_decoder.h5"', nargs="*", default=["my_autoencoder.h5", "my_encoder.h5", "my_decoder.h5"])
args = cmdparser.parse_args()


patience = int(args.patience)
training_epochs = int(args.training_epochs)
loaded_coders = args.load_coders

(X_train, y_train),(X_test, y_test) = mnist.load_data()

#PREPROCESS dataset
model = Autoencoder_Model()
model.set_xy_original_data(X_train,y_train,X_test,y_test,y_to_cat=True)
model.scale_xtrain_mnist()

#LOAD and set trained encoder. 
saved_autoencoder_name = loaded_coders[0]
saved_encoder_name = loaded_coders[1]
saved_decoder_name = loaded_coders[2]
loaded_autoencoder = load_model(model.set_current_path(saved_autoencoder_name))
loaded_encoder = load_model(model.set_current_path(saved_encoder_name))
loaded_decoder = load_model(model.set_current_path(saved_decoder_name))

model.set_autoencoder(loaded_autoencoder)
model.set_encoder(loaded_encoder)
model.set_decoder(loaded_decoder)

#ENCODE xtrain and xtest data using encoder. Turn image from [28, 28] to [5]
model.encode_xtrain_test_with_encoder()

random_num = random.randint(1,999)

#CREATE and compile model
early_stop_patience = patience
epochs = training_epochs

model.create_softmax_model()
model.fit_softmax_with_val_using_encoded_x(epochs,early_stop_patience)
model.main_model.save(model.set_current_path("classifier{a}.h5".format(a=random_num)))

#SAVE model
to_be_name_of_file = "softmax_model{a}.h5".format(a=random_num)
model.main_model.save(model.set_current_path(to_be_name_of_file))

#VIEW training validation loss plot
model.save_loss_plot("classifier_loss_plot{a}.jpg".format(a=random_num))

#PREDICT on the encoded dataset and review performance.
predictions = np.argmax(model.main_model.predict(model.encoded_xtest),axis=-1)
results = classification_report(np.argmax(model.ytest,axis=1),predictions)
model.save_text(results, "classifier_results{a}.txt".format(a=random_num))
print(results)

#Use heatmap to see the confusion matrix on the false negatives and false positives .
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(np.argmax(model.ytest,axis=1),predictions),cmap='viridis', annot=True)
plt.savefig(model.set_current_path("classifier_heatmap{a}.jpg".format(a=random_num)))
plt.show()