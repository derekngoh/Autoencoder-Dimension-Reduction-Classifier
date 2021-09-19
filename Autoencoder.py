import random, argparse
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from Autoencoder_Model import Autoencoder_Model

cmdparser = argparse.ArgumentParser(description='train autoencoder')
cmdparser.add_argument('--patience', help='set number of patience for earlystop', default='10')
cmdparser.add_argument('--training_epochs', help='number of epochs to train for', default='100')
cmdparser.add_argument('--save_as_filename', help='name for saving autoencoder', default='my')
args = cmdparser.parse_args()

save_name = args.save_as_filename
patience = int(args.patience)
training_epochs = int(args.training_epochs)


#PREPROCESS dataset
(X_train, y_train),(X_test, y_test) = mnist.load_data()
autoencoder = Autoencoder_Model()
autoencoder.set_xy_original_data(X_train,y_train,X_test,y_test)
autoencoder.scale_xtrain_mnist()

#CREATE encoder and decoder for autoencoder.
inp_shape = [28,28]
autoencoder.create_encoder_decoder(inp_shape)

random_num = random.randint(1,999)

#CREATE, compile and fit autoencoder.
early_stop_patience = patience
epochs = training_epochs
autoencoder.create_compile_autoencoder(inp_shape)
autoencoder.fit_autoencoder(early_stop_patience,epochs)

#SAVE models
autoencoder.encoder.save(autoencoder.set_current_path(save_name+"_encoder{a}.h5".format(a=random_num)))
autoencoder.decoder.save(autoencoder.set_current_path(save_name+"_decoder{a}.h5".format(a=random_num)))
autoencoder.main_model.save(autoencoder.set_current_path(save_name+"_autoencoder{a}.h5".format(a=random_num)))

#PLOT losses to see how well model fitted. Check for signs of under / overfitting.
autoencoder.save_loss_plot("autoencoder_loss_plot{a}.jpg".format(a=random_num), show=True)

#RECREATE and review 10 test images
n = random.randint(0,9)
predictions = autoencoder.main_model.predict(X_test[:10]) 

print("Recreated Image")
plt.imshow(predictions[n])
plt.show()

print("Original test image")
plt.imshow(X_test[n])
plt.show()
