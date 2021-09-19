import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Flatten,Reshape,Dropout
from tensorflow.keras.utils import to_categorical

import Utils

class Autoencoder_Model():
    def __init__(self) -> None:
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.encoded_xtrain = None
        self.encoded_xtest = None
        self.encoder = None
        self.decoder = None
        self.main_model = None

    def set_xy_original_data(self,xtrain,ytrain,xtest,ytest,y_to_cat=False):
        self.xtrain = np.array(xtrain)
        self.ytrain = np.array(ytrain)
        self.xtest = np.array(xtest)
        self.ytest = np.array(ytest)
        if (y_to_cat): 
            self.ytrain = np.array(to_categorical(ytrain))
            self.ytest = np.array(to_categorical(ytest))
    
    def scale_xtrain_mnist(self):
        self.scaled_xtrain = self.xtrain/255
        self.scaled_xtest = self.xtest/255

    def set_autoencoder(self,autoencoder):
        self.main_model = autoencoder

    def set_encoder(self,encoder):
        self.encoder = encoder

    def set_decoder(self,decoder):
        self.decoder = decoder

    def encode_xtrain_test_with_encoder(self):
        self.encoded_xtrain = self.encoder.predict(self.xtrain)
        self.encoded_xtest = self.encoder.predict(self.xtest)

    def create_softmax_model(self):
        model = Sequential()

        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.main_model = model

    def fit_softmax_with_val_using_encoded_x(self,epochs,patience):
        earlystop = EarlyStopping(patience=patience)
        self.main_model.fit(self.encoded_xtrain,self.ytrain, validation_data=(self.encoded_xtest,self.ytest),epochs=epochs, callbacks=[earlystop])

    def create_encoder_decoder(self,inp_shape):
        layer0 = inp_shape[0]*inp_shape[1]
        layer1 = 400
        layer2 = 200
        layer3 = 100
        layer4 = 50
        layer5 = 25
        layer6 = 10
        layer7 = 5

        input_img = Input(shape=inp_shape)
        enc0 = Flatten(input_shape=inp_shape)(input_img)
        enc1 = Dense(layer1, activation='relu')(enc0)
        enc2 = Dense(layer2, activation='relu')(enc1)
        enc3 = Dense(layer3, activation='relu')(enc2)
        enc4 = Dense(layer4, activation='relu')(enc3)
        enc5 = Dense(layer5, activation='relu')(enc4)
        enc6 = Dense(layer6, activation='relu')(enc5)
        encoded = Dense(layer7, activation='relu')(enc6)

        self.encoder = Model(input_img, encoded)
        print(self.encoder.summary())

        decoder_input = Input(shape=(layer7))
        dec7 = Dense(layer6,activation='relu')(decoder_input)
        dec6 = Dense(layer5,activation='relu')(dec7)
        dec5 = Dense(layer4,activation='relu')(dec6)
        dec4 = Dense(layer3,activation='relu')(dec5)
        dec3 = Dense(layer2,activation='relu')(dec4)
        dec2 = Dense(layer1,activation='relu')(dec3)
        dec1 = Dense(layer0,activation='sigmoid')(dec2)
        decoded = Reshape(inp_shape)(dec1)

        self.decoder = Model(decoder_input,decoded)
        print(self.decoder.summary())

    def create_compile_autoencoder(self,inp_shape):
        inp = Input(shape=(inp_shape))
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        autoenc = Model(inp,decoded)
        autoenc.compile(loss="binary_crossentropy", optimizer=SGD(lr=1.5), metrics=['accuracy'])
        self.main_model = autoenc
        return self.main_model.summary()

    def fit_autoencoder(self,patience,epochs):
        earlystop = EarlyStopping(patience=patience)
        self.main_model.fit(self.scaled_xtrain, self.scaled_xtrain, epochs=epochs, validation_data=(self.scaled_xtest, self.scaled_xtest), callbacks=[earlystop])

    def save_loss_plot(self,filename,show=False,figsize=(15,6)):
        Utils.save_show_loss_plot(self.main_model,filename,show=show,figsize=figsize)

    def set_current_path(self,filename=None):
        return Utils.get_set_current_path(filename)

    def save_text(self,content, filename):
        Utils.save_as_text_file(content,filename)
