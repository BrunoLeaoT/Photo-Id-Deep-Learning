from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import backend as K
import keras

class NetworkSpecies:
    def __init__(self):
        self.inputShape = (150,150,1)
        self.model = Sequential()
        
    def build_model(self):
        opt = keras.optimizers.Adam(learning_rate=0.00001)

        self.model.add(Conv2D(32, (3, 3), input_shape=self.inputShape))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    def load_weights(self):
        self.model.load_weights('assets/my_model_weights.h5')
        return self.model


