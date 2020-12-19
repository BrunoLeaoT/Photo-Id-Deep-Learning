from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Dropout,BatchNormalization,Layer,GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam

class NetworkHumpback:
    def __init__(self):
        self.inputShape = (200,200,1)
    def build_network(self, embeddingsize):
        # Adding layers in the Convolutional Neural Network
        network = Sequential()
        
        network.add(Conv2D(64, (10,10), activation='relu',input_shape=self.inputShape,kernel_initializer='he_uniform',kernel_regularizer=l2(2e-4)))
        network.add(MaxPooling2D())

        network.add(Conv2D(128, (7,7), activation='relu',input_shape=self.inputShape,kernel_initializer='he_uniform',kernel_regularizer=l2(2e-4)))
        network.add(MaxPooling2D())

        network.add(Conv2D(256, (5,5), activation='relu', kernel_initializer='he_uniform',kernel_regularizer=l2(2e-4)))
        network.add(MaxPooling2D())
        
        network.add(Conv2D(512, (3,3), activation='relu', kernel_initializer='he_uniform',kernel_regularizer=l2(2e-4)))
        network.add(GlobalAveragePooling2D())

        network.add(Dense(4096, activation='relu',kernel_regularizer=l2(1e-3),kernel_initializer='he_uniform'))
        
        # Layer to convert all images to the fixed embedding size, to compute distance
        network.add(Dense(embeddingsize, activation=None,kernel_regularizer=l2(1e-3),kernel_initializer='he_uniform'))
        
        # Force the encoding to live on the d-dimentional hypershpere
        network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
        
        return network

    def build_model(self, network, margin=0.2):
        # Building the model
        
        # Define the tensors for the three input images in the triplet loss layer
        anchorInput = Input(self.inputShape, name="anchorInput")
        positiveInput = Input(self.inputShape, name="positiveInput")
        negativeInput = Input(self.inputShape, name="negativeInput") 
        
        # Generate the encodings (feature vectors) for the three images
        encodedA = network(anchorInput)
        encodedP = network(positiveInput)
        encodedN = network(negativeInput)
        
        #TripletLoss Layer, three feature vectores (images) as input
        # margin, float value to ensure distance wont be never 0
        lossLayer = TripletLossLayer(alpha=margin,name='tripletLossLayer')([encodedA,encodedP,encodedN])
        
        # Connect the inputs with the outputs, creating model
        model = Model(inputs=[anchorInput,positiveInput,negativeInput],outputs=lossLayer)
        
        # Defining learning rate
        optimizer = Adam(lr = 0.00006)
        
        #compiling model, loss none cause last layer is already the loss
        model.compile(loss=None,optimizer=optimizer)
        
        return model

    def loadWeights(self,network, path):
        network.load_weights(path)