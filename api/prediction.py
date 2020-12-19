from network_humpback import NetworkHumpback 
from network_species import NetworkSpecies
import numpy as np
class Prediction:
    def __init__(self):
        # Individual network
        networkHumpback = NetworkHumpback()
        self.network = networkHumpback.build_network(128)
        networkHumpback.loadWeights(self.network,'assets/HBWhale_Network_Weights_HardBatch_bestone.h5')
        self.embeddings = np.load('assets/embeddings.npy')
        self.classes = np.load('assets/classes.npy')
            
        # Species network
        networkSpecies = NetworkSpecies()
        networkSpecies.build_model()
        self.modelSpecies = networkSpecies.load_weights()

    def computeDist(self,a,b):
        return np.sum(np.square(a-b))

    def makePredictionSpecie(self,image):
        targets = [image]
        targets = np.array(targets)
        predicition = self.modelSpecies.predict(targets)
        if(predicition < 0.5):
            return "Humpback whale"
        else:
            return "Right whale"

    def makePredictionIndividual(self,target, individuals = None):
        minDistance = None
        indexPrediction = None

        targets = []
        targets.append(target)
        targets = np.array(targets)
        print(targets.shape)
        target = self.network.predict(targets)

        for i in range(len(self.embeddings)):
            # Check if individuals is not empty, so it was requested to compare with specifics
            if(individuals):
                if(self.classes[i] in individuals):
                    distance  = self.computeDist(self.embeddings[i,:],target)
            else:
                distance  = self.computeDist(self.embeddings[i,:],target)

            #initilize variable with first value    
            if i == 0:
                minDistance = distance
            
            # Check if distance is lower then threshold and lower then min distance
            if (distance < 0.8 and distance < minDistance):
                minDistance = distance
                indexPrediction = i

        return minDistance, self.classes[indexPrediction]

    def compareImages(self, image1,image2):
        # Get embedding of images
        images = [image1,image2]
        images = np.array(images)
        embeddings = self.network.predict(images)
        # Compute distance
        distance = self.computeDist(embeddings[0],embeddings[1])

        # Check if distance lower then the threshold
        if(distance < 0.86):
            return True
        return False