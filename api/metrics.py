class Metrics:
    def computeDist(self,a,b):
    return np.sum(np.square(a-b))
    
    # Compute probabilities of similiraty for all images crossed with all images in X
    def computeProbs(self,network,X,Y):
        numberOfImages = len(X)
        
        # Get number of comparisons and intiliaze empty vectos for probability and real target
        numberEvaluation = int(numberOfImages*(numberOfImages-1)/2)
        probs = np.zeros((numberEvaluation))
        y = np.zeros((numberEvaluation))
        
        # For all images received in X, get their embeddings ( Feature vector, representation of image)
        embeddings = network.predict(X)
        
        # get distance for each image from all other images
        k = 0
        for i in range(numberOfImages):
                #Against all other images
                for j in range(i+1,numberOfImages):
                    #compute the probability of being the right decision :  1 for right class, 0 for all other classes
                    probs[k] = computeDist(embeddings[i,:],embeddings[j,:])
                    
                    # compare to see the real result from Y_test 
                    if (Y[i]==Y[j]):
                        y[k] = 0
                    else:
                        y[k] = 1
                    k += 1
                    
        return probs,y

    # Based on probs generate from computeProbs and y_test array, create score with percentual of cases correct
    def computeAccuracy(self,probs,y,thresH):
       correctPrediction = 0
        correctEqualPrediction = 0
        totalPredictions = len(probs)
        probThreshholded = []
        
        # Threshold value to decide wether images are equal or different
        threshhold = thresH
        
        # Creating vector with predicitions 0 or 1, after passing through threshold
        # Check if np.clip does the same
        for prob in probs:
            if(prob > threshhold):
                probThreshholded.append(1)
            else:
                probThreshholded.append(0)
                
        for index,prob in enumerate(probThreshholded):
            # If prediction equal to true target, add 1 to numer of corrects predictions
            if prob == y[index]: 
                correctPrediction += 1
                if(y[index] == 0):
                correctEqualPrediction +=1

        totalAcc = correctPrediction/totalPredictions # Percentage of correct cases
        unique, counts = np.unique(y, return_counts=True)
        equalClassAcc = correctEqualPrediction/counts[0] # Percentage of correct cases with equal class
        return totalAcc,equalClassAcc

    def find_nearest(self,array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1],idx-1
        else:
            return array[idx],idx

    def compute_metrics(self,probs,yprobs):
        '''
        Returns
            fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
            tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
            thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
            auc : Area Under the ROC Curve metric
        '''
        # calculate AUC
        auc = roc_auc_score(yprobs, probs)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(yprobs, probs)
        
        return fpr, tpr, thresholds,auc

    def draw_roc(self,fpr, tpr,thresholds):
        #find threshold
        targetfpr=1e-3
        _, idx = find_nearest(fpr,targetfpr)
        threshold = thresholds[idx]
        recall = tpr[idx]
        
        
        # plot no skill
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        plt.title('AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(auc,targetfpr,recall,abs(threshold) ))
        # show the plot
        plt.show()