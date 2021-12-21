#!/usr/bin/env python
# coding: utf-8

# # SVM

# In[1]:


class svm:
    
    def __init__(self, learning_rate = 0.00001, lambda_parameter = 0.01, epochs = 5000):
        
        self.learning_rate = learning_rate
        self.lambda_parmeter = lambda_parameter
        self.epochs = epochs
        self.weights = None
        self.reg_strength = 10000
        
    
    def cost_computation(self, W, X, Y):
        
        # Hinge loss calculation
        n = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0 # make all distances less than 0 equal to 0.
        hinge_loss = self.reg_strength*(np.nansum(distances)/n)
        
        # Calculate cost
        cost = 1/2 * np.dot(W,W) + hinge_loss
        return cost
    
    
    def cost_gradient_computation(self, W, X_ij, Y_ij):
        
        if type(Y_ij) == np.float64:
            Y_ij = np.array([Y_ij])
            X_ij = np.array([X_ij])  # gives multidimensional array
            
        distance = 1 - (Y_ij * np.dot(X_ij, W))
        dw = np.zeros(len(W))
        
        for i, d in enumerate(distance):
            if max(0,d) == 0:
                di = W
            else:
                di = W - (self.reg_strength * Y_ij[i]* X_ij[i])
            dw += di
            
        dw = dw/len(Y_ij) # average calculation
        return dw
    
    
    def sgd_fit(self, X, Y):
        
        # Stochastic gradient descent
        max_epochs = 5000
        self.weights = np.zeros(X.shape[1])
        nth = 0
        prev_cost = 9999999
        cost_threshold = 0.001  # in percent
        
        # stochastic gradient descent
        for epoch in range(1,max_epochs):
            
            # shuffle
            X_i, Y_i = shuffle(X, Y)
            
            for i, x in enumerate(X_i):

                ascent = self.cost_gradient_computation(self.weights, x, Y_i[i])
                self.weights = self.weights - (self.learning_rate * ascent)
            
            if epoch == 2 ** nth:
                cost = self.cost_computation(self.weights, X, Y)
                print("Epoch is: {} and Cost is: {} \n {}".format(epoch, cost, self.weights))
                nth += 1
                
        return self.weights
    
    
    def predict(self, X, Y):
        
        Y_te_predictions = np.array([])
        
        for i in range(X.shape[0]):
            Yp = np.sign(np.dot(self.weights, X.to_numpy()[i]))
            Y_te_predictions = np.append(Y_te_predictions, Yp)
            
        print("\naccuracy: {}".format(accuracy_score(Y, Y_te_predictions)))
        print("recall: {}".format(recall_score(Y, Y_te_predictions)))
        print("precision: {}".format(precision_score(Y, Y_te_predictions)))
            
        return Y_te_predictions
    

# Testing
if __name__ == "__main__":
    
    # Imports
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pprint as pp
    
    print("reading dataset...\n")
    
    df = pd.read_csv('Iris.csv',index_col='Id')
    
    #Removing third category of Iris two simplify process
    df = df[df.Species != 'Iris-virginica']
    
    df = df.replace(['Iris-setosa','Iris-versicolor'],[-1,1]).astype(np.float64)
    df = shuffle(df)
    Y = df.Species
    X = df.drop(['Species'],axis=1)
    X.insert(loc=len(X.columns), column = 'intercept', value =1)
    XN = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(XN)
    X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,
                                           test_size=0.4,
                                           random_state=42)
    clf = svm()
    clf.sgd_fit(X_tr.to_numpy(), Y_tr.to_numpy())
    predictions = clf.predict(X_te,Y_te)
    print('\nweights')
    print(clf.weights)
    print('\nside by side comparison of outputs vs real solutions:')
    pp.pprint(list(zip(predictions,Y_te.to_list())))

