# -*- coding: utf-8 -*- 

""" 
Author:  Magnier Morgane 
"""

import numpy as np

class LinearClassifier(object):
    def __init__(self, x_train, y_train, x_val, y_val, num_classes, bias=False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.bias = bias  # when bias is True then the feature vectors have an additional 1

        num_features = x_train.shape[1]
        if bias:
            num_features += 1

        self.num_features = num_features
        self.num_classes = num_classes
        self.W = self.generate_init_weights(0.01)

    def generate_init_weights(self, init_scale):
        return np.random.randn(self.num_features, self.num_classes) * init_scale

    def train(self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, init_scale=0.01):
        """
        Train the model with a cross-entropy loss
        Naive implementation (with loop)

        Inputs:
        - num_epochs: the number of training epochs
        - lr: learning rate
        - l2_reg: the l2 regularization strength
        - lr_decay: learning rate decay.  Typically a value between 0 and 1
        - init_scale : scale at which the parameters self.W will be randomly initialized

        Returns a tuple for:
        - training accuracy for each epoch
        - training loss for each epoch
        - validation accuracy for each epoch
        - validation loss for each epoch
        """
        loss_train_curve = []
        loss_val_curve = []
        accu_train_curve = []
        accu_val_curve = []

        self.W = self.generate_init_weights(init_scale)  # type: np.ndarray

        sample_idx = 0
        num_iter = num_epochs * len(self.x_train)
        
        for i in range(num_iter):
            # Take a sample
            x_sample = self.x_train[sample_idx]
            y_sample = self.y_train[sample_idx]
            if self.bias:
                x_sample = augment(x_sample)

            # Compute loss and gradient of loss
            loss_train, dW = self.cross_entropy_loss(x_sample, y_sample, l2_reg)

            # Take gradient step
            self.W -= lr * dW

            # Advance in data
            sample_idx += 1
            if sample_idx >= len(self.x_train):  # End of epoch

                accu_train, loss_train = self.global_accuracy_and_cross_entropy_loss(self.x_train, self.y_train, l2_reg)
                accu_val, loss_val, = self.global_accuracy_and_cross_entropy_loss(self.x_val, self.y_val, l2_reg)

                loss_train_curve.append(loss_train)
                loss_val_curve.append(loss_val)
                accu_train_curve.append(accu_train)
                accu_val_curve.append(accu_val)

                sample_idx = 0
                lr *= lr_decay

        return loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve

    def predict(self, X):
        """
        return the class label with the highest class score i.e.

            argmax_c W.X

         X: A numpy array of shape (D,) containing one or many samples.

         Returns a class label for each sample (a number between 0 and num_classes-1)
        """
        class_label = np.zeros(X.shape[0])

        X = X.T #Comme X_test est au format (N,D) et pas (D,N)
        if self.bias and X.shape[0] < self.W.shape[1]:
            #Condition car le biai est ajouté dans plusieurs fonctions, pour etre sur qu'il n'ai pas été déjà ajouté. 
            X = augment(X)
            
        class_score = np.dot(self.W.T, X) #a
        class_label = np.argmax(class_score,axis = 0) 
        #Retourne un vecteur ou un scalaire contenant pour chaque sample l'index de la 
        #classe associé (0,1,2,3 selon le nombre de classes)

        return class_label

    def global_accuracy_and_cross_entropy_loss(self, X, y, reg=0.0):
        """
        Compute average accuracy and cross_entropy for a series of N data points.
        Naive implementation (with loop)
        Inputs:
        - X: A numpy array of shape (D, N) containing many samples.
        - y: A numpy array of shape (N) labels as an integer
        - reg: (float) regularization strength
        Returns a tuple of:
        - average accuracy as single float
        - average loss as single float
        """
        accu = 0
        loss = 0
   
        if self.bias and X.shape[0] < self.W.shape[1]: 
            #Condition car le biai est ajouté dans plusieurs fonctions, pour etre sur qu'il n'ai pas été déjà ajouté. 
            X = augment(X)
            
        N = X.shape[0]
        correct_samples = 0

        for i in range(N):
            x = X[i,:]  #selectionne la ligne i

            loss += self.cross_entropy_loss(x, y[i], reg)[0]
            
            predicted_class = self.predict(x)  
            
            if predicted_class == y[i]:
                correct_samples += 1

        #Calcul de l'accuracy
        accu = correct_samples / N #Ratio du nbe d'échantillons bien classés. 
        loss /= N #Moyenne des loss entropies croisées

        return accu, loss

    def cross_entropy_loss(self, x, y, reg=0.0):
        """
        Cross-entropy loss function for one sample pair (X,y) (with softmax)
        C.f. Eq.(4.104 to 4.109) of Bishop book.

        Input have dimension D, there are C classes.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - x: A numpy array of shape (D,) containing one sample.
        - y: training label as an integer
        - reg: (float) regularization strength
        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)

        if self.bias and x.shape[0] < self.W.shape[1]: 
            #Condition car le biai est ajouté dans plusieurs fonctions, pour etre sur qu'il n'ai pas été déjà ajouté. 
            x = augment(x)
        #1- Compute of softmax      
        scores = np.dot(self.W.T,x) #a
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)  
        #Pour obtenir un vecteur contenant les probabilités que x appartienne à chaque classe. 

        #2- Compute cross-entropy loss
        loss = -np.log(probs[y]) 
        #Le one hot vector t qui contient des 0 partout sauf à l'indice y 
        #correspondant à l'indice de classe. on a donc 
        #loss = -sum_k(t_k * np.log(probs[k])= -np.log(probs[y])

        #3- Compute gradient
        dif = probs 
        dif[y] -= 1 #vecteur probs - t où t est le one hot vector, d'où dif = probs sauf à dif[y] où t[y] = 1 
        dW = np.outer(x.T,dif) #dW = dLpred/da * da/dw = (probs-t) * x.T (sans le terme de régularisation)

        #4- régularisation
        loss += 0.5 * reg * np.sum(self.W * self.W)
        dW += 2 * reg * self.W

        return loss, dW


def augment(x):
    if len(x.shape) == 1:
        return np.concatenate([x, [1.0]])
    else:
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)
