
# -*- coding: utf-8 -*- 

""" 
Author:  Magnier Morgane
"""


import numpy as np
import matplotlib.pyplot as plt

class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, olynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

    def calcul_noyau(self, x, x_):

        """"
        Fonction qui calcul le noyau selon le type de noyau. 
        Pour être réutilisé directement dans self.entrainement et self.prediction. 
        """
        
        if self.noyau == "lineaire": 
            return np.dot(x,x_)
        
        if self.noyau == "RBF" : 
            return np.exp(- np.dot((x - x_),(x - x_)) / (2*self.sigma_square**2)) 

        if self.noyau == "polynomial": 
            return (np.dot(x,x_) + self.c)**self.M
        
        if self.noyau == "sigmoidal": 
            return np.tanh(self.b * np.dot(x,x_) + self.d) 
    
    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """

        self.x_train = x_train #Mémoire des données d'entrainement
        N = np.shape(x_train)[0] 
        K = np.zeros((N,N)) #Initialisation de la matrice de Graam
        
        for i in range (0,N-1):
            for j in range (0,N-1):
                K[i,j] = self.calcul_noyau(x_train[i,:], x_train[j,:]) 
        
        self.a = np.linalg.solve(K + self.lamb * np.identity(N), t_train) #équation 6.8 de Bishop
            
    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """

        N = np.shape(self.x_train)[0] 
        k = np.zeros(N)

        for i in range (0,N-1): 
            k[i] = self.calcul_noyau(self.x_train[i,:], x) #calcul de k[x] dans l'équation 6.9

        y = np.dot(k.T, self.a) #équation 6.9
        
        if y > 0.5:
            return 1
        else: 
            return 0

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (prediction - t)**2 

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """

        ##Définition des valeurs de paramètres à tester
        sigma_square_values = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2]
        c_values = [0, 1, 2, 3, 4, 5]
        b_values = [0.00001, 0.0001, 0.001, 0.01]
        d_values = [0.00001, 0.0001, 0.001, 0.01]
        M_values = [2, 3, 4, 5, 6]
        lamb_values = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2]

        best_moy_E = float('inf')  # On initiale l'erreur moyenne à une valeur élevée pour etre sur qu'elle soit remplacée à la 1ère itération
        opt_hyperparameters = None #hyperparamètre optimaux

        k = 10
        x_tab_ = np.array_split(x_tab, k) #split en k partie
        t_tab_ = np.array_split(t_tab, k) #split en k partie

        #On fait la recherche d'hyperparamètre selon un grid search (ie. On cherche la meilleure combinaison de paramètres)
        for sigma_square in sigma_square_values:
            for c in c_values:
                for b in b_values:
                    for d in d_values:
                        for M in M_values:
                            for lamb in lamb_values:
                                
                                hyperparameters = (sigma_square, c, b, d, M, lamb) #liste de la combinaison d'hyperparamètres
                                moy_E = 0 #initialisation de la moyenne
                                
                                #k-fold cross validation
                                for i in range(k):
                                    x_val = x_tab_[i]
                                    x_train = np.concatenate([indices for j, indices in enumerate(x_tab_) if j != i])
                                    
                                    t_val = t_tab_[i]
                                    t_train = np.concatenate([indices for j, indices in enumerate(t_tab_) if j != i])
                                    
                                    #mise à jour des attributs de self pour la classification
                                    self.sigma_square, self.c, self.b, self.d, self.M, self.lamb = hyperparameters
                                    
                                    #entrainement sur x_train
                                    self.entrainement(x_train, t_train)

                                    #prediction et calcul de l'erreur sur x_val
                                    for i in range(len(x_val)): 
                                        prediction = self.prediction(x_val[i])
                                        moy_E += self.erreur(t_val[i], prediction)

                                moy_E /= k
                                
                                #recherche de la meilleure combinaison d'hyperparamètre (ie. celle qui minimise l'erreur)
                                if moy_E < best_moy_E:
                                    best_moy_E = moy_E
                                    opt_hyperparameters = hyperparameters

        self.sigma_square, self.c, self.b, self.d, self.M, self.lamb = opt_hyperparameters
        self.entrainement(x_tab, t_tab) #entrainement une dernière fois 
        
    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()
