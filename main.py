import numpy as np
import random
import collections as co


def unpickle(file,test=1):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict
	
def lecture_cifar(taille = -1,liste_path=["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]):
	"""
	A function which loads the data
	
	Parameters : 
		taille : (int) the number of data loaded
		liste_path : (list of str) the files to open
	Returns :
		X,Y which X contains the data in an array and Y the labels in an array
	"""
	X = np.array([])
	Y = np.array([])
	for path in liste_path: 
		dict = unpickle(path)
		for data in dict[b'data']:
			if taille == -1 or taille > X.shape[0]:
				try:
					X = np.vstack([X,data])
				except:
					X = np.array([data]) 
		
		for labels in dict[b'labels']:
			if taille == -1 or taille > Y.shape[0]:
				try:
					Y = np.hstack([Y,labels])
				except:
					Y = np.array([labels]) 
			
		print(dict.keys())
	return X,Y
	
def decoupage_donnees(X, Y):
	"""
	Split the data into two arrays : one for the test and the other for the learning.
	
	Parameters : 
		X : (array) A matrix containing the data
		Y : (array) A vector containing the labels
	Returns :
		Xapp, Yapp, Xtest, Ytest
	"""
	l = X.shape[0]
	chosen_ones = random.sample(range(l),int(0.8*l))
	l = 500
	chosen_ones = np.array([i for i in range(200)])
	chosen_ones = np.sort(chosen_ones)
	print(len(chosen_ones))
	compteur = 0
	indice = 0
	while compteur<l:
		
		if len(chosen_ones)>indice and chosen_ones[indice] == compteur:
			
			try:
				Xtest = np.vstack([Xtest,X[compteur]])
			except:
				Xtest = np.array([X[compteur]])
			try:
				Ytest = np.hstack([Ytest,Y[compteur]])
			except:
				Ytest = np.array([Y[compteur]])
			indice += 1
		
		else:
			try:
				Xapp = np.vstack([Xapp,X[compteur]])
			except:
				Xapp = np.array([X[compteur]]) 
			try:
				Yapp = np.hstack([Yapp,Y[compteur]])
			except:
				Yapp = np.array([Y[compteur]])
		compteur +=1
				
		
	return Xapp,Yapp, Xtest, Ytest
	
def kppv_distances(Xapp,Xtest):
	"""
	Compute the k nearest neighbours 
	
	Parameters :
		Xapp : (array) the vectors used for the learning
		Xtest : (array) the vectors used for the test
	Return :
		An array containing all the euclidean distances (squared)
	"""
	Xapp_carres = np.diag(np.dot(Xapp,np.transpose(Xapp)))
	Xtest_carres = np.diag(np.dot(Xtest,np.transpose(Xtest)))
	
	###############################################################
	### Test unitaire (avec des matrices connues que l'on peut ####
	### calculer à la main  #######################################
	###############################################################
	#Xapp = np.array([[2,3],[1,1],[2,4]])
	#Xtest = np.array([[2,2],[2,3]])
	#Xapp_carres = np.diag(np.dot(Xapp,np.transpose(Xapp)))
	#Xtest_carres = np.diag(np.dot(Xtest,np.transpose(Xtest)))
	###############################################################
	
	AB = -np.transpose(2*np.dot((Xtest),np.transpose(Xapp)))
	X1 = np.transpose(np.tile(Xapp_carres,(Xtest.shape[0],1)))
	X2 = np.tile(Xtest_carres,(Xapp.shape[0],1))
	distance = AB+X1+X2
	return distance

def kkpv_predict(distance, Yapp, k):
	"""
	Compute the class of Yapp knowing the distances between Xtest and the vectors of the learning.
	
	Parameters:
		distance : (array) all the distances between Xtest and Xapp
		Yapp : (array) all the indexes of Xapp
		k : (int) numbers of neighbours used
	Return:
		Ypred : (array) The vectors containing the indexes given by the learning. 
	"""
	Ypred = np.zeros(distance.shape[1])
	indexes = np.argpartition(distance,range(k),axis=0)[:k]
	k_voisin = np.transpose(Yapp[indexes])
	for i in range(len(k_voisin)):
		Ypred[i] = int(co.Counter(k_voisin[i][:]).most_common(1)[0][0])
	return Ypred
	
	
if __name__ == "__main__":
	X,Y = lecture_cifar(800)
	Xapp,Yapp, Xtest, Ytest = decoupage_donnees(X,Y)
	Xtest = Xapp
	Ytest = Yapp
	distance = kppv_distances(Xapp,Xtest)
	Ypred = kkpv_predict(distance, Yapp, 1)
	
	
	
