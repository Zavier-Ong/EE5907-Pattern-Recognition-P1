import numpy as np
from scipy import io

mat = io.loadmat('spamData.mat')
x_train=mat['Xtrain'] #(3065, 57)
x_test=mat['Xtest']   #(1536, 57)

#flatten y training and test data into 1D array for easier processing
y_train=mat['ytrain'].flatten() #(3065,)
y_test=mat['ytest'].flatten()   #(1536,)

def log_transform(dataset):
    return np.log(dataset+0.1)

log_x_train = log_transform(x_train)
log_x_test = log_transform(x_test)

