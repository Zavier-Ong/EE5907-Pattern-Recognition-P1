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

def sigm(x):
    return 1/(1+np.exp(-x))

def l2_regularization(dataset, w, reg_param_val):
    mu = sigm(np.dot(dataset, w))

    g = np.dot(dataset.T, (mu-y_train))
    S = np.diag(mu*(1-mu))
    H = np.dot(np.dot(dataset.T, S), dataset)

    w[0] = 0 #exclude bias term
    g_reg = g + reg_param_val*w
    I = np.identity(len(g)+1)
    I[0][0] = 0 #exclude bias term
    H_reg = H + reg_param_val*I

    return g_reg, H_reg

def newton_method(dataset, reg_param_val):
    feature_count = dataset.shape[1]
    w = np.zeros(feature_count+1)
    bias_dataset = np.insert(dataset, 0, 1, axis=1)

    while True:
        print('here')
        g_reg, H_reg = l2_regularization(bias_dataset, w, reg_param_val)
        n = 1

        d = np.linalg.solve(H_reg, -g_reg)
        next_w = w + n*d

        if next_w == w - np.dot(np.linalg.inv(H_reg), g_reg):
            return next_w
        w = next_w

def calc_error(result, ans):
    error_count = 0.0
    for i in range(len(result)):
        if result[i] != ans[i]:
            error_count +=1
    return error_count/len(ans)

w = newton_method(log_x_train, 1)
probability_arr = sigm(w[0]+np.dot(log_x_train, w[1:]))
result = np.zeros(shape= len(probability_arr))
for i in range(len(probability_arr)):
    if probability_arr[i] > 0.5:
        result[i] = 1
    else:
        result[i] = 0
print(calc_error(result, y_train))