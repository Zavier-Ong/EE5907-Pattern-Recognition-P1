import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

mat = io.loadmat('spamData.mat')
x_train=mat['Xtrain'] #(3065, 57)
x_test=mat['Xtest']   #(1536, 57)

#flatten y training and test data into 1D array for easier processing
y_train=mat['ytrain'].flatten() #(3065,)
y_test=mat['ytest'].flatten()   #(1536,)

error_threshold = 10**-8

def log_transform(dataset):
    return np.log(dataset+0.1)

log_x_train = log_transform(x_train)
log_x_test = log_transform(x_test)

def sigm(x):
    return 1/(1+np.exp(-x))

#l2 regulatization dataset and w already includes the bias term
def l2_regularization(dataset, w, reg_param_val):
    mu = sigm(np.dot(dataset, w))

    g = dataset.T.dot((mu-y_train))
    S = np.diag(mu*(1-mu))
    H = dataset.T.dot(S).dot(dataset)

    #w[0] = 0 #exclude bias term
    w_reg = np.copy(w)
    w_reg[0] = 0
    g_reg = g + reg_param_val*w_reg
    I = np.identity(len(g))
    I[0][0] = 0 #exclude bias term
    H_reg = H + reg_param_val*I

    return g_reg, H_reg

def newton_method(dataset, reg_param_val):
    feature_count = dataset.shape[1]
    w = np.zeros(feature_count+1)
    bias_dataset = np.insert(dataset, 0, 1, axis=1)
    error = 1.0

    #iterate until convergence
    while error > error_threshold:
        g_reg, H_reg = l2_regularization(bias_dataset, w, reg_param_val)

        #since there is no need for line search in this assignment, there is no need to solve for d
        next_w = w - np.linalg.inv(H_reg).dot(g_reg)

        #euclidean norm (l1 norm)
        abs_diff_total = 0
        num_w = len(w)
        for i in range(len(w)):
            abs_diff_total += abs(next_w[i] - w[i])
        error = abs_diff_total / num_w

        w = next_w
    return w

def calc_error(result, ans):
    error_count = 0.0
    for i in range(len(result)):
        if result[i] != ans[i]:
            error_count +=1
    return error_count/len(ans)

def classify(probs):
    result = np.zeros(shape=[len(probs),])
    for i in range(len(probs)):
        if probs[i] > 0.5:
            result[i] = 1
        else:
            result[i] = 0
    return result

lambda_arr = np.concatenate((np.arange(1,11), np.arange(15,105,5)), axis=None)
train_error_arr = np.zeros(shape=[len(lambda_arr), 1])
test_error_arr = np.zeros(shape=[len(lambda_arr), 1])

for i in range(len(lambda_arr)):
    #training
    w = newton_method(log_x_train, lambda_arr[i])
    train_probability_arr = sigm(w[0]+log_x_train.dot(w[1:]))
    train_result = classify(train_probability_arr)
    train_error_arr[i] = calc_error(train_result, y_train)
    #test
    test_probability_arr = sigm(w[0]+log_x_test.dot(w[1:]))
    test_result = classify(test_probability_arr)
    test_error_arr[i] = calc_error(test_result, y_test)

    if lambda_arr[i] == 1 or lambda_arr[i] == 10 or lambda_arr[i] == 100:
        print('train lambda (' + str(lambda_arr[i]) + ') :' + str(train_error_arr[i]))
        print('test lambda  (' + str(lambda_arr[i]) + ') :' + str(test_error_arr[i]))

plt.figure(1)
plt.plot(lambda_arr, np.array(train_error_arr)*100, label='Train dataset')
plt.plot(lambda_arr, np.array(test_error_arr)*100, label='Test dataset')
plt.xlabel('Lambda', fontsize = 16)
plt.ylabel('Error rate (%)', fontsize = 16)
plt.legend(loc=0)
plt.title('Logistic Regression(Error Rate vs Lambda)', fontsize = 16)
plt.show()