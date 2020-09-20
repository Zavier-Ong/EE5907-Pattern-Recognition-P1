import numpy as np
from scipy import io

mat = io.loadmat('spamData.mat')# dictionary (key -> load matrix)
x_train=mat['Xtrain'] #(3065, 57)
x_test=mat['Xtest']   #(1536, 57)

#flatten y training and test data into 1D array for easier processing
y_train=mat['ytrain'].flatten() #(3065,)
y_test=mat['ytest'].flatten()   #(1536,)

def log_transform(dataset):
    return np.log(dataset+0.1)

log_x_train = log_transform(x_train)
log_x_test = log_transform(x_test)

#print("before:\n" + str(x_train[0:2]))
#print("after:\n" + str(log_x_train[0:2]))

MLE = np.mean(y_train)

spam_index_arr = []
not_spam_index_arr = []
for i in range(len(y_train)):
    if y_train[i] == 1:
        spam_index_arr.append(i)
    else:
        not_spam_index_arr.append(i)

#parameters required to calculate univariate gaussian distribution
x_train_mean_spam = np.mean(log_x_train[spam_index_arr], axis=0)
x_train_var_spam = np.var(log_x_train[spam_index_arr], axis=0)
x_train_mean_not_spam = np.mean(log_x_train[not_spam_index_arr], axis=0)
x_train_var_not_spam = np.var(log_x_train[not_spam_index_arr], axis=0)

#calculate feature log likelihood
def calc_univariate_gaussian(mean, var, x):
    result = -((x-mean)**2)/(2*var)
    result -= np.log(np.sqrt(2*np.pi*var))
    return result

def calc_posterior_predictive_distribution(dataset, spam):
    if spam:
        ML = np.log(MLE)
        mean_arr = x_train_mean_spam
        var_arr = x_train_var_spam
    else:
        ML = np.log(1-MLE)
        mean_arr = x_train_mean_not_spam
        var_arr = x_train_var_not_spam

    feature_ppd = np.zeros(shape = [len(dataset),])
    pp = 0
    for i in range(len(dataset)):
        pp = ML
        for j in range(len(dataset[i])):
            pp += calc_univariate_gaussian(mean_arr[j], var_arr[j], dataset[i][j])
        feature_ppd[i] = pp

    return np.array(feature_ppd)

def predict(spam, not_spam):
    result = np.zeros(shape = [len(spam),])
    for i in range(len(spam)):
        if spam[i] > not_spam[i]:
            result[i] = 1
        else:
            result[i] = 0
    return np.array(result)

def calc_error(result, ans):
    error_count = 0.0
    for i in range(len(result)):
        if result[i] != ans[i]:
            error_count +=1
    return error_count/len(ans)


ppd_spam_train = calc_posterior_predictive_distribution(log_x_train, 1)
ppd_not_spam_train = calc_posterior_predictive_distribution(log_x_train, 0)
train_result_arr = predict(ppd_spam_train, ppd_not_spam_train)

ppd_spam_test = calc_posterior_predictive_distribution(log_x_test, 1)
ppd_not_spam_test = calc_posterior_predictive_distribution(log_x_test, 0)
test_result_arr = predict(ppd_spam_test, ppd_not_spam_test)

print('training error: ' + str(calc_error(train_result_arr, y_train)))
print('test error    : ' + str(calc_error(test_result_arr, y_test)))
