import numpy as np
from scipy import io
import matplotlib.pyplot as plot

mat = io.loadmat('spamData.mat')# dictionary (key -> load matrix)
x_train=mat['Xtrain'] #(3065, 57)
x_test=mat['Xtest']   #(1536, 57)

y_train=mat['ytrain'] #(3065, 1)
y_test=mat['ytest']   #(1536, 1)

np.set_printoptions(precision=3, suppress=True)
#print(Xtrain.shape)
#print(ytrain.shape)
#print(Xtest.shape)
#print(ytest.shape)

#print("before: \n" + str(x_train[0:2]))
# if x > 0, set to 1, otherwise set to 0
def binarize(data):
    return np.array([x>0 for x in data]).astype('uint8')

# data processing (binarization)
binarized_x_train = binarize(x_train)
binarized_x_test = binarize(x_test)

#print("after: \n" + str(binarized_x_train[0:2]))

#Posterior predictive distribution for class prior (MLE)
Nc = 0
for mail in y_train:
    if mail == 1:
        Nc += 1
MLE = Nc/len(y_train)
print("MLE: " + str(MLE))

#getting index of spam in training set
spam_index_arr = []
not_spam_index_arr = []
#total spam in y_train
N_spam = 0
#total not spam in y_train
N_not_spam = 0
for i in range(len(y_train)):
    if y_train[i] == 1:
        spam_index_arr.append(i)
        N_spam += 1
    else:
        not_spam_index_arr.append(i)
        N_not_spam += 1

#value of N1(spam) of all 57 features
N_1 = np.sum(binarized_x_train[spam_index_arr], axis=0)
#value of N0(not spam) of all 57 features
N_0 = np.sum(binarized_x_train[not_spam_index_arr], axis=0)
print(N_1)
print(N_0)
print(N_spam)
print(N_not_spam)

#setting hyperparameter alpha
alpha_arr = np.arange(0,100.5, 0.5)

def calc_posterior(N1, N, alpha):
    return (N1+alpha)/(N+2*alpha)

def calc_posterior_predictive_distribution(dataset, feature_sum, N, spam, alpha):
    if spam:
        ML = np.log(MLE)
    else:
        ML = np.log(1-MLE)
    ppd = []
    for i in range(len(dataset)):
        posterior_predictive = ML
        for j in range(len(dataset[i])):
            if dataset[i][j] == 1:
                posterior_predictive += np.log(calc_posterior(feature_sum[j], N, alpha))
            else:
                posterior_predictive += np.log(calc_posterior(feature_sum[j], N, alpha))
        ppd.append(posterior_predictive)
    return np.array(ppd)

#returns array of prediction based on higher value probabilty when comparing between
#spam and not_spam arrays
def predict(spam, not_spam):
    result = []
    for i in range(len(spam)):
        if spam[i] > not_spam[i]:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

def calc_error(result, ans):
    error = 0
    for i in range(len(result)):
        if result[i] != ans[i]:
            error += 1
        return error/len(ans)

error_train = []
error_test = []
#for i in range(len(alpha_arr)):
    #calculation for error in training
posterior_predictive_spam_arr = calc_posterior_predictive_distribution(binarized_x_train, N_1, N_spam, 1, 1)
posterior_predictive_not_spam_arr = calc_posterior_predictive_distribution(binarized_x_train, N_0, N_not_spam, 0, 1)
result = predict(posterior_predictive_spam_arr, posterior_predictive_not_spam_arr)
error_train.append(calc_error(result, y_train))
    #calculation for error in test
posterior_predictive_spam_arr = calc_posterior_predictive_distribution(binarized_x_test, N_1, N_spam, 1, 1)
posterior_predictive_not_spam_arr = calc_posterior_predictive_distribution(binarized_x_test, N_0, N_not_spam, 0, 1)
result = predict(posterior_predictive_spam_arr, posterior_predictive_not_spam_arr)
error_test.append(calc_error(result, y_test))

print(error_train)
print(error_test)

# plot.figure(1)
# plot.plot(alpha_arr, np.array(error_train), label="Train dataset")
# plot.plot(alpha_arr, np.array(error_test), label="Test dataset")
# plot.legend(loc=0)
# plot.title('Error rate (Beta-bernoulli Naive Bayes)')
# plot.show()

print("done")


