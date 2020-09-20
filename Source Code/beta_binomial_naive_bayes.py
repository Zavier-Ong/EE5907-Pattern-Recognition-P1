import numpy as np
import matplotlib.pyplot as plt
from scipy import io

mat = io.loadmat('spamData.mat')# dictionary (key -> load matrix)
x_train=mat['Xtrain'] #(3065, 57)
x_test=mat['Xtest']   #(1536, 57)

#flatten y training and test data into 1D array for easier processing
y_train=mat['ytrain'].flatten() #(3065,)
y_test=mat['ytest'].flatten()   #(1536,)

#print(y_train.shape)
#print(y_test.shape)

def binarize(dataset):
    return 1 * (dataset > 0)

binarized_x_train = binarize(x_train)
binarized_x_test = binarize(x_test)

# print("before")
# print(x_train[0:2])
# print("after")
# print(binarized_x_train[0:2])

#Posterior predictive distribution for class prior (MLE)
#since all data is represented as binary, we can calculate the MLE by simply calculating the mean
MLE = np.mean(y_train)
#print(MLE)
#Calculating N1 and N values
spam_index_arr = []
not_spam_index_arr = []
spam_total = 0
not_spam_total = 0
for i in range(len(y_train)):
    if y_train[i] == 1: #spam
        spam_index_arr.append(i)
        spam_total += 1
    else:
        not_spam_index_arr.append(i)
        not_spam_total += 1

spam_feature_sum = np.sum(binarized_x_train[spam_index_arr], axis=0)
not_spam_feature_sum = np.sum(binarized_x_train[not_spam_index_arr], axis=0)
#print(spam_feature_sum)

def calc_posterior_predictive(N1, N, alpha):
    return (N1 + alpha)/(N+2*alpha)

def calc_posterior_predictive_distribution(dataset, alpha, spam_class_label):
    if spam_class_label:
        ML = np.log(MLE)
        total = spam_total
        feature_sum = spam_feature_sum
    else:
        ML = np.log(1-MLE)
        total = not_spam_total
        feature_sum = not_spam_feature_sum
    feature_ppd = np.zeros(shape = [len(dataset),])
    # for i in range(len(dataset)):
    #     pd = ML
    #     for j in range(len(dataset[i])):
    #         if dataset[i][j] == 1:
    #             pd += np.log(calc_posterior_predictive(feature_sum[j], total, alpha))
    #         else:
    #             pd += np.log(1 - calc_posterior_predictive(feature_sum[j], total, alpha))
    #     feature_ppd.append(pd)
    # return np.array(feature_ppd)

    ## Optimization that utilizes matrix multiplication to speed up the code as it was too slow.
    if alpha ==0:
        for i in range(len(dataset)):
            pd = ML
            for j in range(len(dataset[i])):
                if dataset[i][j] == 1:
                    pd += np.log(calc_posterior_predictive(feature_sum[j], total, alpha))
                else:
                    pd += np.log(1-calc_posterior_predictive(feature_sum[j], total, alpha))
            feature_ppd[i] = pd
        return np.array(feature_ppd)
    else:
        pd = calc_posterior_predictive(feature_sum, total, alpha)
        feature_ppd = dataset.dot(np.log(pd)) + (1-dataset).dot(np.log(1-pd))
        return feature_ppd+ML

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


error_train = []
error_test = []
#setting alpha step sizes
alpha_arr = np.arange(0, 100.5, 0.5)
for alpha in alpha_arr:
    ppd_spam_train = calc_posterior_predictive_distribution(binarized_x_train, alpha, 1)
    ppd_not_spam_train = calc_posterior_predictive_distribution(binarized_x_train, alpha, 0)
    train_result_arr = predict(ppd_spam_train, ppd_not_spam_train)
    error_train.append(calc_error(train_result_arr, y_train))

    ppd_spam_test = calc_posterior_predictive_distribution(binarized_x_test, alpha, 1)
    ppd_not_spam_test = calc_posterior_predictive_distribution(binarized_x_test, alpha, 0)
    test_result_arr = predict(ppd_spam_test, ppd_not_spam_test)
    error_test.append(calc_error(test_result_arr, y_test))

alpha_val = [1, 10, 100]
for i in alpha_val:
    print('train alpha (' + str(i) + ') :'  + str(error_train[i]))
    print('test alpha  (' + str(i) + ') :' + str(error_test[i]))

plt.figure(1)
plt.plot(alpha_arr, np.array(error_train)*100, label="Train dataset")
plt.plot(alpha_arr, np.array(error_test)*100, label="Test dataset")
plt.xlabel('alpha', fontsize=16)
plt.ylabel('error rate (%)', fontsize=16)
plt.legend(loc=0)
plt.title('Beta Binomial Naive Bayes(Error rate vs Alpha)', fontsize=16)
plt.show()
