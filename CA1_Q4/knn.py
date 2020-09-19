import numpy as np
import scipy.io as io
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt

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

def get_Euclidean_distance(dataset, points):
    result = []
    for point in points:
        point_result = []
        for data in dataset:
            point_result.append(distance.euclidean(data, point))
        result.append(point_result)
    return np.array(result)

train_distances = get_Euclidean_distance(log_x_train, log_x_train)
test_distances = get_Euclidean_distance(log_x_train, log_x_test)

def classifyPoint(distances, k):
    k_indexes = np.argsort(distances)[0:k]
    spam_count = np.sum(y_train[k_indexes]==1)
    not_spam_count = np.sum(y_train[k_indexes]==0)
    #in this case where spam and not spam are equal, they would be classified as not spam
    if spam_count > not_spam_count:
    #in this case where spam and not spam are equal, they would be classified as spam
    #if spam_count >= not_spam_count:
        return 1
    else:
        return 0

def classify(distances_arr, k):
    result = np.zeros(shape=[len(distances_arr),])
    for i in range(len(distances_arr)):
        result[i] = (classifyPoint(distances_arr[i], k))
    return np.array(result)

def calc_error(result, ans):
    error_count = 0.0
    for i in range(len(result)):
        if result[i] != ans[i]:
            error_count +=1
    return error_count/len(ans)

k_arr = np.concatenate((np.arange(1,11), np.arange(15,105,5)), axis=None)
train_error_arr = np.zeros(shape=[len(k_arr), 1])
test_error_arr = np.zeros(shape=[len(k_arr), 1])
for i in range(len(k_arr)):
    result = classify(train_distances, k_arr[i])
    train_error_arr[i] = calc_error(result, y_train)

    result = classify(test_distances, k_arr[i])
    test_error_arr[i] = calc_error(result, y_test)
    if k_arr[i] == 1 or k_arr[i] == 10 or k_arr[i] == 100:
        print('train k (' + str(k_arr[i]) + ') :' + str(train_error_arr[i]))
        print('test k  (' + str(k_arr[i]) + ') :' + str(test_error_arr[i]))

plt.figure(1)
plt.plot(k_arr, np.array(train_error_arr)*100, label='Train dataset')
plt.plot(k_arr, np.array(test_error_arr)*100, label='Test dataset')
plt.xlabel('K', fontsize = 16)
plt.ylabel('Error rate (%)', fontsize = 16)
plt.legend(loc=0)
plt.title('KNN(Error rate(%) against K)', fontsize = 16)
plt.show()
