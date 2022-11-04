import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
#import tensorflow as tf
#from predict import predict

X_cats = np.ndarray((10,30000))
X_dogs = np.ndarray((10,30000))

def relu(x):
    x[x < 0] = 0
    return x
def drelu(x):
    return np.array(x>0,dtype = np.float32)

def predict(X,W_1, W_2, W_3, b1, b2, b3):
    z_1 = np.matmul(W_1, X.T) + b1
    z_1[z_1 < 0] = 0
    a_1 = z_1
    # a_2 = np.append(np.ones((1, a_2.shape[1])), a_2, axis=0)
    z_2 = np.matmul(W_2, a_1) + b2
    z_2[z_2 < 0] = 0
    a_2 = z_2
    # a_3 = np.append(np.ones((1, a_3.shape[1])), a_3, axis=0)
    z_3 = np.matmul(W_3, a_2) + b3
    h = 1 / (1 + np.exp(-z_3))
    pred = np.ndarray(h.shape[1])
    for i in range(h.shape[1]):
        if h[0,i] >= 0.4533 :
            pred[i] = 1
        else:
            pred[i] = 0

    return pred


for i in range(10):
    img_cat = cv2.imread('Cats_1\img'+str(i)+'.jpg')
    img_cat = cv2.resize(img_cat, (100, 100), interpolation=cv2.INTER_AREA)
    img_cat = np.asarray(img_cat)
    img_cat = img_cat.flatten()
    img_cat = img_cat.reshape((img_cat.shape[0],1))
    X_cats[i,:] = img_cat.T


    img_dog = cv2.imread('Dogs_1\img'+str(i)+'.jpg')
    img_dog = cv2.resize(img_dog, (100, 100), interpolation=cv2.INTER_AREA)
    img_dog = np.asarray(img_dog)
    img_dog = img_dog.flatten()
    img_dog = img_dog.reshape((img_dog.shape[0], 1))
    X_dogs[i, :] = img_dog.T

X_cats = np.append(X_cats, np.zeros((X_cats.shape[0],1)), axis=1)
X_dogs = np.append(X_dogs, np.ones((X_cats.shape[0],1)), axis=1)

dataset = np.append(X_cats,X_dogs,axis=0)

dataset[:,0:dataset.shape[1]-1] = (dataset[:,0:dataset.shape[1]-1]-np.mean(dataset[:,0:dataset.shape[1]-1],axis=0))/np.std(dataset[:,0:dataset.shape[1]-1],axis=0)

np.random.shuffle(dataset)
data_train = dataset[0:int((dataset.shape[0]*0.7)),:]
data_test = dataset[data_train.shape[0]:dataset.shape[0],:]
data_cv = data_test[0:int((data_test.shape[0]*0.5)),:]
data_test = data_test[data_cv.shape[0]:data_test.shape[0],:]


#Separating the data into features and labels
X_train = data_train[:,0:-1]
X_cv = data_cv[:,0:-1]
X_test = data_test[:,0:-1]
y_train = data_train[:,[data_train.shape[1]-1]]
y_cv = data_cv[:,[data_cv.shape[1]-1]]
y_test = data_test[:,[data_test.shape[1]-1]]


# Selecting optimal parameters
m_train = X_train.shape[0]
m_cv = X_cv.shape[0]
m_test = X_test.shape[0]
n_1 = 5
n_2 = 5
n_3 = 1
n = X_train.shape[1]
lambda_reg = 0.0001
alpha = 0.001
epsilon = 0.0001


# # .......Implementation........
#
y_train = y_train.T
W_1 = np.random.randn(n_1,n)*np.sqrt(2.0/(n))
W_2 = np.random.randn(n_2,n_1)*np.sqrt(2.0/(n_1))
W_3 = np.random.randn(n_3,n_2)*np.sqrt(2.0/(n_2))

b1 = np.zeros((n_1,1))
b2 = np.zeros((n_2,1))
b3 = np.zeros((n_3,1))

dw1 = np.zeros((W_1.shape))
dw2 = np.zeros((W_2.shape))
dw3 = np.zeros((W_3.shape))
db1 = np.zeros((b1.shape))
db2 = np.zeros((b2.shape))
db3 = np.zeros((b3.shape))
J_train = np.ndarray((1,1000))
iter = np.ndarray((1,1000))
for i in range(1000):

    z_1 = np.matmul(W_1, X_train.T) + b1
    z_1[z_1<0] = 0
    a_1 = z_1
    z_2 = np.matmul(W_2, a_1) + b2
    z_2[z_2 < 0] = 0
    a_2 = z_2
    z_3 = np.matmul(W_3, a_2) + b3
    a_3 = 1/(1+np.exp(-z_3))


    dz3 = np.asarray(a_3-y_train)
    dw3 = (1/m_train)*np.matmul(dz3,a_2.T)
    db3 = (1/m_train)*np.sum(dz3, axis=1, keepdims=True)

    dz2 = np.matmul(W_3.T,dz3)
    dz2[z_2<0] = 0
    dw2 = (1/m_train)*np.matmul(dz2,a_1.T)
    db2 = (1 / m_train) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.matmul(W_2.T, dz2)
    dz1[z_1 < 0] = 0
    dw1 = (1 / m_train) * np.matmul(dz1, X_train)
    db1 = (1 / m_train) * np.sum(dz1, axis=1, keepdims=True)
    J_train[0,i] = -(1 / (m_train)) * (np.sum((np.multiply(y_train, np.log(a_3+epsilon))) + np.multiply((1 - y_train), np.log(1 - a_3+epsilon))))# + lambda_reg/(2*m_train)*(np.linalg.norm(theta_3[:, 1:theta_3.shape[1]])+np.linalg.norm(theta_3[:, 1:theta_2.shape[1]])+np.linalg.norm(theta_3[:, 1:theta_1.shape[1]]))

    W_1 = W_1 - alpha * dw1
    W_2 = W_2 - alpha * dw2
    W_3 = W_3 - alpha * dw3
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    b3 = b3 - alpha * db3
    if i == 0:
        print('The cost in iteration ' + str(i) + ' is ' + str(J_train[0, i]))
    if (i + 1) % 100 == 0:
        print('The cost in iteration ' + str(i) + ' is ' + str(J_train[0, i]))
plt.plot(iter[0, :], J_train[0, :], color='red', label='J_train')
plt.show()

pred_train = predict(X_train,W_1,W_2,W_3,b1,b2,b3)
pred_cv = predict(X_cv,W_1,W_2,W_3,b1,b2,b3)
pred_test = predict(X_test,W_1,W_2,W_3,b1,b2,b3)

acc_train = ((pred_train[np.ravel(y_train)==pred_train]).shape[0]/(pred_train).shape[0])*100
acc_cv = (pred_cv[np.ravel(y_cv)==pred_cv].shape[0]/pred_cv.shape[0])*100
acc_test = (pred_test[np.ravel(y_test)==pred_test].shape[0]/pred_test.shape[0])*100

print('training accuracy is ' + str(acc_train))
print('Validation accuracy is ' + str(acc_cv))
print('test accuracy is ' + str(acc_test))