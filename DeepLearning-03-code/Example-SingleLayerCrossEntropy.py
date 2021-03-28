# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

import scipy.io as sio
import numpy as np

from SGD import SGD
from Tensor import Tensor
from Net import Net
from InnerProductLayer import InnerProductLayer
from AccuracyLayer import AccuracyLayer
from SoftMaxWithCrossEntropyLossLayer import SoftMaxWithCrossEntropyLossLayer

noEpochs = 2
batch_size = 100
report_after_X_iterations = 100

net = Net()
inner_product = InnerProductLayer(name='ip1', output_dim=10)
inner_product.input_tensors_names = ['X']
inner_product.output_tensors_names = ['Z']

soft_max_cross_entropy = SoftMaxWithCrossEntropyLossLayer(name='softmax_ce1')
soft_max_cross_entropy.input_tensors_names = ['Z', 'T']
soft_max_cross_entropy.output_tensors_names = ['L']

accuracy = AccuracyLayer(name='acc1')
accuracy.input_tensors_names = ['Z', 'T']
accuracy.output_tensors_names = ['A']

net.layers = [inner_product, soft_max_cross_entropy, accuracy]


MnistTrainX = sio.loadmat('../../datasets/mnist/MnistTrainX')['MnistTrainX']
MnistTrainY = sio.loadmat('../../datasets/mnist/MnistTrainY')['MnistTrainY']
MnistTestX = sio.loadmat('../../datasets/mnist/MnistTestX')['MnistTestX']
MnistTestY = sio.loadmat('../../datasets/mnist/MnistTestY')['MnistTestY']

MnistTrainX = MnistTrainX / 255
MnistTestX  = MnistTestX  / 255
(N, d) = MnistTrainX.shape
NTest = MnistTestX.shape[0]

num_iterations = N // batch_size

X = Tensor(name='X', dimensions=[batch_size, d])
T = Tensor(name='T', dimensions=[batch_size])

for epoch in range(noEpochs):
    print('\n--------------- epoch #{0} of {1} --------------- :\n'.format(epoch, noEpochs))

    for itr in range(num_iterations):
        X.data = MnistTrainX[itr*batch_size:(itr+1)*batch_size, :]
        T.data = MnistTrainY[itr*batch_size:(itr+1)*batch_size]
        if itr == 0 and epoch == 0:
            net.setup([X, T])
            sgd = SGD(net.parameters(), lr=0.001)
        net.forward([X, T])
        sgd.zero_grad()
        net.backward()
        sgd.step()

        if (itr % report_after_X_iterations == 0):
            print('\n---- iteration #{0} of {1} ---- :'.format(itr, num_iterations))
            score = 0.0
            for i in range(num_iterations):
                X.data = MnistTrainX[i * batch_size:(i + 1) * batch_size, :]
                T.data = MnistTrainY[i * batch_size:(i + 1) * batch_size]
                net.forward([X, T])
                score += np.sum(net.tensors['A'].data).squeeze()

            score /= N
            score *= 100
            print('Accuracy on training data = {0}%'.format(score))

            score = 0.0
            for i in range(NTest // batch_size):
                X.data = MnistTestX[i * batch_size:(i + 1) * batch_size, :]
                T.data = MnistTestY[i * batch_size:(i + 1) * batch_size]
                net.forward([X, T])
                score += np.sum(net.tensors['A'].data).squeeze()

            score /= NTest
            score *= 100
            print('Accuracy on testing data = {0}%'.format(score))


score = 0.0
for itr in range(600):
    X.data = MnistTrainX[itr * batch_size:(itr + 1) * batch_size, :]
    T.data = MnistTrainY[itr * batch_size:(itr + 1) * batch_size]
    net.forward([X, T])
    score += np.sum(net.tensors['A'].data).squeeze()

score /= 60000.0
score *= 100

print('\n\nAccuracy on train data = {0}%'.format(score))

score = 0.0
for itr in range(100):
    X.data = MnistTestX[itr * batch_size:(itr + 1) * batch_size, :]
    T.data = MnistTestY[itr * batch_size:(itr + 1) * batch_size]
    net.forward([X, T])
    score += np.sum(net.tensors['A'].data).squeeze()

score /= 10000.0
score *= 100

print('Accuracy on test data = {0}%'.format(score))

