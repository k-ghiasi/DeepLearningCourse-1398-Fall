import scipy.io as sio
from Tensor import Tensor
from Net import Net
from InnerProductLayer import InnerProductLayer
from AccuracyLayer import AccuracyLayer
from SoftMaxWithCrossEntropyLossLayer import SoftMaxWithCrossEntropyLossLayer
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from ReluLayer import ReluLayer

noEpochs = 30
batch_size = 100
draw_rate = 100

net = Net()

ip1 = InnerProductLayer(name='ip1', output_neurons=100)
ip1.input_tensors_names = ['X']
ip1.output_tensors_names = ['Z1']

relu1 = ReluLayer (name = 'ReLU1')
relu1.input_tensors_names = ['Z1']
relu1.output_tensors_names = ['Y1']

ip2 = InnerProductLayer(name='ip2', output_neurons=10)
ip2.input_tensors_names = ['Y1']
ip2.output_tensors_names = ['Z2']

soft_max_cross_entropy = SoftMaxWithCrossEntropyLossLayer(name='softmaxce1')
soft_max_cross_entropy.input_tensors_names = ['Z2', 'T']
soft_max_cross_entropy.output_tensors_names = ['L']

accuracy = AccuracyLayer(name='acc1')
accuracy.input_tensors_names = ['Z2', 'T']
accuracy.output_tensors_names = ['A']

net.layers = [ip1, relu1, ip2, soft_max_cross_entropy, accuracy]

MnistTrainX = sio.loadmat('../../../datasets/mnist/MnistTrainX')['MnistTrainX']
MnistTrainY = sio.loadmat('../../../datasets/mnist/MnistTrainY')['MnistTrainY']
MnistTestX = sio.loadmat('../../../datasets/mnist/MnistTestX')['MnistTestX']
MnistTestY = sio.loadmat('../../../datasets/mnist/MnistTestY')['MnistTestY']

MnistTrainX = MnistTrainX / 255
MnistTestX  = MnistTestX  / 255
(N, d) = MnistTrainX.shape
NTest = MnistTestX.shape[0]

num_iterations = N // batch_size

X = Tensor(name='X', dimensions=[batch_size, d])
T = Tensor(name='T', dimensions=[batch_size])

acc_train: ndarray = np.ones(noEpochs * num_iterations // draw_rate) * -1
acc_test: ndarray = np.ones(noEpochs * num_iterations // draw_rate) * -1

fig1 = plt.figure(1, figsize=(8, 8))
ax_train = plt.subplot(2, 1, 1)
ax_test = plt.subplot(2, 1, 2)
# _x = np.arange(0, noEpochs * num_iterations // draw_rate)
plt.show(block=False)

for epoch in range(noEpochs):
    print('\n--------------- epoch #{0} of {1} --------------- :\n'.format(epoch, noEpochs))

    for it in range(num_iterations):
        X.data = MnistTrainX[it*batch_size:(it+1)*batch_size, :]
        T.data = MnistTrainY[it*batch_size:(it+1)*batch_size]
        if it == 0 and epoch == 0:
            net.setup([X, T])
        net.forward([X, T])
        net.backward()
        net.optimization_step(0.001)

        if (it % draw_rate == 0):
            score = 0.0
            for i in range(num_iterations):
                X.data = MnistTrainX[i * batch_size:(i + 1) * batch_size, :]
                T.data = MnistTrainY[i * batch_size:(i + 1) * batch_size]
                net.forward([X, T])
                score += np.sum(net.tensors['A'].data).squeeze()

            score /= N
            score *= 100
            print('Accuracy on training data = {0}%'.format(score))
            acc_train[epoch * num_iterations // draw_rate + it // draw_rate] = score

            score = 0.0
            for i in range(NTest // batch_size):
                X.data = MnistTestX[i * batch_size:(i + 1) * batch_size, :]
                T.data = MnistTestY[i * batch_size:(i + 1) * batch_size]
                net.forward([X, T])
                score += np.sum(net.tensors['A'].data).squeeze()

            score /= NTest
            score *= 100
            print('Accuracy on testing data = {0}%'.format(score))
            acc_test[epoch * num_iterations //
                     draw_rate + it // draw_rate] = score

            _x = np.arange(0, acc_train[acc_train > -1].shape[0])
            ax_train.clear()
            ax_train.plot(_x, acc_train[acc_train > -1], label='train accuracy', color='royalblue', linewidth=2)
            ax_train.set_ylim(bottom=0)
            ax_train.set_xlim(left=0)
            ax_train.legend()
            ax_train.grid()

            ax_test.clear()
            ax_test.plot(_x, acc_test[acc_test > -1], label='test accuracy', color='green', linewidth=2)
            ax_test.set_ylim(bottom=0)
            ax_test.set_xlim(left=0)
            ax_test.legend()
            ax_test.grid()
            
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            print('    epoch #{0} > iteration #{1} of {2}:\n'.format(epoch, it, num_iterations))


score = 0.0
for it in range(600):
    X.data = MnistTrainX[it * batch_size:(it + 1) * batch_size, :]
    T.data = MnistTrainY[it * batch_size:(it + 1) * batch_size]
    net.forward([X, T])
    score += np.sum(net.tensors['A'].data).squeeze()

score /= 60000.0
score *= 100

print('\n\nAccuracy on train data = {0}%'.format(score))


MnistTestX = sio.loadmat('../../../datasets/mnist/MnistTestX')['MnistTestX']
MnistTestY = sio.loadmat('../../../datasets/mnist/MnistTestY')['MnistTestY']

score = 0.0
for it in range(100):
    X.data = MnistTestX[it * batch_size:(it + 1) * batch_size, :]
    T.data = MnistTestY[it * batch_size:(it + 1) * batch_size]
    net.forward([X, T])
    score += np.sum(net.tensors['A'].data).squeeze()

score /= 10000.0
score *= 100

print('Accuracy on test data = {0}%'.format(score))

input('\n\npress Enter to end...\n')
