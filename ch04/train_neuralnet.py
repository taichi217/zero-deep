import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from two_layer_net import TwoLayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

#ハイパーパラメータ
train_size = x_train.shape[0]
batch_size = 100
iters_num = 1000
learning_rate = 0.1
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

network = TwoLayerNet(input_size = 784, hidden_size = 10, output_size=10)

loss = 0
for i in range(iters_num) :
    #ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)

    #パラメータの更新
    for key in ("W1", "b1", "W2", "b2") :
        network.params[key] -= learning_rate*grad[key]

    #学習経過を記録する
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print("current loss : ".format(loss))
    print("loss_log : ".format(train_loss_list))

# グラフの描画
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(len(train_acc_list))
plt.plot([i for i in range(len(train_loss_list))], train_loss_list, label='train loss')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("iteration")
plt.ylabel("loss")
# plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
