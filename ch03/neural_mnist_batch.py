import sys , os
sys.path.append(os.pardir) #親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import numpy as np
import math as mt
import pickle
from PIL import Image

def get_data() :
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten = True, normalize = False)
    return x_test, t_test

def init_network() :
    with open("sample_weight.pkl", "rb") as f :
        network = pickle.load(f)
    return network


def predict(network, x) :
    """3層ニューラルネットの順伝搬の計算を行う関数"""
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


# def softmax(a) :
#     c = np.max(a)
#     exp_a = np.exp(a - c)#cはオーバフロー対策のための項
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#
#     return y

# def img_show(img) :
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()

def main() :
    batch_size = 100
    x, t = get_data()
    accuracy_cnt = 0
    network = init_network()
    for i in range(0, len(x), batch_size) :
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis = 1) #最も確率(softmax関数の出力)が高いインデックスを取得する．最大値は行方向に探索する
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


if __name__ == "__main__" :
    main()
