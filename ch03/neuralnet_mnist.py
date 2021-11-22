import numpy as np
import math as mt

def sigmoid(x) :
    return 1/(1 + np.exp(-x))

def identity_function(x) :
    return x

def init_network() :
    """
    初期値を与えて，3層のネットワークを作成する
    """
    network = {}
    #1層目の重みを作成
    network["W1"] = np.array([[0.1, 0.3, 0.5],
                              [0.2, 0.4, 0.5]])
    #1層目のバイアス
    network["b1"] = np.array([0.1, 0.2, 0.3])
    #2層目の重み
    network["W2"] = np.array([[0.1, 0.4],
                              [0.2, 0.5],
                              [0.3, 0.6]])
    #2層目のバイアス
    network["b2"] = np.array([0.1, 0.2])
    
    #最終層の重み
    network["W3"] = np.array([[0.1, 0.3],
                              [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network

def forward(network, x) :
    """3層ニューラルネットの順伝搬の計算を行う関数"""
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


def main() :
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)

if __name__ == "__main__" :
    main()
