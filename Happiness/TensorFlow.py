import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import happiness
import time


# 创建一个神经网络层
def add_layer(input, in_size, out_size, activation_function=None):
    """
    :param input: 数据输入
    :param in_size: 输入大小
    :param out_size: 输出大小
    :param activation_function: 激活函数（默认没有）
    :return:output：数据输出
    """
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    Biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    W_mul_x_plus_b = tf.matmul(input, Weight) + Biases
    # 根据是否有激活函数
    if activation_function == None:
        output = W_mul_x_plus_b
    else:
        output = activation_function(W_mul_x_plus_b)
    return Weight, Biases, output


def forward_propagation(X):
    w1, b1, hidden_layer1 = add_layer(X, 1, 4, activation_function=tf.nn.relu)
    w2, b2, hidden_layer2 = add_layer(hidden_layer1, 4, 3, activation_function=tf.nn.relu)
    w3, b3, y_prediction = add_layer(hidden_layer2, 3, 1, activation_function=None)
    params = {}
    params['w1'] = w1
    params['b1'] = b1
    params['w2'] = w2
    params['b2'] = b2
    params['w3'] = w3
    params['b3'] = b3

    return params, y_prediction


def y_predict(x, params):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    w3 = params['w3']
    b3 = params['b3']
    x = x[:, np.newaxis]
    a1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    a2 = tf.nn.relu(tf.matmul(a1, w2) + b2)
    y_prediction = tf.matmul(a2, w3) + b3
    return y_prediction


start_time = time.time()
train = pd.read_csv("./data/happiness_train_abbr.csv")
test = pd.read_csv("./data/happiness_test_abbr.csv")

train, features = happiness.load_train_data(train)
print(np.shape(train[features].values.tolist()))
features.remove('happiness')
test = happiness.load_test_data(test, features)

dateSet = train[features].values.tolist()
print(np.shape(dateSet))

# x = dateSet[:, 0:-1].reshape(-1, 1).astype(np.float32)
# y = dateSet[:, -1].reshape(-1, 1).astype(np.float32)
#
# print(x)
#
# # 1定义输入数据
# X = tf.placeholder(tf.float32, [None, 1], name="X")
# Y = tf.placeholder(tf.float32, [None, 1], name="Y")
#
# # 2定义损失函数及反向传播方法。
# params, y_prediction = forward_propagation(X)
# loss = tf.reduce_mean(tf.square(y - y_prediction))
# optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)  # 三种优化方法选择一个就可以
# # train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
# # train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)
#
# # 3生成会话，训练STEPS轮
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     STEPS = 0
#     delta = 9999
#     while np.fabs(delta) > 1e-3:
#
#         if STEPS > 188888:
#             break
#         STEPS = STEPS + 1
#
#         _, epoch_cost = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
#         if STEPS % 100 == 1:
#             epoch_cost_t = epoch_cost
#         if STEPS % 100 == 0:
#             delta = epoch_cost - epoch_cost_t
#             print("After %d training step(s), loss on all data is %f" % (STEPS, epoch_cost))
#
#     print("STEPS = ", STEPS)
#
#     y_p = sess.run(y_predict(y[2], params=sess.run(params)))
#
#     re = np.loadtxt("re.txt")
#     np.savetxt("re.txt", np.append(re, y_p), fmt="%d")
#
# sess.close()

print("CPU的执行时间 = " + '%f' % (time.time() - start_time) + " 秒")
