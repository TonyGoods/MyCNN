import os
import numpy as np
from PIL import Image
import tensorflow as tf


def getImageValue():
    file_path = '../images/vertification Image'
    image_files = os.listdir(file_path)
    X_data = []
    y_data = []
    index = 0
    for image_file in image_files:
        for image in os.listdir(file_path + '/' + image_file):
            image_data = Image.open(file_path + '/' + image_file + '/' + image).convert('1')
            X_data.append([np.asarray(image_data, dtype=float)])
            # print(np.asarray(image_data, dtype=float))
            y = np.zeros([33], dtype=float)
            y[index] = 1.0
            y_data.append(y.tolist())
        index += 1
    return np.reshape(X_data, [-1, 24, 15, 1]), np.reshape(y_data, [-1, 33])


def getWeight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64))


def getBias(shape):
    return tf.Variable(tf.constant(0.1, dtype=tf.float64, shape=shape))


def mainCode():
    with tf.name_scope('input'):
        x, y = getImageValue()

    with tf.name_scope('conv1'):
        conv1_weight = getWeight([5, 5, 1, 32])
        bias1 = getBias([32])
        conv1 = tf.nn.relu(tf.nn.conv2d(x, conv1_weight, strides=[1, 1, 1, 1], padding='SAME') + bias1)

    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv2'):
        conv2_weight = getWeight([5, 5, 32, 64])
        bias2 = getBias([64])
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME') + bias2)

    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    conn = tf.reshape(pool2, (-1, 4 * 6 * 64))
    with tf.name_scope('conn1'):
        conn1_weight = getWeight([4 * 64 * 6, 128])
        bias_conn1 = getBias([128])
        conn1 = tf.nn.relu(tf.matmul(conn, conn1_weight) + bias_conn1)
        conn1_dropout = tf.nn.dropout(conn1, keep_prob=0.8)

    with tf.name_scope('conn2'):
        conn2_weight = getWeight([128, 33])
        bias2_conn = getBias([33])
        prediction = tf.nn.softmax(tf.matmul(conn1_dropout, conn2_weight) + bias2_conn)

    with tf.name_scope('loss'):
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.9, staircase=True)
        loss = tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(200):
            print(str(i) + '...')
            sess.run(train_step)
            print(sess.run(loss))
            if i % 10 == 0:
                count = 0
                y_index = tf.argmax(y, 1).eval()
                x_index = tf.argmax(sess.run(prediction), 1).eval()
                print(x_index)
                print(y_index)
                for i in range(1595):
                    if x_index[i] == y_index[i]:
                        count += 1
                print(str(count) + "    " + str(count / 1595))
    write = tf.summary.FileWriter('log', tf.get_default_graph())
    write.close()


if __name__ == '__main__':
    mainCode()
