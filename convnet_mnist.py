import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import datos
import os
import numpy as np
from tqdm import tqdm

# Parameters
learning_rate = 0.00001
batch_size = 128
numEpocas = 5


def conv2d(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

def inference(x):
    with tf.variable_scope("capa1"):
        out1 = conv2d(x, [5,5,1,32],[32])
        outmax = max_pool(out1)

    with tf.variable_scope("capa2"):
        out2 = conv2d(outmax,[5,5,32,64],[64])
        outmax = max_pool(out2)

    with tf.variable_scope("capa3"):
        out3 = tf.reshape(outmax, (-1, 7*7*64))
        out4 = layer(out3, [7*7*64, 1024], [1024])
    
    with tf.variable_scope("capa4"):
        outLayer = layer(out4, [1024, 10], [10])
    
    return outLayer

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)    
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

if __name__ == '__main__':
    with tf.device("/CPU:0"):
        x = tf.placeholder("float", [None, 28, 28, 1])
        y = tf.placeholder("float", [None, 10])

        output = inference(x)
        error = loss(output, y)
        entrena = training(error)
        evalua = evaluate(output, y)

        saver = tf.train.Saver()
        sess = tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        label_train, img_train = datos.get_data("mnist_train")
        label_test, img_train_test = datos.get_data("mnist_test")
        
        nDatos = np.shape(img_train)[0]
        nBatch = int(nDatos / batch_size)
        pos = np.arange(nDatos)
        
        nDatosTest = np.shape(img_train_test)[0]
        nBatchTest = int(nDatosTest / batch_size)

        error_train = []
        error_test = []

        for i in tqdm(range(numEpocas)):
            np.random.shuffle(pos)            
            e = 0.0
            e_test = 0.0
            for j in range(nBatch):
                label_batch, img_batch = datos.next_batch(j, pos, label_train, img_train)
                sess.run(entrena, feed_dict={x: img_batch, y: label_batch})
                e += sess.run(error, feed_dict={x: img_batch, y: label_batch}) / nBatch

                label_batch_test, img_batch_test = datos.next_batch_test(j, label_test, img_train_test)
                sess.run(entrena, feed_dict = {x: img_batch_test, y: label_test})
                e_test += sess.run(error, feed_dict = {x:img_batch_test, y:label_test})/ nBatchTest

            error_train.append(e)
            print("Epoca: ", i, " error_train : ", e)
            print("Epoca: ", i, " error_test : ",e)



            # Guarda cada 5 épocas
            if (i + 1) % 5 == 0:
                epoca_directorio = f"modelo_epoca_{i + 1}"
                if not os.path.exists(epoca_directorio):
                    os.makedirs(epoca_directorio)

                saver.save(sess,os.path.join(epoca_directorio, "modelo.ckpt"))
    
    plt.plot(error_train[2:], label='Error de entrenamiento')
    plt.xlabel('Epocas')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    plt.plot(error_test[2:], label='Error de prueba')
    plt.xlabel('Epocas')
    plt.ylabel('Error')
    plt.legend()
    plt.show()