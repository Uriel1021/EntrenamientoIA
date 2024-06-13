
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import datos
import numpy as np
from tqdm import tqdm
# Parameters
learning_rate = 0.0001
training_epochs = 100

batch_size = 64
learning_rate=0.001

numEpocas=30

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
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 1, 32], [32])
        pool_1 = max_pool(conv_1)
    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
        pool_2 = max_pool(conv_2)
    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
        fc_1 = layer(pool_2_flat, [7*7*64, 1024], [1024])
    with tf.variable_scope("output"):
        output = layer(fc_1, [1024, 10], [10])
    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)    
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name="Adam")
    train_op = optimizer.minimize(cost)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

if __name__ == '__main__':
    with tf.device("/CPU:0"):

        x = tf.placeholder("float", [None,28,28,1]) #  28*28=784
        y = tf.placeholder("float", [None, 10]) # 10 clases


        output=inference(x)
        error=loss(output,y)
        entrena=training(error)
        evalua=evaluate(output,y)

        saver = tf.train.Saver()
        sess=tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)


        label_train, img_train=datos.get_data("mnist_train")

        nDatos=np.shape(img_train)[0]
        nBatch=int(nDatos/batch_size)
        pos=np.arange(nDatos)


        error_train=[]
        for i in tqdm(range(numEpocas)):
            np.random.shuffle(pos)            
            
            e=0.0

            if e % 5 == 0:
                for j in range(nBatch):
                    label_batch,img_batch=datos.next_batch(j,pos,label_train,img_train)
                    sess.run(entrena,feed_dict={x:img_batch,y:label_batch})
                    e +=sess.run(error,feed_dict={x:img_batch,y:label_batch})/nBatch
                    saver.save(sess,".ckpt")

            error_train.append(e)

            print("Epoca: ",i, " error_train : ", e)
