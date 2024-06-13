# Red convolucional
#       _________            ____         ___________                      ___
#    __|_______ |           _|__|         |         |                    _|__| 25     ________
#   _|_______ | |          _|__|        w | feature |                   _|__|         |       |
#   |       | |_|--------  |__|3----------|   maps  |-------------------|__| 3--------|       | w  125
#  w|       |_|              3            |_________|                     3           |_______|
#   |_______|                                  h                                          h
#       h                 ... 25           ... 25                        
#                                                                                                                                                           ___
#                                                                                                                                                          /   \
#                                                                                                                                                          |   |  
#                    Capa                Max                                                   Capa                                                       /\___/  
#               Convolucional          Pooling                                             Convolucional                                        _____    /
#                    _1__                ___                                                                                                    |   |   / 7,7 64
#                    |__|3              |   |                                               ______             _______                          |   |  /    ___
#   _________          3                |   |       _________                              _|___ |          __|_____  |                         |   | /    /   \ 
#   |       |        __1_               |   |       |       |        64 neuronas ->      _|___ |_|          |       | | 64  ---------------->>> |   |/     |   |    
# 28|       |--------|__|3 -------------|   |-------|       |-------------------------- |    |_|  32 ------ |       |_|       Conectar          |   |      \___/
#   |_______|          3                |   |     14|_______| 32                       3|____|            14|_______|     con clasificador      |   |
#      28            __1_               |   |           14                                3                     14                              |___|       ___
#                    |__|3              |___|                                                                                                              /   \
#                      3                                                                                                                       Entrada     |   |
#                                                                                                                                                          \___/     
#                    ... 32                                                                                                                                 
#                                                                                                                                                        1024 neuronas          

import tensorflow.compat.v1 as tf 
import cv2
import numpy as np
import datos
mnist = datos.read_data_sets("./", one_hot=True)

tf.disable_v2_behavior()


# Parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 128
display_step = 1

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
        out1 = convolucion(x, [5,5,1,32],[32])
        outmax = fun_max_pool(out1)

    with tf.variable_scope("capa2"):
        out2 = convolucion(outmax,[5,5,32,64],[64])
        outmax = fun_max_pool(out2)

    with tf.variable_scope("capa3"):
        out3 = tf.reshape(outmax,(-1,7,7,64))
        out4 = mlp(out3,[7,7,64,1024],[1024])
    
    with tf.varible_scope("capa4"):
        outLayer = mlp(out4,[1024,10],[10])
    
    return outLayer

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)    
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name="Adam")
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation", (1.0 - accuracy))
    return accuracy

if __name__ == '__main__':
    with tf.device("/gpu:0"):
        #Tensores
        x = tf.placeholder("float32", [None, 28, 28, 1])
        y = tf.placeholder("float32", [None, 10])
        #Modelito xd
        output = inference(x)
        error = loss(output, y)
        entrenamiento = training(error)
        evaluacion = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()
        sess = tf.Session()

        #summary_writer = tf.train.SummaryWriter("conv_mnist_logs/", graph_def=sess.graph_def)
        summary_writer = tf.summary.FileWriter("conv_mnist_logs/", graph=sess.graph)
                
        init_op = tf.initialize_all_variables()

        sess.run(init_op)

                
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            print("Total Batch:",total_batch)
            # Loop over all batches
            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})/total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print ("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))
                    accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1})
                    print ("Validation Error:", (1 - accuracy))
                    summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})
                    summary_writer.add_summary(summary_str, sess.run(global_step))
                    saver.save(sess, "conv_mnist_logs/model-checkpoint", global_step=global_step)

            print ("Optimization Finished!")

            accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})

            print ("Test Accuracy:", accuracy)


