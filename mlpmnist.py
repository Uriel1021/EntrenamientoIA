
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import datos
import numpy as np

# Arquitectura del modelo
n_hidden_1 = 256
n_hidden_2 = 256
numEpocas = 100
batch_size = 128
learning_rate=0.00001

mnist_train = '/home/sc02m01ia/Descargas/code/mnist_train.csv' 
mnist_test = '/home/sc02m01ia/Descargas/code/mnist_test.csv'

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,initializer=weight_init)
    b = tf.get_variable("b", bias_shape,initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

def inference(x):
    with tf.variable_scope("input_layer"):
        hidden_1 = layer(x, [784, n_hidden_1], [n_hidden_1])
    with tf.variable_scope("hidden_layer"):
        hidden_2 = layer(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2])
    with tf.variable_scope("output_layer"):
        output = layer(hidden_2, [n_hidden_2, 10], [10])
    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)    
    loss = tf.reduce_mean(xentropy)
    return loss


def training(cost):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op



def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

if __name__ == '__main__':

    with tf.Graph().as_default():

            x = tf.placeholder("float", [None, 784]) #  28*28=784
            y = tf.placeholder("float", [None, 10]) # 10 clases


            output = inference(x)
            cost = loss(output, y)
            train_op = training(cost)
            eval_op = evaluate(output, y)


            sess = tf.Session()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
     
            #Recupera los datos
            label_train, img_train = datos.get_data(mnist_train)

            label_test, img_train_test = datos.get_data(mnist_test)


            nDatos=np.shape(img_train)[0]
            nBatch=int(nDatos/batch_size)
            pos = np.arange(nDatos)

            
            nBatch_test = int(np.shape(img_train_test)[0]/batch_size)


            listError = []
            listError_test = []
            for i in range(numEpocas):
                error = 0.0
                error_test = 0.0
                np.random.shuffle(pos)
                for j in range(nBatch):
                    label_batch, img_batch = datos.next_batch(j, pos, label_train, img_train)    
                    sess.run(train_op, feed_dict={x: img_batch, y: label_batch})
                    error += (sess.run(cost, feed_dict={x: img_batch, y: label_batch})) / nBatch
                print("Epoca: ", i, " Batch: ", j, " Error train: ", error)
                listError.append(error)
                
                if i % 5 == 0:  
                    for j in range(nBatch_test):
                        label_batch_test, img_batch_test = datos.next_batch_test(j, label_test, img_train_test)
                        error_test += (sess.run(cost, feed_dict={x: img_batch_test, y: label_batch_test})) / nBatch_test
                    print("Epoca: ", i, " Batch: ", j, " Error test: ", error_test)
                    listError_test.append(error_test)





            plt.plot(listError[2:], label='Error de entrenamiento')
            plt.xlabel('Epocas')
            plt.ylabel('Error')
            plt.legend()
            plt.show()


            plt.plot(listError_test[2:], label='Error de prueba')
            plt.xlabel('Epocas')
            plt.ylabel('Error')
            plt.legend()
            plt.show()
        #512 256 128 10
    

