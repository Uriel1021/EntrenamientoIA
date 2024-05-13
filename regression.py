import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()


def puntos(nump):
	x=[]
	y=[]
	for i in range(nump):
		xl=np.random.normal(0.0,0.5)
		yl=xl*0.1+0.3+np.random.normal(0.0,0.03)
		x.append(xl)
		y.append(yl)	
	return x,y

def modelo(x):
	w=tf.Variable(tf.random_uniform([1],-1.0,1.0),name="pendiente")
	b=tf.Variable(tf.random_uniform([1]),name="bias")
	return tf.add(tf.multiply(w,x),b)


def loss(output,y):	
	return tf.reduce_mean(tf.square(output-y))



def train(cost):	
	optimizer=tf.train.GradientDescentOptimizer(0.5)
	return optimizer.minimize(cost)

if __name__=="__main__":
	with tf.Graph().as_default():
		with tf.device("/gpu:0"):

			xp,yp=puntos(100)
			x=tf.placeholder(tf.float32,[None,100])
			y=tf.placeholder(tf.float32,[None,100])

			output=modelo(x)
			cost=loss(output,y)
			training=train(cost)

			init_op=tf.global_variables_initializer()
			sess=tf.Session()
			sess.run(init_op)
	

			for i in range(10):
				sess.run(training,feed_dict={x:[xp],y:[yp]})
			yo=sess.run(output)
			
			plt.plot(x,y,'ro',label='Original Data')
			plt.plot(x,yo)
			plt.show()
