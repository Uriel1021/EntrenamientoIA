import numpy as np
import cv2 as cv
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

imagen = cv.imread('/home/sc02m01ia/Música/EntrenamientoIA/jordan.jpg') 
imagen = np.expand_dims(imagen, axis=0).astype(np.float32)

x = tf.constant(imagen, dtype=tf.float32)

k = np.array([[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]])

k = np.stack([k, k, k], axis=-1)
k = np.expand_dims(k, axis=0)

convolucion = tf.nn.conv2d(x, k, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    resultado = sess.run(convolucion)

cv.imshow('Resultado', np.squeeze(resultado[0]).astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()
