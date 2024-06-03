import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#Cargar imagen
imagen = cv.imread('/home/sc02m01ia/Música/EntrenamientoIA/jordan.jpg')

#Normalizar
imagen = imagen / 255.0
imagen = np.array(imagen, dtype='float32')

#Convertir a escala de gris
imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
imagen_expandida = np.expand_dims(imagen_gris, axis=(0))

#Convertir la imagen a dato de tf
x = tf.constant(imagen_expandida, dtype=tf.float32)


#No se que hace
k = np.array([[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]])


k = np.array(k, dtype='float32')
k = k[:, :, np.newaxis]
k = np.tile(k,(1, 1, 1))


kernel = tf.constant(k, dtype='float32')


convolucion = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    resultado = sess.run(convolucion)

plt.imshow(resultado[0], cmap='gray')
plt.show()


