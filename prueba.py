import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Cargar imagen
imagen = cv.imread('/home/sc02m01ia/Música/EntrenamientoIA/jordan.jpg')

# Normalizar
imagen = imagen / 255.0
imagen = np.array(imagen, dtype='float32')

# Convertir a escala de gris
imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
imagen_expandida = np.expand_dims(imagen_gris, axis=(0))
imagen_expandida = np.expand_dims(imagen_expandida, axis=-1)  # Agregar una dimensión para el canal de color

# Convertir la imagen a dato de tf
x = tf.constant(imagen_expandida, dtype=tf.float32)

# Kernel de convolución
k = np.array([[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]], dtype='float32')

k = np.expand_dims(k, axis=-1)  # Agregar una dimensión para el canal de entrada
k = np.expand_dims(k, axis=-1)  # Agregar una dimensión para el canal de salida
kernel = tf.constant(k, dtype='float32')

# Operación de convolución
convolucion = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    resultado = sess.run(convolucion)

plt.imshow(resultado[0, :, :, 0], cmap='gray')  # Mostrar solo el primer canal de salida
plt.show()
