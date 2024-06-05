import tensorflow.compat.v1 as tf 
import cv2
import os
import numpy as np

tf.disable_v2_behavior()

def convolucion(img, kernel):
    return tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME') 

def images():
    l = os.listdir("images/")
    limg = []
    for img in l:
        image = cv2.imread("images/" + img)
        image = cv2.resize(image, (740, 740))
        image = image / 255.0
        limg.append(image)
    limg = np.array(limg, dtype='float32')
    return limg

def creaFiltros():
    f1 = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    f2 = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    f3 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    f4 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    f = [f1, f2, f3, f4]
    f = np.array(f, dtype='float32')
    f = f[:, :, :, np.newaxis]
    
    f = np.transpose(f, (1, 2, 3, 0))
    f = np.tile(f,(1,1,3,1)) 
    return f

if __name__ == '__main__':
    with tf.device("/gpu:0"):
        with tf.Graph().as_default():            
            x = tf.placeholder("float32", [None, 740, 740, 3])  
            y = tf.placeholder("float32", [3, 3, 3, 4])  

            img = images()
            kernel = creaFiltros()
            conv = convolucion(x, y)
            sess = tf.Session()
            r = sess.run(conv, feed_dict={x: img, y: kernel})
            
            for i, imagen in enumerate(r):
                for j in range(4):
                    cv2.imshow("imagen", imagen[:, :, j])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
