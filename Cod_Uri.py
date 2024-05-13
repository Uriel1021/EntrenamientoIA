import csv
import os
import numpy as np
import matplotlib.pyplot as plt

dir = os.listdir()
dir[1]

# leer el archivo
file = open(dir[1],"r")
data = csv.reader(file)
file.close()

# Declarar variables
label_train = []
img_train = []


# omitir la primera linea
next(data)

#Recorrer el primer reglon
for data in data:
    label_train.append(data[0])
    #Convertirlo a entero
    img_vect = np.array(data[1:], dtype = "int64")
    img_train.append(img_vect)




print("Img_train: " + np.shape(img_train))
print("Img_train: " + np.shape(label_train))

print("Etiqueta:" + label_train[100])
plt.imshow(img_train[100].reshape(28,28),cmap="gray")
plt.show()

label = {
    "0": [1,0,0,0,0,0,0,0,0,0],
    "1": [0,1,0,0,0,0,0,0,0,0],
    "2": [0,0,1,0,0,0,0,0,0,0],
    "3": [0,0,0,1,0,0,0,0,0,0],
    "4": [0,0,0,0,1,0,0,0,0,0],
    "5": [0,0,0,0,0,1,0,0,0,0],
    "6": [0,0,0,0,0,0,1,0,0,0],
    "7": [0,0,0,0,0,0,0,1,0,0],
    "8": [0,0,0,0,0,0,0,0,1,0],
    "9": [0,0,0,0,0,0,0,0,0,1]
}


#definir un diccionario
etiqueta = []
for i in label_train:
    etiqueta.append




label = {"0":[1,0,0,0],"1":[0,1,1,1]}

label_train(label[label_train[0]])


tamBatch = 64

for e in range(100)
