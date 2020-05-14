# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:46:54 2020

@author: Ivanxrto
"""

import cv2
from os import listdir
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np


imagenes = []

matricula_cascade = cv2.CascadeClassifier('./haar/matriculas.xml')
frontal_cascade = cv2.CascadeClassifier('./haar/coches.xml')


def comprobarmatricula(date_image):

    trozosMatricula = [];
    for i in date_image:
        im = i[0]
        imc = i[1]

        imagenP = matricula_cascade.detectMultiScale(im, 1.1, 11)  # 1parametro imagen
        #  if imagenP is not():
        # Por cada matricula detectada, dibujamos un rectangulo
        if imagenP is not ():
            for (x, y, w, h) in imagenP:
                # Recorto la matricula
                imagen_GrisRecortada = im[y: (y + h), x: (x + w)]

                # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
                ret, th3 = cv2.threshold(imagen_GrisRecortada, 0, 255, cv2.THRESH_OTSU)
                # https://www.programcreek.com/python/example/89437/cv2.boundingRect
                contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contornos = []
                for cor in contours:

                    # Sacamos coordenadas de contorno
                    a, b, c, d = cv2.boundingRect(cor)

                    if d > c and c > 3 and d > 10:
                        contornos.append(cor)
                        #cv2.rectangle(imagenrecortadanormal, (a, b), (a + c, b + d), (200, 105, 0), 2) Codigo usado para comprobar si detectaba los numeros correctamente

                trozosMatricula.append([i[1], imagen_GrisRecortada, contornos, [x,y,w,h]]) #Imagen original, imagen recortada (Matricula),
                # contornos de los numeros y [x,y,w,h] -> posiciones del recorte de la matricula

                #cv2.imshow("prueba", imc) Codigo usado para comprobar si detectaba los numeros correctamente
                #cv2.waitKey() Codigo usado para comprobar si detectaba los numeros correctamente

        else:
            contorno = []
            imagenPP = frontal_cascade.detectMultiScale(im, 1.1, 11)  # 1parametro imagen

            for (x, y, w, h) in imagenPP:
                data_x = np.array([])
                data_y = np.array([])
                recortar = im[y: (y + h), x: (x + w)]
                recortarNormal = imc[y: (y + h), x: (x + w)]
                th3 = cv2.adaptiveThreshold(recortar,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,15)

                cv2.imshow("pruebath", th3)
                cv2.waitKey()
                # https://www.programcreek.com/python/example/89437/cv2.boundingRect
                contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # https://stackoverflow.com/questions/46971769/how-to-extract-only-characters-from-image
                sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

                # contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                contornos =[]
                for cor in sorted_ctrs:
                    # get the bounding rect
                    a, b, c, d = cv2.boundingRect(cor)
                    anchoMin = 4
                    anchoMax = 7
                    if d > c and c > 3 and d > 10:
                        data_x = np.append(data_x, float(a))
                        data_y = np.append(data_y, float(b))
                        contorno.append([a, b, c, d])
                        cv2.rectangle(recortarNormal, (a, b), (a + c, b + d), (200, 105, 0), 2)
                        contornos.append(cor)
                print(len(contorno))
                cv2.imshow("prueba", recortarNormal)
                cv2.waitKey()
            if data_y.shape[0] != 0:
                data_x = data_x.reshape((data_x.shape[0], 1))
                ransac = linear_model.RANSACRegressor()
                ransac.fit(data_x, data_y)

                plt.scatter(data_x, data_y, color='black')
                x0 = data_x[0]
                x1 = data_x[data_x.shape[0] - 1]
                y0 = ransac.estimator_.intercept_
                y1 = x1 * ransac.estimator_.coef_ + ransac.estimator_.intercept_
                plt.plot([x0, x1], [y0, y1], color='blue', linewidth=2)
                plt.show()
                m = ransac.estimator_.coef_
                n = ransac.estimator_.intercept_
                # print(str(ransac.estimator_.coef_) + " " + str(ransac.estimator_.intercept_))
                for i in contorno:
                    if (i[0] * m) + n + 2 > i[1] and (i[0] * m) + n - 2 < i[1]:
                        print("---" + str((i[0] * m) + n) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]))

                        cv2.rectangle(recortarNormal, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0, 105, 0), 2)

            cv2.imshow("prueba", imc)
            cv2.waitKey()
    return trozosMatricula





def cargar_imagen(date_image):
    for img in listdir("./testing_full_system"):
        # 1.1 La carga debería realizarse en niveles de gris. Esto se consigue poniendo a 0 el segundo
        # parámetro del comando cv2.imread.
        img_grey = cv2.imread("./testing_full_system/" + img, 0)
        img_color = cv2.imread("./testing_full_system/" + img)
        date_image.append([img_grey, img_color])


def main():
    cargar_imagen(imagenes)
    informacion = comprobarmatricula(imagenes)


if __name__ == "__main__":
    main()