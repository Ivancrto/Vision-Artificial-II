import cv2
from os import listdir
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import math
import sys

imagenes_full_sytstem = []
imagenes_testing = []
imagenes_training = []
infoMatricula = []
matricula_cascade = cv2.CascadeClassifier('./haar/matriculas.xml')
frontal_cascade = cv2.CascadeClassifier('./haar/coches.xml')


def localizar_matricula(date_image):
    trozosMatricula = [];
    for i in date_image:
        im = i[0]
        imc = i[1]

        imagenP = matricula_cascade.detectMultiScale(im, 1.1, 5)  # 1parametro imagen
        imagenPP = frontal_cascade.detectMultiScale(im, 1.1, 5)  # 1parametro imagen

        # Por cada matricula detectada, dibujamos un rectangulo
        if len(imagenP) != 0:

            for (x, y, w, h) in imagenP:
                # Recorto la matricula
                imagen_GrisRecortada = im[y: (y + h), x: (x + w)]
                imagenrecortadanormal = imc[y: (y + h), x: (x + w)]
                # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
                ret, th3 = cv2.threshold(imagen_GrisRecortada, 0, 255, cv2.THRESH_OTSU)
                # https://www.programcreek.com/python/example/89437/cv2.boundingRect
                contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contornos = []
                for cor in contours:
                    # Sacamos coordenadas de contorno
                    a, b, c, d = cv2.boundingRect(cor)

                    if d > 1.1 * c and c > 5 and d > 12:
                        contornos.append(cor)
                        # cv2.rectangle(imagenrecortadanormal, (a, b), (a + c, b + d), (200, 105, 0), 2) #Codigo usado para comprobar si detectaba los numeros correctamente
                trozosMatricula.append([th3, contornos, (a, b, c, d), i[2], imc])  # La imagen gris cortada, sus contornos, sus coordenadas, su titulo y la imagen normal
                # cv2.imshow(i[2], imc) #Codigo usado para comprobar si detectaba los numeros correctamente
                # cv2.waitKey() #Codigo usado para comprobar si detectaba los numeros correctamente
        '''else:
            ret, th3 = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contornos = []
            for cor in contours:
                # Sacamos coordenadas de contorno
                a, b, c, d = cv2.boundingRect(cor)
                if d > 1.1*c and c > 5 and d > 12:
                    contornos.append(cor)
            trozosMatricula.append([im, contornos, None, i[2], imc]) # La imagen gris cortada, sus contornos, sus coordenadas, su titulo y la imagen normal'''

    return trozosMatricula

"""elif len(imagenPP) !=0:
            contorno = []
            data_x = np.array([])
            data_y = np.array([])
            contornos =[]
            for (x, y, w, h) in imagenPP:
                recortar = im[y: (y + h), x: (x + w)]
                recortarNormal = imc[y: (y + h), x: (x + w)]
                th3 = cv2.adaptiveThreshold(recortar,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,15)
                # https://www.programcreek.com/python/example/89437/cv2.boundingRect
                contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # https://stackoverflow.com/questions/46971769/how-to-extract-only-characters-from-image
                # contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                for cor in contours:
                    # get the bounding rect
                    a, b, c, d = cv2.boundingRect(cor)
                    area = cv2.contourArea(cor)
                    areaMaxima = 10*10
                    if d > 1.1*c and c > 3 and d > 10 and area<areaMaxima:
                        data_x = np.append(data_x, float(a))
                        data_y = np.append(data_y, float(b))
                        contorno.append([a, b, c, d])
                        contornos.append(cor)
            if len(contornos)>5: 
                trozosMatricula.append([recortar, contornos, (a,b,c,d), i[2], imc]) # La imagen gris cortada, sus contornos, sus coordenadas, su titulo y la imagen normal
      
            if data_y.shape[0] != 0:
                data_x = data_x.reshape((data_x.shape[0], 1))
                ransac = linear_model.RANSACRegressor()
                ransac.fit(data_x, data_y)

                plt.scatter(data_x, data_y, color='black')
                x0 = data_x[0]
                x1 = data_x[data_x.shape[0] - 1]
                y0 = ransac.estimator_.intercept_
                y1 = x1 * ransac.estimator_.coef_ + ransac.estimator_.intercept_
                #plt.plot([x0, x1], [y0, y1], color='blue', linewidth=2)
                #plt.show()
                m = ransac.estimator_.coef_
                n = ransac.estimator_.intercept_
                # print(str(ransac.estimator_.coef_) + " " + str(ransac.estimator_.intercept_))
                contornosNumero = []
                for i in contorno:
                    if (i[0] * m) + n + 2 > i[1] and (i[0] * m) + n - 2 < i[1]:
                        contornosNumero.append(i)
                trozosMatricula.append([im, contornosNumero, None, i[2], imc]) # La imagen gris cortada, sus contornos, sus coordenadas, su titulo y la imagen normal
           '''                 
"""

# El metodo para crear las clases
def crearEtiquetas():
    clases = np.zeros((8000), np.uint8)
    i=0
    while i<8000:
        if (0 <= i < 250):
            clases[i] = 0
        elif (250 <= i < 500):
            clases[i] = 1
        elif (500 <= i < 750):
            clases[i] = 2
        elif (750 <= i < 1000):
            clases[i] = 3
        elif (1000 <= i < 1250):
            clases[i] = 4
        elif (1250 <= i < 1500):
            clases[i] = 5
        elif (1500 <= i < 1750):
            clases[i] = 6
        elif (1750 <= i < 2000):
            clases[i] = 7
        elif (2000 <= i < 2250):
            clases[i] = 8
        elif (2250 <= i < 2500):
            clases[i] = 9
        elif (2500 <= i < 2750):
            clases[i] = 10
        elif (2750 <= i < 3000):
            clases[i] = 11
        elif (3000 <= i < 3250):
            clases[i] = 12
        elif (3250 <= i < 3500):
            clases[i] = 13
        elif (3500 <= i < 3750):
            clases[i] = 14
        elif (3750 <= i < 4000):
            clases[i] = 15
        elif (4000 <= i < 4250):
            clases[i] = 16
        elif (4250 <= i < 4500):
            clases[i] = 17
        elif (4500 <= i < 4750):
            clases[i] = 18
        elif (4750 <= i < 5000):
            clases[i] = 19
        elif (5000 <= i < 5250):
            clases[i] = 20
        elif (5250 <= i < 5500):
            clases[i] = 21
        elif (5500 <= i < 5750):
            clases[i] = 22
        elif (5750 <= i < 6000):
            clases[i] = 23
        elif (6000 <= i < 6250):
            clases[i] = 24
        elif (6250 <= i < 6500):
            clases[i] = 25
        elif (6500 <= i < 6750):
            clases[i] = 26
        elif (6750 <= i < 7000):
            clases[i] = 27
        elif (7000 <= i < 7250):
            clases[i] = 28
        elif (7250 <= i < 7500):
            clases[i] = 29
        elif (7500 <= i < 7750):
            clases[i] = 30
        else:
            clases[i] = 31
        i = i + 1
    return clases

# El metodo para traducir los valores predecidos a las letras y a los numeros y construir la matricula ordenada
def construirMatricula(digitos):

    matricula = []
    digitosTotal = []
    digitos.sort(key=lambda digitos: digitos[1])
    for i in range(len(digitos)):
        if(i < len(digitos) - 1 and abs(digitos[i][2] - digitos[i+1][2]) < 6 and abs(digitos[i][1] - digitos[i+1][1]) > 8):
            digitosTotal.append((digitos[i][0], digitos[i][1], digitos[i][2], digitos[i][3], digitos[i][4]))
        if(i == len(digitos) - 1 and abs(digitos[i - 1][2] - digitos[i][2]) < 6 and abs(digitos[i][1] - digitos[i-1][1]) > 8):
            digitosTotal.append((digitos[i][0], digitos[i][1], digitos[i][2], digitos[i][3], digitos[i][4]))

    for d in digitosTotal:
        if (d[0] == 0):
            matricula.append("0")
        elif (d[0] == 1):
            matricula.append("1")
        elif (d[0] == 2):
            matricula.append("2")
        elif (d[0] == 3):
            matricula.append("3")
        elif (d[0] == 4):
            matricula.append("4")
        elif (d[0] == 5):
            matricula.append("5")
        elif (d[0] == 6):
            matricula.append("6")
        elif (d[0] == 7):
            matricula.append("7")
        elif (d[0] == 8):
            matricula.append("8")
        elif (d[0] == 9):
            matricula.append("9")
        elif (d[0] == 10):
            matricula.append("B")
        elif (d[0] == 11):
            matricula.append("C")
        elif (d[0] == 12):
            matricula.append("D")
        elif (d[0] == 13):
            matricula.append("ES")
        elif (d[0] == 14):
            matricula.append("F")
        elif (d[0] == 15):
            matricula.append("G")
        elif (d[0] == 16):
            matricula.append("H")
        elif (d[0] == 17):
            matricula.append("J")
        elif (d[0] == 18):
            matricula.append("K")
        elif (d[0] == 19):
            matricula.append("L")
        elif (d[0] == 20):
            matricula.append("M")
        elif (d[0] == 21):
            matricula.append("N")
        elif (d[0] == 22):
            matricula.append("P")
        elif (d[0] == 23):
            matricula.append("Q")
        elif (d[0] == 24):
            matricula.append("R")
        elif (d[0] == 25):
            matricula.append("S")
        elif (d[0] == 26):
            matricula.append("T")
        elif (d[0] == 27):
            matricula.append("V")
        elif (d[0] == 28):
            matricula.append("W")
        elif (d[0] == 29):
            matricula.append("X")
        elif (d[0] == 30):
            matricula.append("Y")
        else:
            matricula.append("Z")
    return matricula, digitosTotal

def obtenerCarcteristicasMatricula(img, coord):
    x = coord[0]
    y = coord[1]
    w = coord[2]
    h = coord[3]

    # Umbralizar los caracteres recortados de la matricula
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    # Recortar los caracteres
    caracter_recortado = th[y: (y + h), x: (x + w)]

    # Redimensionar los caracteres de la matricula
    caracter_redim = cv2.resize(caracter_recortado, (10, 10), interpolation=cv2.INTER_LINEAR)

    # El vector de caracteristicas
    caracteristicas = np.zeros((1, 100), np.uint8)

    k = 0
    # Agrupar las caracteristicas de los digitos de la matricula en una matriz
    for i in range(caracter_redim.shape[0]):
        for j in range(caracter_redim.shape[1]):
            caracteristicas[0][k] = caracter_redim[i][j]
            k = k + 1
    return caracteristicas

# Metodo para leer los caracteres de la matricula
def leer_matricula(infoMatricula, img_training):

    # Creamos el vector de caracteristica
    matrizC = []
    pos = 0
    contornoAnterior = []
    cont1 = -1
    # Cargar y umbralizar las imagenes de los caracteres de entrenamiento
    for caracter in img_training:
        cont1 = cont1 + 1
        # Umbralizar el caracter
        ret, th1 = cv2.threshold(caracter[0], 0, 255, cv2.THRESH_OTSU)

        # Sacar los contornos
        contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cor in contours:
            # Sacamos coordenadas de contorno
            x, y, w, h = cv2.boundingRect(cor)
            contornoAnterior.append([x,y])
            if h > 1.1 * w and w > 5 and h > 12:
                '''if (len(contornoAnterior) == 1):
                    # Recortar los caracteres
                    caracter_recortado = caracter[0][y: (y + h), x: (x + w)]

                    # Umbralizar los caracteres recortados
                    ret2, th2 = cv2.threshold(caracter_recortado, 0, 255, cv2.THRESH_OTSU)

                    # Redimensionar los caracteres
                    caracter_redim = cv2.resize(th2, (10, 10), interpolation=cv2.INTER_LINEAR)

                if (len(contornoAnterior) >= 2):
                    if (abs(contornoAnterior[pos][0] - contornoAnterior[pos - 1][0]) > 5 and abs(contornoAnterior[pos][1] - contornoAnterior[pos - 1][1]) < 6):'''

                # Umbralizar los caracteres
                ret2, th2 = cv2.threshold(caracter[0], 0, 255, cv2.THRESH_OTSU)

                # Recortar los caracteres
                caracter_recortado = th2[y: (y + h), x: (x + w)]
                #cv2.imshow("img", caracter_recortado) #Codigo usado para comprobar si detectaba los numeros correctamente
                #cv2.waitKey() #Codigo usado para comprobar si detectaba los numeros correctamente

                # Redimensionar los caracteres
                caracter_redim = cv2.resize(caracter_recortado, (10, 10), interpolation=cv2.INTER_LINEAR)

            pos = pos + 1
        # El vector de caracteristicas asociado al caracter de entrenamiento
        caracteristicas = np.zeros((1, 100), np.uint8)
        a = 0
        # Agrupar las caracteristicas en una matriz C
        if len(caracter_redim) != 0 and not(2500 <= cont1 < 2750 or 3750 <= cont1 < 4000 or 4750 <= cont1 < 5000 or 6250 <= cont1 < 6500 or 7750 <= cont1 < 8000):
            for i in range(caracter_redim.shape[0]):
                for j in range(caracter_redim.shape[1]):
                    caracteristicas[0][a] = caracter_redim[i][j]
                    a = a + 1
            matrizC.append(caracteristicas)

    # Matriz que tiene todas las caracteristicas que tiene cada clase (X)
    C = np.zeros((8000, 100))
    f = -1
    for car in matrizC:
        f = f + 1
        for c in range(car.shape[1]):
            C[f][c] = car[0][c]

    # Crear las clases (y)
    E = crearEtiquetas()

    # Crear objeto de LDA
    objectLDA = LDA()

    # La matriz de proyeccion LDA
    objectLDA.fit(C, E)

    dig = []
    cont = 0
    contornoAnterior = []
    pos = 0
    # Bucle para recortar los digitos de la matricula segun los contornos
    for info in infoMatricula:
        cont = cont + 1
        for contours in info[1]:

            x, y, w, h = cv2.boundingRect(contours)
            #contornoAnterior.append([x,y])

            if h > 1.1*w and w>5 and h>12:
                '''if(len(contornoAnterior)==1):
                    caracteristicas2 = obtenerCarcteristicasMatricula(info[0], (x, y, w, h))
                    # Recogemos prediccion LDA
                    prediccion = objectLDA.predict(caracteristicas2)
                    dig.append((prediccion[0], x, y, w, h))
                if(len(contornoAnterior)>=2):
                    if(abs(contornoAnterior[pos][0] - contornoAnterior[pos-1][0]) > 5 and abs(contornoAnterior[pos][1] - contornoAnterior[pos-1][1]) < 6):'''
                caracteristicas2 = obtenerCarcteristicasMatricula(info[0], (x,y,w,h))
                # Recogemos prediccion LDA
                prediccion = objectLDA.predict(caracteristicas2)
                dig.append((prediccion, x, y, w, h))
            #pos = pos + 1
        l = 0
        matriculaConstruida = construirMatricula(dig)
        for letra in matriculaConstruida[0]:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(info[4], letra, (matriculaConstruida[1][l][1] + 210, matriculaConstruida[1][l][2] + 200), font, 0.6, (255, 128, 0), 1, cv2.LINE_AA)
            l = l + 1
        cv2.imshow(info[3], info[4]) #Codigo usado para comprobar si detectaba los numeros correctamente
        cv2.waitKey() #Codigo usado para comprobar si detectaba los numeros correctamente
        print("Imagen ", cont, ":", info[3], matriculaConstruida[0])
        dig = []

def cargar_imagen(date_image, path):
    for img in listdir(path):
        # 1.1 La carga debería realizarse en niveles de gris. Esto se consigue poniendo a 0 el segundo
        # parámetro del comando cv2.imread.
        img_grey = cv2.imread(path + "/" + img, 0)
        img_color = cv2.imread(path + "/" + img)
        date_image.append([img_grey, img_color, img])


def main(fichero):
    cargar_imagen(imagenes_testing, fichero)
    #cargar_imagen(imagenes_testing, "./testing_full_system")
    # localizar_matricula(imagenes_full_sytstem)
    #cargar_imagen(imagenes_testing, "./testing_ocr")
    infoMatricula = localizar_matricula(imagenes_testing)
    cargar_imagen(imagenes_training, "./training_ocr")
    leer_matricula(infoMatricula, imagenes_training)


if sys.argv.__len__() == 2:
    main(sys.argv[1])
else:
    print("El numero de argumentos pasados es incorrecto")
