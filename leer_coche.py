# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:46:54 2020

@author: Ivanxrto
"""

import numpy as np
import cv2 
from os import listdir

imagenes = []

matricula_cascade = cv2.CascadeClassifier('./haar/matriculas.xml')
frontal_cascade = cv2.CascadeClassifier('./haar/coches.xml')
def comprobarmatricula(date_image):
    for i in date_image:
        im = i[0]
        imc =i[1]
        
        imagenP = matricula_cascade.detectMultiScale (im, 1.1, 11) # 1parametro imagen
        if imagenP is not():
            # Por cada matricula detectada, dibujamos un rectangulo
            for (x, y, w, h) in imagenP:
                # Recorto la matricula
                imagenrecortadagris =im[y: (y + h), x: (x + w)]
                imagenrecortadanormal = imc[y: (y + h), x: (x + w)]
                #https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
                
                ret, th3 = cv2.threshold(imagenrecortadagris, 0, 255, cv2.THRESH_OTSU)
                #https://www.programcreek.com/python/example/89437/cv2.boundingRect
                contours, hierarchy = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                #https://stackoverflow.com/questions/46971769/how-to-extract-only-characters-from-image
                sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
                #contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                for cor in sorted_ctrs:
                    # get the bounding rect
                    a, b, c, d = cv2.boundingRect(cor)
                    if d>c and c>5 and d>10:
                        print(str(a) + " " + str(b) + " " + str(c)+ " " +str(d))        
                        cv2.rectangle(imagenrecortadanormal, (a, b), (a+c, b+d), (200, 105, 0), 2)
                    
                cv2.imshow("prueba",imc)
                cv2.waitKey()
                print("--------------------------")
        else:
        

            th3 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                        cv2.cv2.THRESH_BINARY_INV,19,2)
            #cv2.THRESH_BINARY   o cv2.THRESH_BINARY_INV
            cv2.imshow("pruebath",th3)
            cv2.waitKey()
            #https://www.programcreek.com/python/example/89437/cv2.boundingRect
            contours, hierarchy = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            #https://stackoverflow.com/questions/46971769/how-to-extract-only-characters-from-image
            sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
            
            #contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for cor in sorted_ctrs:
                # get the bounding rect
                a, b, c, d = cv2.boundingRect(cor)
                if d>c and c>5 and d>5 and c<40 and d<40:
                    print(str(a) + " " + str(b) + " " + str(c)+ " " +str(d))
                    cv2.rectangle(imc, (a, b), (a+c, b+d), (200, 105, 0), 2)
                    
            cv2.imshow("prueba",imc)
            cv2.waitKey()
            

def cargar_imagen(date_image):
    for img in listdir("./testing_full_system"):
         #1.1 La carga debería realizarse en niveles de gris. Esto se consigue poniendo a 0 el segundo
             #parámetro del comando cv2.imread. 
         img_grey = cv2.imread("./testing_full_system/" + img, 0)
         img_color = cv2.imread("./testing_full_system/" + img)
         date_image.append([img_grey, img_color])
    
def main():
    cargar_imagen(imagenes)
    comprobarmatricula(imagenes)
   
    
    
if __name__ == "__main__":
    main()