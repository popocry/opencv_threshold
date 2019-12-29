import cv2;  #opencv
import matplotlib.pyplot as plt;
import numpy as np 
from PIL import Image, ImageDraw,ImageEnhance   

def threshold(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#轉灰階

    ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)#固定閥值二值化
    ret,Otsu_binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)#Otsu’s二值化(雙峰)
    ret,Triangle_binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)#TRIANGLE(單峰)
    Mean_binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,25,25)#自適應閥值二值化(平均) 
    Gaussian_binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,25)#自適應閥值二值化(高斯) 
    
    #BGR2RBG
    gray=BGR2RBG(gray)
    binary=BGR2RBG(binary)
    Otsu_binary=BGR2RBG(Otsu_binary)
    Triangle_binary=BGR2RBG(Triangle_binary)
    Mean_binary=BGR2RBG(Mean_binary)
    Gaussian_binary=BGR2RBG(Gaussian_binary)


    plt.subplot(2, 3, 1) 
    plt.imshow(gray)
    plt.title("gray")

    plt.subplot(2, 3, 2) 
    plt.imshow(binary)
    plt.title("binary")

    plt.subplot(2, 3, 3) 
    plt.imshow(Otsu_binary)
    plt.title("Otsu_binary")

    plt.subplot(2, 3, 4) 
    plt.imshow(Triangle_binary)
    plt.title("Triangle_binary")

    plt.subplot(2, 3, 5) 
    plt.imshow(Mean_binary)
    plt.title("Mean_binary")

    plt.subplot(2, 3, 6) 
    plt.imshow(Gaussian_binary)
    plt.title("Gaussian_binary")

    plt.show()

def BGR2RBG(img):
   img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))   
   return img

if __name__ == '__main__':
    imgpath=r"C:\Users\po_po\Desktop\hand\hand\image\1.jpg"
    img=cv2.imread(imgpath)
    threshold(img)