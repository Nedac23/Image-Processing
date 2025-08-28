# libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import os
from math import log10, sqrt 
import pywt

#https://github.com/behnamasadi/PythonTutorial/blob/master/signal_system/white_noise_gaussian_noise.ipynb
def AddGaussianNoise():
    # original image
    imagepath = 'Filepath'
    f = cv2.imread(imagepath, 0)
    f = f/255 
    # create gaussian noise
    x, y = f.shape
    mean = 0
    var = .01
    sigma = np.sqrt(var)
    n = np.random.normal(loc=mean, 
                        scale=sigma, 
                        size=(x,y))
    # add a gaussian noise
    g = f + n
    g = (g*255).round().astype(np.uint8)
    plt.hist(g.flat)
    plt.xlim([0,255]); plt.ylim([0,60000])
    plt.xlabel('pixel value'); plt.ylabel('frequency')
    plt.show()
    os.chdir("path to save to")  
    filename = 'Filename'
    #saves file 
    cv2.imwrite(filename,g)
    return g


def AddSaltandPepper(opt):
   
    imagepath = 'filepath'
    img = cv2.imread(imagepath, 0)
    img = img/255
    # blank image
    x,y = img.shape
    g = np.zeros((x,y), dtype=np.float32)
    # salt and pepper amount
    if(opt == 0):
        pepper = .1
        salt = 1
    if(opt == 1):
        pepper = 0
        salt = .9
    if(opt == 2):
        pepper = 0.1
        salt = 0.9
    # create salt and peper noise image    
    for i in range(x):
        for j in range(y):
            rdn = np.random.random()
            if rdn < pepper:
                g[i][j] = 0
            elif rdn > salt:
                g[i][j] = 1
            else:
                g[i][j] = img[i][j]
    #used for final image formating
    g = (g*255).round().astype(np.uint8)
    cv2.imshow('image with noise', g)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.hist(g.flat)
    plt.xlim([0,255]); plt.ylim([0,60000])
    plt.xlabel('pixel value'); plt.ylabel('frequency')
    plt.show()
    return g
    
#https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr, mse 
  

def Wavelet(img):
    #pywt.wavedec2(img, 'db1', mode='symmetric', level=3)
    #https://pywavelets.readthedocs.io/en/latest/ref/dwt-coefficient-handling.html
    
    #cam = pywt.data.camera()
    coeffs = pywt.wavedecn(img, wavelet='db2', level=3)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.abs(arr), cmap='gray')
    plt.title("Flattened Wavelet Coefficients (3-Level, db2)")
    plt.axis('off')
    plt.show()
    os.chdir('Path to save to')  
   # filename ='filename'
    g = arr
    #cv2.imwrite(filename,g)

def main():
    #ArithmaticMean()
   
   
      
    #os.chdir('file to save to')  
    #filename ='filename'
   # img = AddGaussianNoise()





   
   # os.chdir('path to save to')  
   # filename = 'filename'
   # img = AddSaltandPepper(1)
   # cv2.imwrite(filename,img)
   





    
    imagepath = 'original image path'
    imagepath2 = 'new image path'
    img = cv2.imread(imagepath, 0)
    img2 = cv2.imread(imagepath2, 0)
    
    value, mse = PSNR(img, img2) 
    print(f"PSNR value is {value} dB") 
    print(f"MSE value is {mse} dB") 

    #Wavelet(img)
   
if __name__=="__main__":
    main()