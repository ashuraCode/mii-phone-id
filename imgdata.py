import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm

color = ('b','g','r')


def saveBGR(dataBGR, filename):
    for col in color:
        plt.plot(dataBGR[col],color = col)
        plt.xlim([0,256])
    plt.savefig(filename+'.jpg')
    plt.close('all')

def histBGR(img):
    out = dict()
    for i,col in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        hist = cv2.normalize(hist, None)
        out[col] = hist 
    return out

def backProjectBGR(img): # dystr. empiryczna
    out = dict()
    levels = np.linspace(0, 255, 256)
    for i,col in enumerate(color):
        ecdf = ECDF(np.concatenate(img[:,:,i]))
        dist = ecdf(levels)
        out[col] = np.ndarray(shape=(256,1), buffer=dist) 
    return out

def saveCosineTransformBGR(img, fileprefix):
    for i,col in enumerate(color):
        imf = np.float32(img[:,:,i])/255.0  
        transformed = np.uint8(cv2.dct(imf)*255.0)  
        cv2.imwrite(fileprefix+"_"+col+".jpg", transformed)

def saveDFTransformBGR(img, fileprefix):
    for i,col in enumerate(color):
        dft = cv2.dft(np.float32(img[:,:,i]),flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        cv2.imwrite(fileprefix+"_"+col+".jpg", magnitude_spectrum)
        

if __name__ == "__main__":
    print("Testy:")
    
    img = cv2.imread("Images/Xiaomi_RedmiNote4_1.jpg", cv2.IMREAD_COLOR)
    histBGR = histBGR(img)
    saveBGR(histBGR, "myPhHist")
    bpBGR = backProjectBGR(img)
    saveBGR(bpBGR, "myPhEmpirDist")
    saveCosineTransformBGR(img, "myPhCosTr")
    saveDFTransformBGR(img, "myPhDiscFourTr")

    sys.exit(0)
