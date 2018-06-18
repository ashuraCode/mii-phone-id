import os
import sys
import numpy as np
import cv2
import imgdb
import imgdata
import nnmpl
import pickle
from scipy import stats

OPIS = """recognizephone.py <mode> <database> <method> <images>
mode:
    create - uczenie na podstawie folderu podanego jako parametr images
    check - sprawdzenie wybranego zdjęcia podanego jako parametr images
database:
    nazwa pliku danych - binarny obiekt sieci neuronowej
method:
    histogram - używa histogramu do rozpoznania
    empirdist - używa dystrybuanty empirycznej
image:
    folder lub obraz w zależności od wybranego mode
"""

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Nie podano parametrów.")
        print(OPIS)
        sys.exit(-1)

    if sys.argv[1] == "create":
        print("Creating..")
        nndatabase = sys.argv[2]
        fproc = sys.argv[3]
        imgPrefix = sys.argv[4]

        if fproc != "histogram" and fproc != "empirdist":
            print("Nie poprawny parametr: " + fproc)
            print(OPIS)
            sys.exit(-1)

        if imgPrefix[-1] != "/":
            imgPrefix += "/"

        firms = imgdb.readBase(imgPrefix)

        if len(firms) == 0:
            print("Nie prawidłowy folder z obrazami.")
            print(OPIS)
            sys.exit(-1)

        models = imgdb.createListOfModels(firms)
        data = {}

        for i,(firm,model) in enumerate(models):
            for imgFile in firms[firm][model]:
                print(imgFile)
                print(str(i) + " " + firm + ":" + model)
                img = cv2.imread(imgPrefix+imgFile, cv2.IMREAD_COLOR)
                if fproc == "histogram":
                    dataBGR = imgdata.histBGR(img)
                elif fproc == "empirdist":
                    dataBGR = imgdata.backProjectBGR(img)
                vector = np.concatenate( ( dataBGR["b"], dataBGR["g"], dataBGR["r"]), axis=0).T
                data[imgFile] = vector
            
        with open(nndatabase, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('models', 'wb') as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif sys.argv[1] == "check":
        print("Checking..")
        nndatabase = sys.argv[2]
        fproc = sys.argv[3]
        img = cv2.imread(sys.argv[4], cv2.IMREAD_COLOR)

        if isinstance(img, np.ndarray) == 0:
            print("Nie prawidłowy obraz.")
            print(OPIS)
            sys.exit(-1)
        
        models = []
        data = {}
        with open('models', 'rb') as fp:
            models = pickle.load(fp)

        with open(nndatabase, 'rb') as fp:
            data = pickle.load(fp)

        if len(models) == 0:
            print("Baza modeli jest uszkodzona")
            sys.exit(-1)

        if fproc == "histogram":
            dataBGR = imgdata.histBGR(img)
        # elif fproc == "empirdist":
        #     dataBGR = imgdata.backProjectBGR(img)
            
            vector = np.concatenate( ( dataBGR["b"], dataBGR["g"], dataBGR["r"]), axis=0).T

        # print(vector[0,:])

        if fproc == "histogram":
            mse = []
            imgs = []
            for dta in data:
                imgs.append(dta)
                mse.append( ((vector - data[dta]) ** 2).mean(axis=1)[0] )

            print(mse)
            idx = np.argmin(mse)
            print(idx)
        elif fproc == "empirdist":
            ks_test = []
            imgs = []
            for dta in data:
                img2 = cv2.imread("images/"+dta, cv2.IMREAD_COLOR)
                imgs.append(dta)
                s = np.shape(img)
                s2 = np.shape(img2)
                img_1 = np.reshape(img, (s[0]*s[1]*s[2],1,1))
                img2_1 = np.reshape(img2, (s2[0]*s2[1]*s2[2],1,1))
                print(np.shape(img_1))
                print(np.shape(img2_1))
                ks_test.append(stats.ks_2samp(img_1[:,0,0], img2_1[:,0,0])[1])
                
            print(ks_test)
            idx = np.argmax(ks_test)
            print(idx)

        phone = imgdb.getFirmModelFromName(imgs[idx])

        if phone != 0:
            print("Wykryto obraz z telefonu: " + phone[0] + ", model: " + phone[1])
    