import os
import sys
import numpy as np
import cv2
import imgdb
import imgdata
import nnmpl
import pickle

OPIS = """recognizephone.py <mode> <database> <method> <images>
mode:
    learn - uczenie na podstawie folderu podanego jako parametr images
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

    if sys.argv[1] == "learn":
        print("Learning..")
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

        Ymod = np.eye(len(models))
        X = np.ndarray(shape=(0,3*256))
        y = np.ndarray(shape=(0,len(models)))

        for i,(firm,model) in enumerate(models):
        # for firm,model in models:
            for imgFile in firms[firm][model]:
                print(imgFile)
                print(str(i) + " " + firm + ":" + model)
                img = cv2.imread(imgPrefix+imgFile, cv2.IMREAD_COLOR)
                if fproc == "histogram":
                    dataBGR = imgdata.histBGR(img)
                elif fproc == "empirdist":
                    dataBGR = imgdata.backProjectBGR(img)
                vector = np.concatenate( ( dataBGR["b"], dataBGR["g"], dataBGR["r"]), axis=0).T
                X = np.concatenate( (X, vector), axis=0)
                y = np.concatenate( (y,[Ymod[i]]), axis=0)

        nnmpl.learn(X.astype(float), y.astype(float), nndatabase)
            
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
        with open('models', 'rb') as fp:
            models = pickle.load(fp)

        if len(models) == 0:
            print("Baza modeli jest uszkodzona")
            sys.exit(-1)

        if fproc == "histogram":
            dataBGR = imgdata.histBGR(img)
        elif fproc == "empirdist":
            dataBGR = imgdata.backProjectBGR(img)
            
        vector = np.concatenate( ( dataBGR["b"], dataBGR["g"], dataBGR["r"]), axis=0).T

        klasyfikacja = nnmpl.classify(vector.astype(float), nndatabase)
        idx = np.argmax(klasyfikacja)

        print("Wektor klasyfikacji: ", klasyfikacja, "\n")
        print("Wykryto obraz z telefonu: " + models[idx][0] + ", model: " + models[idx][1])
    