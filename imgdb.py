import os
import sys
import re

def readBase(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    p = re.compile('([a-z0-9]+)_([a-z0-9]+)_([a-z0-9]+)', re.IGNORECASE)
    firms = dict()

    for fl in onlyfiles:
        file = fl[:fl.rfind('.')]
        m = p.search(file)

        if m is None:
            continue

        g = m.groups()
        if len(g) == 0:
            continue
        
        if g[0] in firms:
            models = firms[g[0]]
            if g[1] in models:
                firms[g[0]][g[1]].append(fl)
            else:
                models[g[1]] = [fl]
        else:
            firms[g[0]] = {g[1]: [fl]}

    return firms

def createListOfModels(firms):
    listaModeli = []

    for firm in firms:
        for model in firms[firm]:
            listaModeli.append([firm,model])

    return listaModeli

def getFirmModelFromName(jpgfile):
    p = re.compile('([a-z0-9]+)_([a-z0-9]+)_([a-z0-9]+)', re.IGNORECASE)

    file = jpgfile[:jpgfile.rfind('.')]
    m = p.search(file)

    if m is None:
        return 0

    g = m.groups()
    if len(g) == 0:
        return 0

    return [g[0], g[1]]
        
if __name__ == "__main__":
    print("Testy:")

    firms = readBase("Images/")
    models = createListOfModels(firms)
    print(models)
