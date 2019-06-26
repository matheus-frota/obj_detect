# coding: utf-8

from sklearn.svm import SVC
import cv2
import numpy as np

winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)

# classificador notebook
#clf_note = cv2.CascadeClassifier('/home/matheus/github/diversos/obj_detect/haarcasdade/cascade_notebook.xml')

def hoggify(x,z):
    # Criando array vazio
    data=[]
    label = []
    # Laço para percorrer cada imagem do diretório
    for j in range(0,3):
        for i in range(1,int(z)):
            # Imagem do diretório
            image = cv2.imread("../images/"+str(j)+"/"+str(i)+".jpg", 0)
            #print(image.shape)
            # Se o diretorio for das imagens positivas a dimensão recebe 20
            if x == "positiva":
                dim = 20
            # Caso for negativa a dimensão recebe 100
            elif x == "negativa":
                dim = 100
            # Fazendo o redimensionamento da imagem
            img = cv2.resize(image, (dim,dim), interpolation = cv2.INTER_AREA)
            #print(img.shape)
            # Calculando os descritores HOG
            img = hog.compute(img)
            #print(img.shape)
            # Removendo entradas unidimensionais
            img = np.squeeze(img)
            #print(img.shape)
            # Adicionando img tratada
            data.append(img)
            label.append(j)
            #print(img.shape)
    return data,label


def svmClassify(features,labels):
    # Treinando o modelo SVM com kernel polinomial
    clf=SVC(C=10000,kernel="poly",gamma=0.000001)

    clf.fit(features,labels)

    return clf

def list_to_matrix(lst):
    # Transforma lista em matriz numpy
    return np.stack(lst)

def train():
    # Variável de controle
    x = 'positiva'
    y = 11
    # função para retornar lista de imagens com os descritores HOG
    lst,labels = hoggify(x,y)
    #print(label)
    # Converter lista python em array numpy
    labels = np.array(labels)
    # Convertendo a lista HOG em matriz numpy
    data = list_to_matrix(lst)
    # Fazendo o treinamento do modelo
    clf = svmClassify(data,labels)
    # Retornando o modelo treinado
    return clf

def webcam():
    cap = cv2.VideoCapture(0)
    dim = 20
    clf = train()
    while True:
        # Capturar a imagem
        ret, frame = cap.read()

        # Mostrar a imagem da webcam
        #cv2.imshow('Webcam', frame)

        # Converter a escala da imagem para cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Redimencionamento da imagem
        gray = cv2.resize(gray, (dim, dim), interpolation = cv2.INTER_AREA)
        # Criando descritores HOG
        features = hog.compute(gray)
        # Transpondo vetor de descritores
        features = features.T
        """
        # hog svm
        hog.setSVMDetector(features)
        #Tentativa para imprimir o retangulo - Sem Sucesso
        (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),padding=(8, 8))
        print(rect)
        """
        # Predição da imageqm na tela
        pred = clf.predict(features)
        # Caso a imagem for da classe 1 então:
        if pred == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Notebook',(200,200), font, 1,(0,255,255))
            # Imprime que o objeto na tela é da classe 1
            print("No vídeo tem um notebook!")
            #for (x,y,l,a) in deteccoes:
                #cv2.rectangle(frame, (x,y), (x + l, y + a), (0,255,0),2)
        elif pred == 2:
            print("No vídeo tem um celular!")
        elif pred == 3:
            print("pincel")
        # Mostrando saida na imagem de vídeo
        cv2.imshow("Video",frame)
        # Caso pressione a tecla q é desligado o teste com um delay de 25ms.
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release() # Release the camera resource
    cv2.destroyAllWindows() # Close the image window


if __name__ == '__main__':
    webcam()
