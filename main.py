import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import _thread

arquivo = open('modelo_01_expressoes.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

modelo = model_from_json(estrutura_rede)
modelo.load_weights('model_weights.h5')

cascade_faces = 'haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(cascade_faces)
expressoes = ['Raiva', 'Nojo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']
cap = cv2.VideoCapture(0)
expressao = expressoes[5]
identificando = False


def identificar_expressao(modelo, face, cinza):
    global expressao
    global identificando
    identificando = True
    # Extraindo o ponto de interesse
    roi = cinza[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
    # Redimensionando a imagem para o tamanho necessário no treinamento do modelo
    roi = cv2.resize(roi, (48, 48))
    # Passando os valores RGB da imagem para a escala de números decimais de 0 a 1
    roi = roi / 255
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    # Utilizando o modelo para detectar a emoção
    preds = modelo.predict(roi)[0]
    modelo.predict(roi)
    expressao = expressoes[preds.argmax() - 1]
    identificando = False


while True:
    faces_qtd = 0
    _, frame = cap.read()  # OBTENDO A CAPTURA DA WEBCAM
    original = frame.copy()
    faces = face_detection.detectMultiScale(original, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    for face in faces:
        if not identificando:
            _thread.start_new_thread(identificar_expressao, (modelo, face, cinza))
        # Escrevendo a emoção resultante na imagem original
        cv2.putText(original, expressao, (face[0], face[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255), 2, cv2.LINE_AA)
        # Desenhando o retângulo ao redor do rosto que está sendo analisado na imagem original
        cv2.rectangle(original, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 0, 255), 2)

    cv2.imshow("Frame", original)
    key = cv2.waitKey(1)
    if key == 27:
        break