import os 
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Inicializando o módulo de detecção de mãos do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Utilitário para desenhar landmarks (não usado neste código)
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos de desenho para landmarks (não usado diretamente)

# Inicializando o detector de mãos, em modo de imagem estática, com confiança mínima de 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Definindo o diretório onde as imagens de exemplo estão armazenadas
DATA_DIR = './data/images'

# Inicializando listas para armazenar os dados (landmarks normalizados) e rótulos (labels)
data = []
labels = []

# Iterando por todas as subpastas do diretório (cada subpasta é uma classe/rótulo)
for dir_ in os.listdir(DATA_DIR):
    # Iterando por todas as imagens dentro de cada subpasta
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Lista auxiliar para armazenar as coordenadas de uma única imagem
        x_ = []  # Lista auxiliar para armazenar os valores de x dos landmarks
        y_ = []  # Lista auxiliar para armazenar os valores de y dos landmarks

        # Carregando a imagem usando OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convertendo a imagem de BGR (padrão do OpenCV) para RGB (padrão do MediaPipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processando a imagem com o detector de mãos do MediaPipe
        results = hands.process(img_rgb)

        # Se uma mão for detectada na imagem
        if results.multi_hand_landmarks:
            # Itera por todas as mãos detectadas (no caso de haver mais de uma)
            for hand_landmarks in results.multi_hand_landmarks:
                # Para cada ponto de referência (landmark), extrai as coordenadas x e y
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Coordenada x normalizada
                    y = hand_landmarks.landmark[i].y  # Coordenada y normalizada

                    # Adiciona os valores x e y às listas auxiliares
                    x_.append(x)
                    y_.append(y)

                # Normalizando as coordenadas: centralizando os landmarks ao subtrair os valores mínimos de x e y
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Normalização: subtraindo o valor mínimo para cada coordenada (alinhando as mãos)
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Adiciona os dados normalizados da imagem à lista principal de dados
            data.append(data_aux)
            # Adiciona o nome da subpasta (que corresponde ao rótulo/classe) à lista de rótulos
            labels.append(dir_)

# Abrindo um arquivo pickle para salvar os dados processados
f = open('./data/data.pickle', 'wb')
# Serializando e salvando os dados (landmarks normalizados e rótulos) no arquivo
pickle.dump({'data': data, 'labels': labels}, f)
# Fechando o arquivo após a escrita
f.close()
