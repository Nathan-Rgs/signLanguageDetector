import os
import cv2
import string
import numpy as np

DATASET_SIZE = 300  # Número de sequências que queremos adicionar por letra
SEQUENCE_LENGTH = 10  # Número de frames por sequência

DATA_DIR = './data/sequences'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Lista de letras do alfabeto A-Z e 'None'
alphabet = list(string.ascii_uppercase) + ['None']

# Função para obter o maior número de arquivo na pasta ou começar em 0 se não houver arquivos
def get_starting_counter(directory):
    if not os.path.exists(directory):
        return 0
    files = os.listdir(directory)
    numbers = [int(file.split('.')[0]) for file in files if file.endswith('.npy')]
    if numbers:
        return max(numbers) + 1
    else:
        return 0

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# Loop para coletar dados de cada letra do alfabeto
for j, letter in enumerate(alphabet):
    class_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Coletando dados para a letra {letter}')

    # Mostrar na tela qual letra está sendo coletada e aguardar confirmação do usuário
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro: Não foi possível capturar a imagem da câmera.")
            break

        # Mostra a letra atual e a instrução para pressionar "q"
        cv2.putText(frame, f'Letra: {letter} - Pressione "Q" para iniciar', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Aguarda até que o usuário pressione "q" para iniciar a captura
        if cv2.waitKey(25) == ord('q'):
            break

    counter = get_starting_counter(class_dir)
    sequencias_restantes = DATASET_SIZE

    while sequencias_restantes > 0:
        sequence = []
        for _ in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível capturar a imagem da câmera.")
                break
            sequence.append(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        if len(sequence) == SEQUENCE_LENGTH:
            # Salvar a sequência como um arquivo numpy
            sequence = np.array(sequence)
            np.save(os.path.join(class_dir, f'{counter}.npy'), sequence)
            counter += 1
            sequencias_restantes -= 1
            print(f'Sequência {counter} salva para a letra {letter}')

    print(f'Capturadas mais {DATASET_SIZE} sequências para a letra {letter}.')

cap.release()
cv2.destroyAllWindows()
