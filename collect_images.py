import os
import cv2
import string

DATA_DIR = './amostra_imagens'
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

alphabet = list(string.ascii_uppercase)  # Lista de letras do alfabeto A-Z
dataset_size = 100  # Número de imagens por letra


cap = cv2.VideoCapture(0)
if not cap.isOpened():
  print("Erro: Não foi possível acessar a câmera.")
  exit()

for j, letter in enumerate(alphabet):
  class_dir = os.path.join(DATA_DIR, letter)
  if not os.path.exists(class_dir):
      os.makedirs(class_dir)

  print(f'Coletando dados para a letra {letter}')

  while True:
    ret, frame = cap.read()

    if not ret:
      print("Erro: Não foi possível capturar a imagem da câmera.")
      break

    cv2.putText(frame, f'Letra: {letter} - Pressione "Q"', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) == ord('q'):
      break

  counter = 0
  while counter < dataset_size:
    ret, frame = cap.read()

    if not ret:
      print("Erro: Não foi possível capturar a imagem da câmera.")
      break

    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
    counter += 1

cap.release()
cv2.destroyAllWindows
