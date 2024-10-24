import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import random
import time

# Carregar o modelo LSTM treinado
model = load_model('./data/lstm_model.h5')

# Carregar o Label Encoder
with open('./data/label_encoder.pickle', 'rb') as f:
    le = pickle.load(f)

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Inicializar o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Definir o tamanho da sequência
SEQUENCE_LENGTH = 10
sequence = deque(maxlen=SEQUENCE_LENGTH)

# Função para normalizar landmarks de uma única mão
def normalize_landmarks(hand_landmarks):
    x = [lm.x for lm in hand_landmarks.landmark]
    y = [lm.y for lm in hand_landmarks.landmark]
    
    # Centralizar os landmarks
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x = [coord - x_mean for coord in x]
    y = [coord - y_mean for coord in y]
    
    # Escalonar baseado no tamanho da mão (distância entre landmarks 0 e 9)
    size = np.linalg.norm(np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y]) - 
                          np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y]))
    if size == 0:
        size = 1e-6  # Evitar divisão por zero
    x = [coord / size for coord in x]
    y = [coord / size for coord in y]
    
    # Combinar x e y
    normalized = []
    for xi, yi in zip(x, y):
        normalized.append(xi)
        normalized.append(yi)
    
    return normalized

# Lista de letras para o jogo
GAME_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Função principal
def main(game_mode=False, difficulty='easy'):
    score = 0
    current_letter = ""
    letter_start_time = 0
    letter_display_time = 5  # Tempo em segundos para o usuário sinalizar a letra
    letters_guessed = 0
    total_letters = 10  # Número de letras a serem sinalizadas no jogo

    if game_mode:
        # Embaralhar as letras para o jogo
        game_letters = random.sample(GAME_LETTERS, total_letters)
    else:
        game_letters = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a câmera.")
            break

        H, W, _ = frame.shape

        # Converter a imagem de BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if game_mode and letters_guessed < total_letters:
            # Mostrar a letra atual que o usuário deve sinalizar
            if not current_letter:
                current_letter = game_letters[letters_guessed]
                letter_start_time = time.time()
                print(f"Nova letra: {current_letter}")

            # Exibir a letra na tela
            cv2.putText(frame, f"Letra: {current_letter}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

            # Verificar o tempo decorrido
            elapsed_time = time.time() - letter_start_time
            cv2.putText(frame, f"Tempo restante: {int(letter_display_time - elapsed_time)}s", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            if elapsed_time > letter_display_time:
                # Tempo esgotado para a letra atual
                print(f"Tempo esgotado para a letra: {current_letter}")
                current_letter = ""
                letters_guessed += 1
                continue

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Normalizar landmarks
            normalized = normalize_landmarks(hand_landmarks)
            sequence.append(normalized)

            if len(sequence) == SEQUENCE_LENGTH:
                input_seq = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 42)
                prediction = model.predict(input_seq)
                predicted_class = le.inverse_transform([np.argmax(prediction)])

                # Exibir a letra prevista na tela
                cv2.putText(frame, f'Letra: {predicted_class[0]}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

                if game_mode and letters_guessed < total_letters:
                    if predicted_class[0] == current_letter:
                        score += 1
                        letters_guessed += 1
                        print(f"Correto! Letra: {current_letter}")
                        current_letter = ""
                        sequence.clear()
                        time.sleep(1)  # Pausa breve antes da próxima letra
                    else:
                        print(f"Incorreto. Letra esperada: {current_letter}, Detected: {predicted_class[0]}")

        else:
            # Se não houver mão detectada, limpar a sequência
            sequence.clear()

        if game_mode and letters_guessed >= total_letters:
            # Jogo concluído
            cv2.putText(frame, "Jogo Concluído!", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Pontuação: {score}/{total_letters}", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
            print(f"Jogo Concluído! Pontuação: {score}/{total_letters}")
            # Opcional: Resetar o jogo ou encerrar
            # Aqui, vamos encerrar após mostrar a pontuação
            cv2.imshow('frame', frame)
            cv2.waitKey(5000)  # Espera 5 segundos antes de fechar
            break

        # Desenhar um retângulo ao redor da mão
        if results.multi_hand_landmarks:
            x_coords = [lm.x * W for lm in hand_landmarks.landmark]
            y_coords = [lm.y * H for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_coords)) - 10, int(min(y_coords)) - 10
            x2, y2 = int(max(x_coords)) + 10, int(max(y_coords)) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Opcional: indicar que nenhuma mão foi detectada
            cv2.putText(frame, "Mão não detectada", (50, H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar o frame com os resultados
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalizar, liberar a câmera e fechar as janelas
    cap.release()
    cv2.destroyAllWindows()

# Perguntar ao usuário se deseja entrar no modo de jogo
response = input("Você deseja entrar no modo de jogo? (s/n): ").strip().lower()

# Perguntar a dificuldade se o modo de jogo for selecionado
difficulty = 'easy'
if response == 's':
    difficulty = input("Escolha a dificuldade (easy, medium, hard): ").strip().lower()

# Definir o modo de jogo com base na resposta do usuário
if response == 's':
    game_mode = True
elif response == 'n':
    game_mode = False
else:
    print("Resposta inválida. Executando no modo leitor de sinais por padrão.")
    game_mode = False

# Chamar a função principal com o modo e a dificuldade escolhidos
main(game_mode=game_mode, difficulty=difficulty)
