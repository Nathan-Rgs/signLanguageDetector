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

# Listas de palavras por dificuldade
easy_words = [
    "AMOR", "PAZ", "LUA", "SOL", "FLOR", "BOCA", "OLHO", "MAO", "PE", "CEU", "MAR", "BOM", "MAU", "VIDA", "FE", 
    "CALMA", "DIA", "NOITE", "SORRIR", "AGUA"
]
medium_words = [
    "CASA", "MUNDO", "FAMILIA", "LIVRO", "ESCOLA", "AMIGO", "FELIZ", "CIDADE", "PEIXE", "FLORIDA"
]
hard_words = [
    "CONHECIMENTO", "RESPONSABILIDADE", "DESENVOLVIMENTO", "UNIVERSIDADE", "INTELIGENCIA", "COMUNICACAO", 
    "SIGNIFICADO", "AUTOMOVEL", "INFORMAÇÃO", "IMPLEMENTACAO"
]

# Função para escolher uma palavra aleatória com base na dificuldade
def new_word(difficulty):
    if difficulty == 'easy':
        return random.choice(easy_words)
    elif difficulty == 'medium':
        return random.choice(medium_words)
    elif difficulty == 'hard':
        return random.choice(hard_words)
    else:
        return random.choice(easy_words)  # Padrão para fácil

# Função para exibir a palavra por 3 segundos antes de começar a soletrar
def show_spelled_word(frame, word, time_sec=3):
    start_time = time.time()
    while time.time() - start_time < time_sec:
        # Mostrar a palavra no centro da tela
        cv2.putText(frame, word, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Função principal que alterna entre o modo de jogo e leitor de linguagem de sinais
def main(game_mode=True, difficulty='easy'):
    if game_mode:
        correct_words = 0
        current_word = new_word(difficulty)
        current_letter_idx = 0
        hit_letters = []
        erros = False
    
        # Exibir a palavra por 3 segundos antes do jogo começar
        ret, frame = cap.read()
        if ret:
            show_spelled_word(frame, current_word, time_sec=3)

    # Loop principal
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            print("Erro ao acessar a câmera.")
            break

        H, W, _ = frame.shape

        # Converter a imagem de BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Processar apenas a primeira mão detectada
            hand_landmarks = results.multi_hand_landmarks[0]

            # Desenhar os pontos de referência e conexões da mão
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

                if game_mode:
                    # Verificar se a letra detectada corresponde à letra atual da palavra
                    letra_atual = current_word[current_letter_idx]

                    if predicted_class[0] == letra_atual:
                        # Acertou a letra atual
                        hit_letters.append(letra_atual)
                        current_letter_idx += 1
                        erros = False

                        # Se a palavra foi soletrada completamente
                        if current_letter_idx == len(current_word):
                            correct_words += 1
                            current_letter_idx = 0
                            hit_letters = []
                            current_word = new_word(difficulty)

                            # Exibir a nova palavra por 3 segundos
                            ret, frame = cap.read()
                            if ret:
                                show_spelled_word(frame, current_word, time_sec=3)

                    else:
                        # Errou a letra
                        erros = True

                    # Exibir a palavra no canto inferior da tela, pintando as letras acertadas de verde
                    word_display = ""
                    for i, letter in enumerate(current_word):
                        if i < current_letter_idx:
                            word_display += f"{letter} "
                        else:
                            word_display += f"_ "

                    # Desenhar a palavra na tela
                    cv2.putText(frame, word_display, (50, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

                    # Se houve um erro, mostrar um X vermelho na tela
                    if erros:
                        cv2.putText(frame, "X", (W // 2 - 50, H // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)

        else:
            # Se não houver mão detectada, limpar a sequência
            sequence.clear()

        if game_mode:
            # Desenhar um retângulo ao redor da mão
            if results.multi_hand_landmarks:
                x_coords = [lm.x * W for lm in hand_landmarks.landmark]
                y_coords = [lm.y * H for lm in hand_landmarks.landmark]
                x1, y1 = int(min(x_coords)) - 10, int(min(y_coords)) - 10
                x2, y2 = int(max(x_coords)) + 10, int(max(y_coords)) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # Indicar que nenhuma mão foi detectada
                cv2.putText(frame, "Mão não detectada", (50, H - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Modo leitor de sinais padrão
            if results.multi_hand_landmarks:
                x_coords = [lm.x * W for lm in hand_landmarks.landmark]
                y_coords = [lm.y * H for lm in hand_landmarks.landmark]
                x1, y1 = int(min(x_coords)) - 10, int(min(y_coords)) - 10
                x2, y2 = int(max(x_coords)) + 10, int(max(y_coords)) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                if predicted_class[0] != 'None':
                    cv2.putText(frame, predicted_class[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "N/A", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                # Indicar que nenhuma mão foi detectada
                cv2.putText(frame, "Mão não detectada", (50, H - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar o frame com os resultados
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fim do jogo, liberar a câmera e fechar as janelas
    cap.release()
    cv2.destroyAllWindows()
    if game_mode:
        print(f"Jogo encerrado! Você soletrou {correct_words} palavras corretamente!")

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
