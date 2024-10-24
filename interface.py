import numpy as np
import pickle
import cv2
import mediapipe as mp
import random
import time

# Carregar o modelo treinado
model_dict = pickle.load(open('./data/random_forrest_model.p', 'rb'))
model = model_dict['model']

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

# Inicializar o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar o detector de mãos
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

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
    "SIGNIFICADO", "AUTOMOVEL", "INFORMACAO", "IMPLEMENTACAO"
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
        # Centralizar a palavra na tela
        text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2

        cv2.putText(frame, word, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Função principal
def main(game_mode=True, difficulty='easy'):
    threshold = 0.6  # Definir um limiar de confiança para a predição

    if game_mode:
        correct_words = 0
        current_word = new_word(difficulty)
        current_letter_idx = 0
        hit_letters = []
        erros = False
    
        # Exibir a palavra por 3 segundos antes do jogo começar
        ret, frame = cap.read()
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

        # Converter a imagem de BGR para RGB (necessário para o MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar a imagem para detectar as mãos
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Processar apenas a primeira mão detectada (otimização)
            hand_landmarks = results.multi_hand_landmarks[0]

            # Desenhar os pontos de referência e conexões da mão
            mp_drawing.draw_landmarks(
                frame,  # imagem para desenhar
                hand_landmarks,  # pontos detectados
                mp_hands.HAND_CONNECTIONS,  # conexões da mão
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extrair e normalizar as coordenadas x e y dos landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x * W  # Converter para coordenadas da imagem
                y = hand_landmarks.landmark[i].y * H

                x_.append(x)
                y_.append(y)

                # Normalizar as coordenadas e criar o vetor de features
                data_aux.append(hand_landmarks.landmark[i].x - min(x_) / W)
                data_aux.append(hand_landmarks.landmark[i].y - min(y_) / H)

            # Fazer a previsão usando o modelo treinado
            if len(data_aux) == 42:
                prediction_proba = model.predict_proba([np.asarray(data_aux)])
                predicted_confidence = np.max(prediction_proba)  # Pega a maior confiança
                predicted_character = model.classes_[np.argmax(prediction_proba)]  # Pega a classe correspondente à maior confiança

                # Se a confiança for menor que o limiar ou a predição for "None", classifica como "desconhecido"
                if predicted_confidence < threshold or predicted_character == 'None':
                    predicted_character = "Desconhecido"

                if game_mode:
                    # Verificar se a letra detectada corresponde à letra atual da palavra
                    letra_atual = current_word[current_letter_idx]

                    if predicted_character == letra_atual:
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
                            show_spelled_word(frame, current_word, time_sec=3)

                    else:
                        # Errou a letra
                        erros = True

                    # Otimização: Exibir a palavra atual com letras acertadas e sublinhar o restante
                    word_display = "".join(hit_letters) + "_ " * (len(current_word) - current_letter_idx)
                    text_size = cv2.getTextSize(word_display, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
                    text_x = (W - text_size[0]) // 2

                    # Desenhar a palavra na tela
                    cv2.putText(frame, word_display, (text_x, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

                    # Se houve um erro, mostrar um X vermelho na tela
                    if erros:
                        cv2.putText(frame, "X", (W // 2 - 50, H // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)

                else:
                    # Apenas mostrar a letra detectada (sem jogo)
                    # Definindo os limites do retângulo com base nos pontos de referência da mão
                    x1 = int(min(x_)) - 10
                    y1 = int(min(y_)) - 10

                    x2 = int(max(x_)) + 10
                    y2 = int(max(y_)) + 10

                    # Desenhar o retângulo e exibir a letra
                    
                    if predicted_character == 'Desconhecido':
                        # Desenhar um retângulo vermelho ao redor da mão
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Vermelho
                        cv2.putText(frame, "N/A", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

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
