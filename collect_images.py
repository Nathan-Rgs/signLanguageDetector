import os
import cv2
import string

DATASET_SIZE = 300  # Número de imagens que queremos adicionar por letra

# Diretório para salvar as imagens
DATA_DIR = './data/images'
if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)

# Lista de letras do alfabeto A-Z
alphabet = list(string.ascii_uppercase)

# Função para obter o maior número de arquivo na pasta ou começar em 0 se não houver arquivos
def get_starting_counter(directory):
	if not os.path.exists(directory):
		return 0
	# Listar todos os arquivos da pasta
	files = os.listdir(directory)
	# Filtrar somente os arquivos com extensão .jpg e extrair o número do nome do arquivo
	numbers = [int(file.split('.')[0]) for file in files if file.endswith('.jpg')]
	if numbers:
		return max(numbers) + 1  # Iniciar um número acima do maior encontrado
	else:
		return 0  # Se não houver arquivos, começar do zero

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
		cv2.putText(frame, f'Letra: {letter} - Pressione "Q"', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
		cv2.imshow('frame', frame)

		# Aguarda até que o usuário pressione "q" para iniciar a captura
		if cv2.waitKey(25) == ord('q'):
			break

	# Obter o valor inicial do contador para salvar as imagens
	counter = get_starting_counter(class_dir)

	# Queremos adicionar mais 100 imagens, independente de quantas já existam
	imagens_restantes = DATASET_SIZE

	# Capturar e salvar imagens até adicionar mais 100
	while imagens_restantes > 0:
		ret, frame = cap.read()

		if not ret:
			print("Erro: Não foi possível capturar a imagem da câmera.")
			break

		# Mostrar o frame na tela
		cv2.imshow('frame', frame)
		cv2.waitKey(1)

		# Salvar a imagem com o nome correspondente ao valor do counter
		cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
		counter += 1
		imagens_restantes -= 1

	print(f'Capturadas mais {DATASET_SIZE} imagens para a letra {letter}.')

cap.release()
cv2.destroyAllWindows()
