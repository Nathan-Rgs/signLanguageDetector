import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carrega os dados do arquivo 'data.pickle' que foi previamente salvo usando pickle
data_dict = pickle.load(open('./data/data.pickle', 'rb'))

# Verifica o comprimento de cada lista em 'data' para garantir consistência
data_lengths = [len(d) for d in data_dict['data']]
max_length = max(data_lengths)

# Inicializa um conjunto (set) para armazenar rótulos de letras com dados inconsistentes
inconsistent_labels = set()

# Verifica se há inconsistências no tamanho das listas
for i, length in enumerate(data_lengths):
	if length != max_length:
		# Adiciona o rótulo da letra inconsistente ao conjunto (sem repetições)
		label = data_dict['labels'][i]
		inconsistent_labels.add(label)

# Mostra as letras com dados inconsistentes ao final da execução
if inconsistent_labels:
	print("Letras com dados inconsistentes:")
	for label in inconsistent_labels:
		print(f"- {label}")
else:
	print("Todos os dados estão consistentes!")

# Padroniza as listas: Preenche com zeros as listas que forem menores
data = np.array([d + [0] * (max_length - len(d)) for d in data_dict['data']])

# Converte os rótulos para array NumPy
labels = np.asarray(data_dict['labels'])

# Divide os dados em conjunto de treino (80%) e teste (20%)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializa o modelo Random Forest
model = RandomForestClassifier()

# Treina o modelo usando o conjunto de treino
model.fit(x_train, y_train)

# Faz previsões no conjunto de teste
y_predict = model.predict(x_test)

# Calcula a acurácia
score = accuracy_score(y_predict, y_test)

# Exibe a acurácia
print('{}% de amostras classificadas corretamente!'.format(score * 100))

# Salva o modelo treinado em um arquivo pickle
with open('./data/model.p', 'wb') as f:
	pickle.dump({'model': model}, f)
