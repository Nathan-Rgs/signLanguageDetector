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

# Verifica se há inconsistências no tamanho das listas
for i, length in enumerate(data_lengths):
	if length != max_length:
		# Obtendo o rótulo e a imagem inconsistente
		label = data_dict['labels'][i]
		print(f"Inconsistencia no tamanho da lista, index {i}: esperado {max_length}, obtido {length}. Rotulo: {label}")

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
print('{}% of samples were classified correctly!'.format(score * 100))

# Salva o modelo treinado em um arquivo pickle
with open('./data/model.p', 'wb') as f:
	pickle.dump({'model': model}, f)
