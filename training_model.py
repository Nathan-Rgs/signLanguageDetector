import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carrega os dados do arquivo 'data.pickle' que foi previamente salvo usando pickle
data_dict = pickle.load(open('./data/data.pickle', 'rb'))

# Converte os dados e rótulos (labels) do formato de lista para arrays NumPy, para melhor eficiência e compatibilidade com o scikit-learn
data = np.asarray(data_dict['data'])  # Contém os landmarks (pontos de referência) das mãos
labels = np.asarray(data_dict['labels'])  # Contém os rótulos correspondentes (classes das mãos)

# Divide os dados em conjunto de treino (80%) e teste (20%) de forma aleatória, mantendo a proporção de classes igual (stratify)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializa o modelo Random Forest (classificador baseado em múltiplas árvores de decisão)
model = RandomForestClassifier()

# Treina o modelo usando o conjunto de treino (x_train são os dados, y_train são os rótulos)
model.fit(x_train, y_train)

# Faz previsões no conjunto de teste (dados de teste)
y_predict = model.predict(x_test)

# Calcula a acurácia, que é a porcentagem de previsões corretas no conjunto de teste
score = accuracy_score(y_predict, y_test)

# Exibe a acurácia (percentual de amostras classificadas corretamente)
print('{}% of samples were classified correctly !'.format(score * 100))

# Salva o modelo treinado em um arquivo pickle ('model.p') para uso posterior
f = open('./data/model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
