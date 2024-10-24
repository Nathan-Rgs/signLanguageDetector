import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense, LSTMV1, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# Carregar os dados processados
with open('./data/sequences_data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])  # [n_samples, SEQUENCE_LENGTH, 42]
labels = np.array(data_dict['labels'])

# Codificar os rótulos
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)

# Salvar o codificador para uso futuro
with open('./data/label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

# Construir o modelo LSTM
model = Sequential()
model.add(LSTMV1(128, input_shape=(data.shape[1], data.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTMV1(64))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

# Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Treinar o modelo
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')

# Salvar o modelo treinado
model.save('./data/lstm_model.h5')
