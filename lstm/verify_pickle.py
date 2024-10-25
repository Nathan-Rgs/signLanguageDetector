import pickle, os

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'sequences_data.pickle')
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    print("Pickle carregado com sucesso!")
    print(f"Quantidade de amostras: {len(data_dict['data'])}")
    print(f"Quantidade de labels: {len(data_dict['labels'])}")
except pickle.UnpicklingError as e:
    print(f"Erro ao carregar o pickle: {e}")
except Exception as e:
    print(f"Ocorreu um erro: {e}")
