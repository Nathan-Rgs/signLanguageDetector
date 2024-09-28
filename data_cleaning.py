import os

# Definir o diretório base para a busca (substitua pelo caminho desejado)
base_dir = "./data/images"

# Função para verificar e excluir arquivos cujo nome tenha mais de 99 caracteres
def remove_long_filenames(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            # Obter o nome do arquivo sem a extensão
            file_name_without_extension = os.path.splitext(file)[0]
            
            # Verificar se o comprimento do nome do arquivo é maior que 99
            if int(file_name_without_extension) > 99:
                file_path = os.path.join(root, file)
                print(f"Removendo arquivo: {file_path}")
                os.remove(file_path)

# Executa a função no diretório especificado
remove_long_filenames(base_dir)
