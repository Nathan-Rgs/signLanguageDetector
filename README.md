# Projeto de Reconhecimento de Linguagem de Sinais

Este projeto foi desenvolvido com o propósito de **auxiliar no aprendizado da Linguagem de Sinais** através de uma ferramenta interativa que reconhece gestos de mão capturados pela webcam. O sistema utiliza **Visão Computacional** e **Aprendizado de Máquina** para detectar e identificar gestos, fornecendo feedback em tempo real. É uma solução beneficente destinada a facilitar o aprendizado e o treinamento da linguagem de sinais.

## Objetivo do Projeto

O projeto visa criar uma plataforma simples e acessível para que qualquer pessoa possa praticar a Linguagem de Sinais com a ajuda de um sistema que reconhece gestos manuais e identifica as letras do alfabeto em tempo real.

## Funcionalidades

- **Captura de gestos manuais** utilizando uma webcam.
- **Reconhecimento de letras do alfabeto** da Linguagem de Sinais.
- **Feedback em tempo real** mostrando a letra correspondente ao gesto reconhecido.
- **Ferramenta de coleta de imagens** para expandir o conjunto de dados.

## Requisitos

- **Python 3.x**
- **OpenCV**
- **MediaPipe**
- **Scikit-learn**
- **Numpy**
- **Pickle**

### Instalação dos Requisitos

Certifique-se de instalar todas as dependências necessárias para rodar o projeto. Você pode usar o arquivo `requirements.txt` para isso.

## Como Executar o Projeto

O projeto consiste em três etapas principais: **coleta de imagens**, **criação do dataset** e **reconhecimento em tempo real**. Siga as etapas abaixo para executar o projeto corretamente:

### 1. **Coleta de Imagens (collect_images.py)**

Este script permite que você capture imagens dos gestos que deseja reconhecer no futuro. O script usa a webcam para capturar e salvar imagens, que serão utilizadas para treinar o modelo.

- **Saída esperada**: Imagens serão salvas em diretórios específicos, com pastas separadas para cada gesto (por exemplo, uma pasta para cada letra do alfabeto).

### 2. **Criação do Dataset (create_dataset.py)**

Após coletar as imagens, você precisa processá-las e extrair os landmarks das mãos. Este script cria um dataset que será usado para treinar o modelo.

- **Saída esperada**: O script irá gerar um arquivo `data.pickle`, que conterá as características dos gestos (landmarks) e os rótulos correspondentes.

### 3. **Interface de Reconhecimento (interface.py)**

Com o dataset criado e o modelo treinado, você pode executar a interface de reconhecimento em tempo real. A câmera detectará os gestos, e o sistema exibirá a letra reconhecida na tela.

- **Saída esperada**: A câmera será ativada e, ao detectar uma mão, mostrará os landmarks e a letra correspondente ao gesto reconhecido em tempo real.

## Estrutura do Projeto

.
├── data/ # Diretório onde os dados processados e o modelo serão salvos
│ └── model.p # Modelo treinado
| ├── images/ # Diretório para armazenar as imagens capturadas
│ └── A/ # Imagens para a letra A
│ └── B/ # Imagens para a letra B
│ └── ... # Imagens para outras letras
├── collect_images.py # Script para coletar as imagens para treinamento
├── create_dataset.py # Script para processar as imagens e criar o dataset
├── interface.py # Script para executar a interface de reconhecimento em tempo real
└── README.md # Instruções e informações sobre o projeto

## Como Contribuir

Este projeto é de código aberto e tem como objetivo ajudar a comunidade a aprender Linguagem de Sinais. Se você deseja contribuir, você pode:

- **Coletar mais dados**: Ajude a expandir o conjunto de dados coletando mais imagens de gestos.
- **Melhorar o modelo**: Trabalhe para melhorar o desempenho do modelo de reconhecimento.
- **Relatar problemas**: Caso encontre problemas, abra uma issue ou envie sugestões de melhoria.

### 4. **Executando com LSTM ao invés de Random_Forrest**

A ideia é a mesma, temos alguns steps:
1. collect_images.py: Ao executar lembre-se de fazer o movimento, se a letra for parada deixe o maximo estático durante a captura.
2. create_dataset.py: Ao executar irá gerar o dataset (sequences_data.pickle, label_encoder.pickle).
3. train_lstm_model.py: Ao criar os datasets você poderá usar o LSTM train para fazer o treinamento, ele possui 30 epochs.
4. verify_pickle.py: Simples verificação de arquivos.
5. interface.py: Irá rodar a interface assim como para random_forrest, opção game e sem game.

## Créditos

Este projeto utiliza bibliotecas como **OpenCV**, **MediaPipe** e **Scikit-learn** para a detecção de gestos e aprendizado de máquina. Um agradecimento especial a todos os desenvolvedores dessas ferramentas, que tornam este projeto possível.
