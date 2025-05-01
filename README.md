# Predição de Idade em Tempo Real

Este projeto utiliza YOLO e Redes Neurais Convolucionais (CNNs) para detectar rostos e prever a faixa etária em tempo real a partir de uma webcam.

## Funcionalidades
- Detecção de faces em tempo real usando YOLO.
- Predição da faixa etária com um modelo CNN.
- Aplicativo interativo utilizando Gradio.

## Estrutura do Projeto
```plaintext
├── README.md                 # Documentação do projeto
├── app/                      # Código do aplicativo e componentes principais
│   ├── age_predictor.py      # Classe para predição de idade e detecção de faces
│   ├── webcam_processor.py   # Processamento de frames da webcam
├── models/                   # Modelos treinados
│   ├── age_cnn_model_final.h5 # Modelo CNN treinado para predição de idade
├── requirements.txt          # Dependências do projeto
├── data/                     # Dados ou exemplos para o modelo
│   main.py                   # Script principal para rodar o aplicativo


```

## Como Rodar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute o aplicativo:
   ```bash
   python app/main.py
   ```

3. Abra o link gerado no navegador para acessar a interface Gradio.

## Dependências
- TensorFlow
- Torch
- OpenCV
- Gradio
- ultralytics (YOLO)
- huggingface_hub

## Licença
[MIT License](LICENSE)