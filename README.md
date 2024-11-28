# Sistema de Monitoreo Automático de Fauna Silvestre en el Bosque del Chocó 🦁 🌳
## Descripción
Este proyecto implementa un sistema de visión artificial de última generación para la detección y clasificación automática de especies en la Reserva Jocotoco Canandé, una de las regiones más biodiversas del planeta. Utilizando una arquitectura dual que combina modelos YOLO para detección y redes neuronales profundas (ResNet50 y MobileNetV3) para clasificación, el sistema procesa videos de cámaras trampa para identificar y monitorear seis especies nativas:
- 🦊 Agutí Centroamericano
- 🐿️ Ardillas
- 🦡 Armadillo de Nueve Bandas
- 🦫 Paca de Tierras Bajas
- 🐀 Roedores
- 🦃 Tinamú Grande

El sistema alcanza un F1-score ponderado de 0.951, demostrando su efectividad para el monitoreo automatizado de biodiversidad. Esta herramienta representa un avance significativo en la conservación de especies, permitiendo a investigadores y conservacionistas procesar grandes volúmenes de datos de manera eficiente y precisa.
Desarrollado como parte de una iniciativa para modernizar las prácticas de conservación en el Chocó ecuatoriano, este proyecto establece un precedente para la implementación de tecnologías de inteligencia artificial en la protección de ecosistemas críticos.

## 🔄Pipeline del Sistema 

El sistema implementa un enfoque de dos etapas para el procesamiento de videos de cámaras trampa:

### 📸 Fase 1: Detección de Objetos

- Preprocesamiento: Extracción de frames de videos con resolución 640x368 píxeles

- Detección:

   - YOLOv5l: Detección general con umbral de confianza >55%

   - YOLOv8x: Implementado específicamente para especies nocturnas y de difícil detección

- Extracción: Hasta 350 frames por video con sus respectivos bounding boxes

### 🔍 Fase 2: Clasificación de Especies

- Modelos Implementados:

  - ResNet50: Optimizado para máxima precisión

  - MobileNetV3: Diseñado para eficiencia computacional

- Técnicas de Mejora:

Transfer Learning con pesos preentrenados de ImageNet

Data Augmentation (RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter)

Manejo de desbalance de clases mediante pesos adaptativos

![diagram 2 more big](https://github.com/user-attachments/assets/11e59021-bb86-4e25-a773-b526bc11915d)

## 📊 Dataset
  
El conjunto de datos consta de 780 videos de cámaras trampa (130 por cada una de las seis especies objetivo) capturados en la Reserva Jocotoco Canandé. Los datos se dividieron en tres conjuntos: 70% para entrenamiento, 15% para validación y 15% para pruebas. El procesamiento mediante detectores YOLO resultó en aproximadamente 33,000 frames relevantes, proporcionando una base sólida para el entrenamiento y evaluación de los modelos de clasificación.

## 📋 Requisitos
- torch>=2.3.0
- torchvision>=0.18.0
- pytorch-lightning>=2.4.0
- ultralytics>=8.2.81
- opencv-python>=4.8.0
- pandas>=2.0.3
- numpy>=1.25.2
- Pillow>=9.4.0
- scikit-learn>=1.2.2
- matplotlib>=3.7.1
- seaborn>=0.13.1
- torchmetrics>=1.4.1
## ⚙️ Instalación

```
# Clonar repositorio
git clone [URL del repositorio]

# Instalar dependencias
pip install -r requirements.txt
```
## 💻 Requisitos de Hardware
- CUDA 12.1 compatible
- RAM: 32GB recomendado
- GPU: NVIDIA (8GB VRAM recomendado)
  
## 🚀 Uso
### 1. Instalación de Requisitos
Asegúrese de tener todas las dependencias necesarias instaladas con las versiones especificadas.

```
# Clonar repositorio
pip install -r requirements.txt
```
### 2. Detección de Frames en Videos
El primer paso consiste en procesar los videos mediante el detector YOLO para extraer frames relevantes:
```
# Configuración del detector YOLO
from ultralytics import YOLO

# Cargar el modelo
model = YOLO('yolov8x.pt')  # o 'yolov5l.pt'

# Configurar parámetros
conf_threshold = 0.55  # Umbral de confianza >55%
max_frames = 350      # Frames máximos por video

# Ejecutar detección
results = model(video_path, conf=conf_threshold)
```
Este proceso generará un dataset de frames con sus respectivos bounding boxes para cada especie detectada.
### 3. Clasificación de Especies
Una vez obtenidos los frames, se procede con el entrenamiento del clasificador:
```
# Configuración del modelo de clasificación
from pytorch_lightning import Trainer

# Definir hiperparámetros
hyperparameters = {
    'learning_rate': 0.0001,
    'batch_size': 32,
    'max_epochs': 100,
    'early_stopping': 10
}

# Iniciar entrenamiento con monitoreo
trainer = Trainer(
    max_epochs=hyperparameters['max_epochs'],
    accelerator='gpu',
    logger=TensorBoardLogger('logs/'),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
)
```
### 4. Monitoreo y Evaluación
Los resultados del entrenamiento pueden visualizarse en tiempo real mediante TensorBoard:
```
tensorboard --logdir=logs/
```
En la interfaz de TensorBoard podrá monitorear:
Acceder a http://localhost:6006 para visualizar:

- Evolución del entrenamiento
- Métricas de rendimiento por especie
- Visualizaciones de matrices de confusión
- Curvas ROC para evaluación de rendimiento
## 📊 Resultados
