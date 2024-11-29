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
### Rendimiento de sistema de detección
El sistema de detección implementa un enfoque dual utilizando dos arquitecturas YOLO. El detector principal, YOLOv5l, configurado con un umbral de confianza superior al 55%, procesó eficientemente los videos en condiciones diurnas, logrando extraer aproximadamente 33,000 frames relevantes de un total de 780 videos. Complementariamente, YOLOv8x se implementó específicamente para el procesamiento de videos del Armadillo de Nueve Bandas y Roedores, demostrando una capacidad superior en la detección bajo condiciones nocturnas y en situaciones donde las especies presentaban patrones de camuflaje complejos.

### Comparación Detallada de Modelo MobilenetV3 por Clase
| Modelo | Conf. | Clase | Precision | Recall | F1-score | Support |
|--------|--------|--------|------------|---------|-----------|----------|
| MobileNetV3 | 60% | 0 | 0.991 | 0.955 | 0.973 | 2012.0 |
|  |  | 1 | 0.550 | 0.750 | 0.635 | 44.0 |
|  |  | 2 | 0.964 | 0.930 | 0.947 | 316.0 |
|  |  | 3 | 0.516 | 0.926 | 0.663 | 121.0 |
|  |  | 4 | 0.912 | 0.989 | 0.949 | 378.0 |
|  |  | 5 | 0.958 | 0.926 | 0.942 | 1834.0 |
|  |  | accuracy | - | - | 0.942 | 4705.0 |
|  |  | macro avg | 0.815 | 0.913 | 0.851 | 4705.0 |
|  |  | weighted avg | 0.954 | 0.942 | 0.946 | 4705.0 |
| MobileNetV3 | 70% | 0 | 0.989 | 0.964 | 0.984 | 1555.0 |
|  |  | 1 | 0.759 | 1.000 | 0.863 | 22.0 |
|  |  | 2 | 0.875 | 0.980 | 0.925 | 100.0 |
|  |  | 3 | 0.484 | 0.859 | 0.619 | 71.0 |
|  |  | 4 | 0.700 | 0.988 | 0.954 | 166.0 |
|  |  | 5 | 0.938 | 0.938 | 0.956 | 1459.0 |
|  |  | accuracy | - | - | 0.940 | 3373.0 |
|  |  | macro avg | 0.791 | 0.816 | 0.751 | 3373.0 |
|  |  | weighted avg | 0.937 | 0.940 | 0.925 | 3373.0 |

### Comparación Detallada de Modelo Resnet50 por Clase
| Modelo | Conf. | Clase | Precision | Recall | F1-score | Support |
|--------|--------|--------|------------|---------|-----------|----------|
| ResNet50 | 60% | 0 | 0.989 | 0.990 | 0.989 | 2012.0 |
|  |  | 1 | 0.667 | 0.818 | 0.735 | 44.0 |
|  |  | 2 | 0.695 | 0.975 | 0.812 | 316.0 |
|  |  | 3 | 0.929 | 0.860 | 0.893 | 121.0 |
|  |  | 4 | 0.899 | 0.704 | 0.789 | 378.0 |
|  |  | 5 | 0.989 | 0.963 | 0.976 | 1834.0 |
|  |  | accuracy | - | - | 0.950 | 4705.0 |
|  |  | macro avg | 0.861 | 0.885 | 0.866 | 4705.0 |
|  |  | weighted avg | 0.957 | 0.950 | 0.951 | 4705.0 |
| ResNet50 | 70% | 0 | 0.988 | 0.976 | 0.982 | 1555.0 |
|  |  | 1 | 0.647 | 1.000 | 0.786 | 22.0 |
|  |  | 2 | 0.375 | 0.980 | 0.543 | 100.0 |
|  |  | 3 | 0.438 | 0.789 | 0.563 | 71.0 |
|  |  | 4 | 0.636 | 0.042 | 0.079 | 166.0 |
|  |  | 5 | 0.975 | 0.938 | 0.956 | 1459.0 |
|  |  | accuracy | - | - | 0.910 | 3373.0 |
|  |  | macro avg | 0.677 | 0.787 | 0.651 | 3373.0 |
|  |  | weighted avg | 0.923 | 0.910 | 0.903 | 3373.0 |
### Curva de aprendizaje de los modelos Resnet50 y MovileNetv3
<div align="center">
<table>
 <tr>
   <td><img src="https://github.com/user-attachments/assets/bd31838d-d0e4-4523-8dd8-482b94b9ff59" width="400px"><br>A) Curva aprendizaje Resnet 50 confiabilidad 60%</td>
   <td><img src="https://github.com/user-attachments/assets/94a53e3b-2793-4bc9-bd28-1a11f92e6c98" width="400px"><br>B) Curva aprendizaje Resnet 50 confiabilidad 70%</td>
 </tr>
 <tr>
   <td><img src="https://github.com/user-attachments/assets/dac0443d-19fc-4079-86b0-a597f37af8bf" width="400px"><br>C) Curva aprendizaje Mobilenetv3 confiabilidad 60%</td>
   <td><img src="https://github.com/user-attachments/assets/b37f6609-124e-4258-934b-f19237aedf24" width="400px"><br>D) Curva aprendizaje Mobilenetv3 confiabilidad 70%</td>
 </tr>
</table>
</div>

### Curva ROC de los modelos Resnet50 y MovileNetv3
<div align="center">
<table>
 <tr>
   <td><img src="https://github.com/user-attachments/assets/e12a027b-8544-4287-8c83-8e84681938ae" width="400px"><br>A) Curva ROC Mobilenetv3 confiabilidad 60%</td>
   <td><img src="https://github.com/user-attachments/assets/7568c3e4-29de-48e3-a9ee-61c922051e5f" width="400px"><br>B) Curva ROC Mobilenetv3 confiabilidad 70%</td>
 </tr>
 <tr>
   <td><img src="https://github.com/user-attachments/assets/27a7e65f-9a5c-47a6-8e3c-e59c3bd82c79" width="400px"><br>C) Curva ROC  Resnet 50 confiabilidad 60%</td>
   <td><img src="https://github.com/user-attachments/assets/6dbc2872-178a-48a5-9835-142f90a19ed3" width="400px"><br>D) Curva ROC Resnet 50 confiabilidad 70%</td>
 </tr>
</table>
</div>

### Matriz de confución de los modelos Resnet50 y MovileNetv3

<div align="center">
<table>
 <tr>
   <td><img src="https://github.com/user-attachments/assets/10ff6104-ff65-4c00-be94-e20be9756cc0" width="400px"><br>A) Matriz de confución Mobilenetv3 confiabilidad 60%</td>
   <td><img src="https://github.com/user-attachments/assets/9d241a97-7ee6-4bb7-abcd-8398cbc661a3" width="400px"><br>B) Matriz de confución Mobilenetv3 confiabilidad 70%</td>
 </tr>
 <tr>
   <td><img src="https://github.com/user-attachments/assets/9378dd41-2f2c-4b9e-9453-2f2ec3a1b7a3" width="400px"><br>C) Matriz de confución Resnet 50 confiabilidad 60%</td>
   <td><img src="https://github.com/user-attachments/assets/074250a1-7e3d-45e9-9fe2-72a4ece27a57" width="400px"><br>D) Matriz de confución Resnet 50 confiabilidad 70%</td>
 </tr>
</table>
</div>
