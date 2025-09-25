# 🧠 Perceptrón y Redes Neuronales desde Cero con Python

Este proyecto es una implementación **desde cero** de varios modelos de aprendizaje automático utilizando **NumPy** y visualizaciones con **matplotlib**. Aquí no usamos frameworks como TensorFlow o PyTorch, lo que permite entender los fundamentos del aprendizaje automático.

---

## 📚 ¿Qué se aprende con este proyecto?

### 1. **Funciones de activación**
- Lineal
- Escalón (step)
- Sigmoid
- ReLU

### 2. **Funciones de pérdida**
- MSE (Mean Squared Error)
- BCE (Binary Crossentropy)
- Crossentropy para clasificación multiclase

### 3. **Perceptrón simple**
- Implementación básica de un perceptrón para regresión y clasificación binaria
- Visualización de la frontera de decisión durante el entrenamiento
- Uso del dataset `Iris` de `sklearn`

### 4. **Clasificación multiclase**
- Implementación de regresión Softmax para 3 clases (Iris Setosa, Versicolor, Virginica)
- Visualización del proceso de aprendizaje durante distintas épocas
- Función `softmax` implementada manualmente

### 5. **Red neuronal multicapa (MLP)**
- Una red con una capa oculta (`Hidden Layer`)
- Activación `ReLU` y pérdida MSE
- Entrenamiento usando **backpropagation**
- Predicción de una función cuadrática
- Visualización de cómo aprende la red

---

## 📊 Visualizaciones

Se incluyen visualizaciones del:
- comportamiento de funciones de activación
- ajuste del perceptrón en regresión lineal
- frontera de decisión del perceptrón
- evolución de la clasificación multiclase
- ajuste de una red neuronal para regresión

---

## ⚙️ Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
