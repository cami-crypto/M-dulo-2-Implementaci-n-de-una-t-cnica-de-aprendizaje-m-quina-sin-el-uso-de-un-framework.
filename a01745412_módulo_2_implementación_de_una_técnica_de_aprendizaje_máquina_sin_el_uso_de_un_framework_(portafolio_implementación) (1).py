"""a01745412_Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)

#**ALGORITMO - ÁRBOL DE DECISIÓN**
"""

#Se usan operaciones matemáticas básicas y funciones estándar de Python para calcular métricas como la entropía.
import numpy as np
from collections import Counter
import math

"""#**Generación de Datos Sintéticos Más Informativos**

Conjunto de datos que tenga características correlacionadas con las clases.
"""

# Generar datos sintéticos con valores aleatorios cada ejecución.
def generate_synthetic_data(n_samples=50):
    X1 = np.random.normal(0, 1, n_samples)  # Primera característica centrada en 0
    X2 = X1 + np.random.normal(0, 0.5, n_samples)  # Segunda característica relacionada con X1
    y = np.where(X1 + X2 > 0, 1, 0)  # Clase depende de una combinación de X1yX2
    return np.column_stack((X1, X2, y))

# Generar eldataset
dataset = generate_synthetic_data()

# Normalizar características
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset[:, :-1] = scaler.fit_transform(dataset[:, :-1])  #(x1, X2)

"""#**Selección de la Mejor División para un Nodo**

Evalúa cada característica y sus valores únicos para encontrar la división que maximiza la ganancia de información. Devuelve el índice de la característica, el valor de la división y los grupos resultantes que ofrecen la mejor separación de las clases en el dataset.
"""

# Seleccionar el mejor punto de division para un ndo.
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, -1, None
    # Iterar sobre cada característica
    for index in range(len(dataset[0])-1):
        unique_values = set(row[index] for row in dataset)  # Mejorar precisión en valores únicos
        for value in unique_values:
            groups = test_split(index, value, dataset)
            gain = entropy([dataset]) - entropy(groups)
            if gain > b_score:
                b_index, b_value, b_score, b_groups = index, value, gain, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

"""#**Construcción Recursiva del Árbol de Decisión**

Se definen las funciones para construir recursivamente el árbol de decisión, estableciendo nodos y hojas.
"""

# Dividir el nodo o crear unnodo terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # Verificar si no hay división
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # Verificar profundidad máxima
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # Procesar rama izquierda
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        # Realizar poda
        if gini_index([left], list(set(row[-1] for row in left))) < 0.1:
            node['left'] = to_terminal(left)
        else:
            split(node['left'], max_depth, min_size, depth+1)
    # Procesar rama derecha
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        # Realizar poda
        if gini_index([right], list(set(row[-1] for row in right))) < 0.1:
            node['right'] = to_terminal(right)
        else:
            split(node['right'], max_depth, min_size, depth+1)

# Construir un árbol de decisión
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

"""#**Predicción con el Árbol de Decisión y Cálculo de Métricas**

Se presentan funciones para hacer predicciones con el árbol de decisión entrenado y para calcular métricas de evaluación como precisión, recall, F1-score y la matriz de confusión.
"""

# Hacer una predicción con el árbol de decisión
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Evaluar la precisión del modelo
def accuracy_metric(actual, predicted):
    correct = sum(1 for i in range(len(actual)) if actual[i] == predicted[i])
    return correct / float(len(actual)) * 100.0

# Generar métricas como precisión, recall, F1-score y matriz de confusión
def calculate_metrics(actual, predicted):
    tp = sum(1 for a, p in zip(actual, predicted) if a == p == 1)
    tn = sum(1 for a, p in zip(actual, predicted) if a == p == 0)
    fp = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)

    # Precision, Recall y F1-Score
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    # Matriz de confusión
    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    return precision, recall, f1_score, confusion_matrix

"""#**Preparación del Conjunto de Datos Sintético, Entrenamiento del Modelo y Evaluación**

Se crea un conjunto de datos sintético, dividiéndolo en subconjuntos de entrenamiento, validación y prueba, entrenando el modelo, y realizando las predicciones y evaluando las métricas.
"""

# División del dataset en train, validation, y test
train, validation, test = dataset[:30], dataset[30:40], dataset[40:]

# Árbol de decisión con parámetros ajustados
tree = build_tree(train, max_depth=6, min_size=2)

# Predicciones y evaluar con los datos de prueba
predictions = [predict(tree, row) for row in test]
actual = [row[-1] for row in test]

# Cálculo de métricas
accuracy = accuracy_metric(actual, predictions)
precision, recall, f1_score, confusion_matrix = calculate_metrics(actual, predictions)

print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')
print('Confusion Matrix:')
print(confusion_matrix)
