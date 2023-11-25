#-------------IMPORTS----------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from tensorflow.keras.utils import plot_model
import pydot
import graphviz

#-------------CONSTANTS----------------
fs=512
segment_length_samples = 2 * fs

#-------------FUNCTIONS----------------
def create_dataset(segments_x, segments_y, segments_z, all_labels):
    # Convertir a tensores
    segments_x_tensor = tf.convert_to_tensor(segments_x, dtype=tf.float32)
    segments_y_tensor = tf.convert_to_tensor(segments_y, dtype=tf.float32)
    segments_z_tensor = tf.convert_to_tensor(segments_z, dtype=tf.float32)

    # Reestructurar los tensores
    segments_x_tensor = tf.reshape(segments_x_tensor, shape=(-1, segment_length_samples, 1))
    segments_y_tensor = tf.reshape(segments_y_tensor, shape=(-1, segment_length_samples, 1))
    segments_z_tensor = tf.reshape(segments_z_tensor, shape=(-1, segment_length_samples, 1))

    # Combinar los ejes en un tensor de características
    features_tensor = tf.stack([segments_x_tensor, segments_y_tensor, segments_z_tensor], axis=2)

    # Convertir las etiquetas a tensor
    labels_tensor = tf.convert_to_tensor(all_labels, dtype=tf.float32)

    # Crear un Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))

    # Mezclar y agrupar los datos
    dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

    return dataset

#-------------MAIN----------------
test_segments_x = np.load('Proyecto/train_segments_x.npy')
test_segments_y = np.load('Proyecto/train_segments_y.npy')
test_segments_z = np.load('Proyecto/train_segments_z.npy')
test_labels = np.load('Proyecto/train_labels.npy')

test_segments_x = test_segments_x[:256*segment_length_samples]
test_segments_y = test_segments_y[:256*segment_length_samples]
test_segments_z = test_segments_z[:256*segment_length_samples]


# Crear el conjunto

test_dataset=create_dataset(test_segments_x, test_segments_y, test_segments_z, test_labels)

# Cargar el modelo
model = tf.keras.models.load_model('Proyecto/ouput_files_multilabel/Model.keras',safe_mode=False)
plot_model(model, to_file='modelo.png', show_shapes=True, show_layer_names=True)

# Evaluación del modelo con el conjunto de prueba
test_loss, Mr_recall, AS_recall, TR_recall, Mr_precision, AS_precision, TR_precision = model.evaluate(test_dataset)
print(f'Loss en prueba: {test_loss}')
print(f'Recall en prueba: Mr: {Mr_recall}')
print(f'Precision en prueba: Mr: {Mr_precision}')
print(f'Recall en prueba: AS: {AS_recall}')
print(f'Precision en prueba: AS: {AS_precision}')
print(f'Recall en prueba: TR: {TR_recall}')
print(f'Precision en prueba: TR: {TR_precision}')

evaluacion_resultados = {
    "loss_en_prueba": test_loss,
    "recall_en_prueba": {
        "Mr": Mr_recall,
        'AS': AS_recall,
        'Tr': TR_recall
    },
    'precision_en_prueba': {
        "Mr": Mr_precision,
        'AS': AS_precision,
        'Tr': TR_precision
    }
}

with open('Proyecto/ouput_files_multilabel/optimizacion_resultados.json', 'w') as file:
    json.dump(evaluacion_resultados, file)

# Obtener las predicciones del modelo
predicted_probs = model.predict(test_dataset)
predicted_labels = (predicted_probs > 0.5).astype(int)

# Número de clases (flags)
num_classes = predicted_probs.shape[1]

# Graficar Matrices de Confusión
plt.figure(figsize=(15, 10))
for i in range(num_classes):
    cm = confusion_matrix(test_labels[:, i], predicted_labels[:, i])
    plt.subplot(3, 2, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión para la Clase {i}')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
plt.tight_layout()
plt.show()
