"""
Nombre del autor: Victor Munoz S. victormunozs@usm.cl
Descripcion: Codigo para realizar el Proyecto del ramo IPD477
*Se omitieron las tildes en todo el codigo y graficos debido problemas de codificacion.
"""
#------------------DEFINE-------------------
fs=512 #Frecuencia de muestreo
gpu_Use=80 #Porcentaje de memoria de GPU a utilizar
segment_length_samples = 2 * fs # Longitud de cada segmento en muestras
#------------------IMPORTS------------------
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Flatten, Dense, Concatenate, LSTM,Lambda
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.metrics import Recall,Precision
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import GPUtil
import numpy as np
import time
from keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import json


#-----------------FUNCTIONS-----------------
def assign_gpu_memory(percentage):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Obteniendo la memoria total de la primera GPU
            gpu = GPUtil.getGPUs()[0]
            total_memory = gpu.memoryTotal  # Memoria total en MB

            percentage = min(percentage, 80)

            # Calculando el límite de memoria como un porcentaje del total
            memory_limit = int(total_memory * (percentage / 100))

            # Configurando el dispositivo lógico con el límite de memoria calculado
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU with", memory_limit, "MB")
        except RuntimeError as e:
            print(e)
        except IndexError as e:
            print("No se encontraron GPUs disponibles.")
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
def build_model(lr,dense_units, kernel_size1, kernel_size2, kernel_size3, filters1, filters2, filters3):
    # Entrada
    input_shape = (segment_length_samples, 3)  
    inputs = Input(shape=input_shape)

    # Separar los ejes
    x_axis = inputs[:, :, 0]
    y_axis = inputs[:, :, 1]
    z_axis = inputs[:, :, 2]
   
    x_axis = Lambda(lambda x: tf.expand_dims(x, axis=-1))(x_axis)
    y_axis = Lambda(lambda x: tf.expand_dims(x, axis=-1))(y_axis)
    z_axis = Lambda(lambda x: tf.expand_dims(x, axis=-1))(z_axis)

    # Capa convolucional para el eje X
    x = Conv1D(filters=filters1, kernel_size=kernel_size1, activation='PReLU')(x_axis)
    x = Conv1D(filters=filters2, kernel_size=kernel_size2, activation='PReLU')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=filters3, kernel_size=kernel_size3, activation='PReLU')(x)
    x = BatchNormalization()(x)
 

    # Capa convolucional para el eje Y
    y = Conv1D(filters=filters1, kernel_size=kernel_size1, activation='PReLU')(y_axis)
    y = Conv1D(filters=filters2, kernel_size=kernel_size2, activation='PReLU')(y)
    y = BatchNormalization()(y)
    y = Conv1D(filters=filters3, kernel_size=kernel_size3, activation='PReLU')(y)
    y = BatchNormalization()(y)
 

    # Capa convolucional para el eje Z
    z = Conv1D(filters=filters1, kernel_size=kernel_size1, activation='PReLU')(z_axis)
    z = Conv1D(filters=filters2, kernel_size=kernel_size2, activation='PReLU')(z)
    z = BatchNormalization()(z)
    z = Conv1D(filters=filters3, kernel_size=kernel_size3, activation='PReLU')(z)
    z = BatchNormalization()(z)
   
    lstm_units = int(64)
    # Capas LSTM para cada eje
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = LSTM(int(lstm_units / 2))(x)

    y = LSTM(lstm_units, return_sequences=True)(y)
    y = LSTM(int(lstm_units / 2))(y)

    z = LSTM(lstm_units, return_sequences=True)(z)
    z = LSTM(int(lstm_units / 2))(z)

    # Concatenar los resultados de las capas LSTM
    combined = Concatenate(axis=-1)([x, y, z])
    flat = Flatten()(combined)

    # Regularización L1-L2 con hiperparámetro lr
    regularizer = l1_l2(l1=lr, l2=lr)

    # Capas finales con unidades densas variables
    final_layer = Dense(dense_units, activation='relu', kernel_regularizer=regularizer)(flat)
    final_layer = Dense(int(dense_units / 2), activation='relu', kernel_regularizer=regularizer)(final_layer)

    outputs = Dense(1, activation='sigmoid', kernel_regularizer=regularizer)(final_layer)

    # Crear el modelo
    model = Model(inputs=inputs, outputs=outputs)

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=[
                           Recall(name='Mr_recall', thresholds=0.5, class_id=0),
                           Precision(name='Mr_precision', thresholds=0.5, class_id=0),
                          ])
    return model
def objective(trial, train_dataset, val_dataset, train_labels):
    # Rangos de hiperparámetros
    lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
    dense_units = trial.suggest_categorical('dense_units', [64, 128])
    kernel_size1 = trial.suggest_categorical('kernel_size1', [5, 10, 15])
    kernel_size2 = trial.suggest_categorical('kernel_size2', [2,3,5])
    kernel_size3 = trial.suggest_categorical('kernel_size3', [1,2,3])  
    filters1 = trial.suggest_categorical('filters1', [32, 64, 128])
    filters2 = trial.suggest_categorical('filters2', [64, 128])
    filters3 = trial.suggest_categorical('filters3', [128, 256])  

    model = build_model(lr, dense_units, kernel_size1, kernel_size2, kernel_size3, filters1, filters2, filters3)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001)
    
    # Entrenar el modelo
    history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping,reduce_lr],use_multiprocessing=True,batch_size=64)
    
    # Calcular el recall promedio en el conjunto de validación
    recalls = [history.history[f'val_{name}_recall'] for name in [ 'Mr']]
    avg_recall = np.mean([np.amax(recall) for recall in recalls])

    return avg_recall
#-------------------MAIN---------------------
# Asigna memoria de GPU
assign_gpu_memory(80)

#Cargar datos
train_segments_x = np.load('Proyecto/train_segments_x.npy')
train_segments_y = np.load('Proyecto/train_segments_y.npy')
train_segments_z = np.load('Proyecto/train_segments_z.npy')
train_labels = np.load('Proyecto/train_labels.npy')

val_segments_x = np.load('Proyecto/val_segments_x.npy')
val_segments_y = np.load('Proyecto/val_segments_y.npy')
val_segments_z = np.load('Proyecto/val_segments_z.npy')
val_labels = np.load('Proyecto/val_labels.npy')

test_segments_x = np.load('Proyecto/test_segments_x.npy')
test_segments_y = np.load('Proyecto/test_segments_y.npy')
test_segments_z = np.load('Proyecto/test_segments_z.npy')
test_labels = np.load('Proyecto/test_labels.npy')

#Crear Dataset
train_dataset = create_dataset(train_segments_x,train_segments_y,train_segments_z,train_labels)
val_dataset = create_dataset(val_segments_x,val_segments_y,val_segments_z,val_labels)
test_dataset = create_dataset(test_segments_x,test_segments_y,test_segments_z,test_labels)

#Tiempo de ejecucion
Tic=time.time()

#CNN
print(train_dataset.element_spec)

# Optimización de hiperparámetros
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, train_dataset, val_dataset,train_labels), n_trials=2)

# Mejores hiperparámetros encontrados
print('Número de ensayos terminados: ', len(study.trials))
print('Mejores parámetros: ', study.best_trial.params)
print('Mejor valor de recall promedio: ', study.best_value)

# Obtén los mejores hiperparámetros de Optuna
best_params = study.best_trial.params
model = build_model(**best_params)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Entrenar el modelo
# Reentrena el modelo
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=[early_stopping,reduce_lr], use_multiprocessing=True,batch_size=64)

# Guardar el modelo
model.save('Proyecto/output_files/Model.keras')

# Graficar Pérdida de Entrenamiento y Validación
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs, loss, 'b', label='Entrenamiento')
plt.plot(epochs, val_loss, 'r', label='Validación')
plt.title('Pérdida de Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Guardar los resultados en un archivo JSON
optimizacion_resultados = {
    "numero_ensayos_terminados": len(study.trials),
    "mejores_parametros": study.best_trial.params,
    "mejor_valor_recall": study.best_value,
}

with open('Proyecto/output_files/optimizacion_resultados.json', 'w') as file:
    json.dump(optimizacion_resultados, file)

# Tiempo de ejecución
Toc = time.time()
print("Tiempo de ejecución: ", (Toc-Tic)/60, " minutos")
print('Fin del programa')




