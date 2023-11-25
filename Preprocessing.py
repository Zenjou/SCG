"""
Nombre del autor: Victor Munoz S. victormunozs@usm.cl
Descripcion: Codigo para realizar el Proyecto del ramo IPD477
*Se omitieron las tildes en todo el codigo y graficos debido problemas de codificacion.
"""
#-------------------------IMPORTS-------------------------#
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter , freqz,butter
import tensorflow as tf
import GPUtil
import random
from ecgdetectors import Detectors
import time
import pandas as pd
from scipy.fft import fft

#-------------------------DEFINE-------------------------#
fs=512 #Frecuencia de muestreo
plot_ON=1
zoom_ON=1 #Zoom en graficos de señales
Filter='Butter' #FIR o Butter
gpu_Use=60 #Porcentaje de memoria de GPU a utilizar
segment_length_samples = 2 * fs # Longitud de cada segmento en muestras

#-------------------------SETUP-----------------------------#
tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#-------------------------FUNCTIONS-------------------------#
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

def Filter_FIR_Configuration():
    # Parámetros del filtro
    highcut = 40.0       # Frecuencia de corte alta
    lowcut = 1.0         # Frecuencia de corte baja
    numtaps = 101        # Orden del filtro + 1
    window = 'blackman'  # Ventana de Blackman
    

    # Diseño del filtro FIR pasa-alto
    nyquist = 0.5 * fs
  
    # Diseño del filtro FIR pasa-banda
    low = lowcut / nyquist
    high = highcut / nyquist
    b_bandpass = firwin(numtaps, [low, high], pass_zero=False, window=window)

    b_combined =  b_bandpass  # Combinar filtros para mayor atenuación en bajas frecuencias

    if plot_ON:
        # Respuesta en Frecuencia del Filtro Combinado
        w, h = freqz(b_combined, worN=8000)
        plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)), label='FIR')
        plt.title("Respuesta en Frecuencia del Filtro FIR")
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Amplitud [dB]")
        plt.grid()
        plt.legend()
        plt.show()
    return b_combined , 1

def Filter_Butter_Configuration():
    # Parámetros del filtro
    lowcut = 1      # Frecuencia de corte baja
    highcut = 20     # Frecuencia de corte alta
    
    w_lowcut= (lowcut / (fs / 2))
    w_highcut= (highcut / (fs / 2))
    
    #Coeficientes Pasa_Banda
    b_band, a_band = butter(5, [w_lowcut, w_highcut], btype='bandpass', analog=False)
    
    if plot_ON==1:
        # Respuesta en Frecuencia Filtro:
        frecuencia, respuesta_frecuencia_band = freqz(b_band, a_band)
        frecuencia_hz_band = frecuencia * (fs / (2 * np.pi))
        plt.figure()
        plt.plot(frecuencia_hz_band, 20 * np.log10(np.abs(respuesta_frecuencia_band)), 'b', label='Respuesta en Frecuencia')
        plt.title("Respuesta en Frecuencia del Filtro Butterworth Pasa Banda")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Ganancia (dB)")
        plt.grid()
        plt.legend()
        plt.xlim([-5, 250]) 
        plt.ylim([-30, 5])
        plt.show()

    return b_band, a_band

def plot_signals(signal1, signal2, signal3,z_peak):
    
    length = len(signal1)/fs
    start_time = random.randint(0,int(length)-5) 
    end_time = start_time + 5

    
    plt.figure()
    titles = ['Lateral', 'Head to Foot', 'Dorsal Ventral']
    for i, signal in enumerate([signal1, signal2, signal3], start=1):
        plt.subplot(3, 1, i)
        n_samples = len(signal)
        time_array = np.arange(n_samples) / fs
        plt.plot(time_array, signal)
        
        if len(z_peak) > 0:
            if i == 3:
                time_peaks = np.array(z_peak) / fs
                plt.scatter(time_peaks, [signal3[i] for i in z_peak], c='red', marker='x', label='Picos R')
        
        plt.title(titles[i-1])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        if zoom_ON == 1:
            plt.xlim([start_time, end_time])

    plt.legend()  
    plt.tight_layout()  # Ajusta el layout para evitar la superposición de subplots
    plt.show()
    
def segment_signal(senal):
    # Calcula el número de segmentos de la señal
    num_segmentos = -(-len(senal) // segment_length_samples)  

    segmentos = []

    for i in range(num_segmentos):
        inicio = i * segment_length_samples
        fin = inicio + segment_length_samples

        # Extrae el segmento o añade ceros si es el último segmento y es más corto
        if fin <= len(senal):
            segmento = senal[inicio:fin]
        else:
            # Añade padding con ceros si el segmento es más corto que segment_length_samples
            segmento = np.pad(senal[inicio:], (0, segment_length_samples - (len(senal) - inicio)), 'constant')

        segmentos.append([segmento])

    return segmentos

def Proccesing_Subject(record_path,b,a,detectors,Start,fs,subject):
    # Leer el archivo CSV
    record = pd.read_csv(record_path, sep='\s*,\s*|\s+', skiprows=3+Start, header=None, engine='python')
    
    x= np.array(record.iloc[:, 1])
    y= np.array(record.iloc[:, 2])
    z= np.array(record.iloc[:, 3])
    
    if subject == 'CP1-38':
        y=y*-1

    #Resampling
    if fs==256:
        x = signal.resample(x, len(x)*2)
        y = signal.resample(y, len(y)*2)
        z = signal.resample(z, len(z)*2)



    #Preprocesamiento de los datos.
    #CAR
    all_signals = np.column_stack((x, y, z))
    average_signal = np.mean(all_signals, axis=1)
    
    x_CAR = x - average_signal
    y_CAR = y - average_signal
    z_CAR = z - average_signal
   
    #Z-score Normalization
    #X
    mean_x=np.mean(x_CAR)
    std_x=np.std(x_CAR)
    x=(x_CAR-mean_x)/std_x

    #Y
    mean_y=np.mean(y_CAR)
    std_y=np.std(y_CAR)
    y=(y_CAR-mean_y)/std_y

    #Z
    mean_z=np.mean(z_CAR)
    std_z=np.std(z_CAR)
    z=(z_CAR-mean_z)/std_z

    # Ahora grafica el registro modificado.
    if plot_ON==1:
        plot_signals(x,y,z,[])
        plot_fft(x, 'Señal X')
        plot_fft(y, 'Señal Y')
        plot_fft(z, 'Señal Z')


    # Filtrado
    if Filter=='FIR':
        x= lfilter(b,1,x)
        y= lfilter(b,1,y)
        z= lfilter(b,1,z)
    elif Filter=='Butter':
        x= lfilter(b,a,x)
        y= lfilter(b,a,y)
        z= lfilter(b,a,z)

    # Metodo de deteccion de picos
    z_peak = detectors.engzee_detector(z)
        
    if plot_ON==1:
        # Grafica de las señales filtradas
        plot_signals(x,y,z,z_peak)
        plot_fft(x, 'Señal X')
        plot_fft(y, 'Señal Y')
        plot_fft(z, 'Señal Z')
        
    # Segmentación de las señales
    segments_x = segment_signal(x)
    segments_y = segment_signal(y)
    segments_z = segment_signal(z)

    return segments_x, segments_y, segments_z

def Start_Time_to_Seconds(Start):
    horas, minutos, segundos = map(int, str(Start).split(':'))
    total_segundos = horas * 3600 + minutos * 60 + segundos
       
    return total_segundos

def leer_csv(Csv_path):
    datos = pd.read_csv(Csv_path, sep=';')
    # Lista de sujetos de entrenamiento y prueba predefinidos
    train_subjects = ['CP-08', 'CP-30']
    test_subjects = ['CP-06', 'CP-07']

    # Filtrar los datos para obtener solo los sujetos que se van a barajar
    datos_para_barajar = datos[~datos['Patient ID'].isin(train_subjects + test_subjects)]

    # Barajar los datos
    datos_barajados = datos_para_barajar.sample(frac=1).reset_index(drop=True)

    # Dividir en conjuntos de entrenamiento, validación y prueba
    num_sujetos = len(datos_barajados)
    train_idx = int(num_sujetos * 0.7)
    valid_idx = int(num_sujetos * 0.85)

    train_datos = datos_barajados.iloc[:train_idx]
    validation_datos = datos_barajados.iloc[train_idx:valid_idx]
    test_datos = datos_barajados.iloc[valid_idx:]


    return train_datos , validation_datos , test_datos

def Dataset_from_csv(Dataframe,b,a):
    all_segments_x= []
    all_segments_y= []
    all_segments_z= []
    all_labels= []

    fs=512
    # Inicializar detector de picos
    detectors = Detectors(fs)
    # Procesamiento de los sujetos
    tic=time.time()
    for i in range(len(Dataframe)):
        tic_bucle=time.time()
        
        row=Dataframe.iloc[i]
        subject=row['Patient ID']
        Start=row['Start']

        MR=row['Moderate or greater MR']
        AS=row['Moderate or greater AS']
        TR=row['moderate or greater TR']

        flags = [ MR, AS, TR]

        fs=row['Sampling rate(Hz)']
        Start=Start_Time_to_Seconds(Start)*fs
        
        record_path = f"Proyecto/Raw_Recordings/{subject}-Raw.csv"
        print(f"Procesando sujeto {subject}")
        segments_x, segments_y, segments_z = Proccesing_Subject(record_path,b,a,detectors,Start,fs,subject)

        all_segments_x.extend(segments_x)
        all_segments_y.extend(segments_y)
        all_segments_z.extend(segments_z)
        all_labels.extend([flags] * len(segments_x))
   
        toc_bucle=time.time()
        print(f"Procesado sujeto {subject} , tiempo: {round(toc_bucle-tic_bucle,4)} s\n")

    toc=time.time()
    print(f"Tiempo total de procesamiento: {round(toc-tic,4)} s")

    return all_segments_x,all_segments_y,all_segments_z, all_labels

def plot_fft(signal, title):
    # Aplica la ventana de Blackman
    window = np.blackman(len(signal))
    signal_windowed = signal * window

    N = len(signal_windowed)
    T = 1.0 / fs

    yf = fft(signal_windowed)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    amplitud = 2.0/N * np.abs(yf[:N//2])

    plt.figure(figsize=(12, 6))
    plt.plot(xf, amplitud)
    plt.title(f'FFT de {title} con Ventana de Blackman')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.xlim(-2, 40)  
    plt.show()
#-------------------------MAIN-------------------------#
# Asigna memoria de GPU
assign_gpu_memory(gpu_Use)

# CSV con los sujetos
Csv_path = 'Proyecto/Summary_Pub_Deidentified.csv'
train_data , validation_data , test_data = leer_csv(Csv_path)

# Configuracion del filtro
if Filter == 'FIR':
    b, a = Filter_FIR_Configuration()
elif Filter == 'Butter':
    b, a = Filter_Butter_Configuration()
else:
    print('Error en el tipo de filtro')
    exit()

# Crear dataset
print('\nCreando Arrays de Entrenamiento...\n')
train_segments_x,train_segments_y,train_segments_z, train_labels= Dataset_from_csv(train_data,b,a)
print('\nCreando Arrays de Validacion...\n')
val_segments_x,val_segments_y,val_segments_z, val_labels= Dataset_from_csv(validation_data,b,a)
print('\nCreando Arrays de Prueba...\n')
test_segments_x,test_segments_y,test_segments_z, test_labels= Dataset_from_csv(test_data,b,a)

#Guardar Arrays
np.save('Proyecto/train_segments_x.npy',train_segments_x)
np.save('Proyecto/train_segments_y.npy',train_segments_y)
np.save('Proyecto/train_segments_z.npy',train_segments_z)
np.save('Proyecto/train_labels.npy',train_labels)

np.save('Proyecto/val_segments_x.npy',val_segments_x)
np.save('Proyecto/val_segments_y.npy',val_segments_y)
np.save('Proyecto/val_segments_z.npy',val_segments_z)
np.save('Proyecto/val_labels.npy',val_labels)

np.save('Proyecto/test_segments_x.npy',test_segments_x)
np.save('Proyecto/test_segments_y.npy',test_segments_y)
np.save('Proyecto/test_segments_z.npy',test_segments_z)
np.save('Proyecto/test_labels.npy',test_labels)


