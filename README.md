# Laboratorio-2-Procesamiento
## Introducción
En esta práctica de laboratorio se estudiaron tres herramientas fundamentales para el procesamiento digital de señales: la convolución, la correlación y la transformada de Fourier. Estas operaciones permitieron analizar el comportamiento de las señales o conjunto de señales tanto en el dominio del tiempo como en el dominio de la frecuencia. Asimismo, realizamos el análisis de una señal de electrooculografía generada con ayuda del generador biológico de señales. 
## Importación de librerias 
Para el desarrollo de esta práctica se instalaron las siguientes librerías:
```python
!pip install wfdb
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import pandas as pd
import os
from scipy.stats import norm
import seaborn as sea
from scipy.fft import fft, fftfreq
from scipy.signal import welch
```
La librería `wfdb` se utilizó para la lectura y manipulación de las señales biológicas. `Matplotlib` y `Seaborn`  se usaron para la representación gráfica y visualización de datos. `NumPy` y `Pandas` facilitaron el manejo de arreglos numéricos y estructuras de datos. El módulo `os` permitió la gestión de archivos dentro del entorno de trabajo. Asimismo, `scipy.stats.norm` se aplicó para análisis estadístico, mientras que las funciones de `scipy.fft` y `scipy.signal.welch` se usaron para la obtención de la transformada de Fourier, el cálculo de frecuencias y la estimación espectral de potencia de las señales. 

