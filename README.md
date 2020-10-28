# Redes neuronales - PyTorch

### Overview
En este repositorio se muestra con un simple ejemplo como crear y entrenar redes neuronales usando PyTorch.

Creamos [una red](https://github.com/sebaF96/nn-torch/blob/master/Net.py "una red") para reconocimiento de digitos escritos a mano, para ello usamos un dataset de imagenes provisto por torchvision. Se entrena la red y se evalua su porcentaje de acierto (97% promedio) con un dataset de prueba distinto al de entrenamiento.

Luego, se agarran 8 imagenes al azar y la red predice de que digito se trata, mientras que nos muestra la imagen para corroborar su presicion.

El archivo[ experiment.py](https://github.com/sebaF96/nn-torch/blob/master/experiment.py " experiment.py") contiene algunos metodos de prueba realizados sobre los datasets para terminar de entender como deberia estar organizada nuestra informacion antes de comenzar a entrenar.

### Que sigue?
Lo proximo a hacer seria guardar el estado de la red en el disco cuando termina su ejecucion, y cargarlo cuando comienza de nuevo. Asi no se va a olvidar de lo que va aprendiendo.

Tambien estaría bueno analizar con mayor detenimiento el output de la red. De esta manera podremos ver que tan segura esta al hacer predicciones correctas, y que tan lejos esta de acertar cuando se equivoca.

Luego podemos ver hasta que punto podemos mejorar lo dicho anteriormente.

### Como probar localmente?
Se ha provisto un archivo de las dependencias necesarias para ejecutar la red. Se debe clonar el repositorio e instalar las dependencias en un entorno virtual. Abajo se detallan los pasos a seguir

```shell
$ git clone https://github.com/sebaF96/nn-torch.git nn-torch
```

```shell
$ cd nn-torch && python3 -m venv venv
```

```shell
$ source venv/bin/activate
```

```shell
(venv) $ pip install -r requirements.txt
```

```shell
(venv) $ python3 Net.py
```
