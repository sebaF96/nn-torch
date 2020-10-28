# Redes neuronales - PyTorch

En este repositorio se muestra con un simple ejemplo como crear y entrenar redes neuronales usando PyTorch.

Creamos [una red](https://github.com/sebaF96/nn-torch/blob/master/Net.py "una red") para reconocimiento de digitos escritos a mano, para ello usamos un dataset de imagenes provisto por torchvision. Se entrena la red y se evalua su porcentaje de acierto (97% promedio) con un dataset de prueba distinto al de entrenamiento.

Luego, se agarran 8 imagenes al azar y la red predice de que digito se trata, mientras que nos muestra la imagen para corroborar su presicion.

El archivo[ experiment.py](https://github.com/sebaF96/nn-torch/blob/master/experiment.py " experiment.py") contiene algunos metodos de prueba realizados sobre los datasets para terminar de entender como deberia estar organizada nuestra informacion antes de comenzar a entrenar.
