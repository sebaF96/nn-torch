import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


"""
torchvision nos provee una coleccion de datos para fines educativos. En la vida real
vamos a tener que recolectar, preparar, formatear, etc. nuestro dataset y probablemente
ésta sea la parte mas tediosa y larga de construir y entrenar redes neuronales.

Para este ejemplo vamos a usar torchvision.datasets.MNIST (Digitos del 0 al 9 dibujados a mano)
"""

# Definimos info de entrenamiento e info para test. Transformamos las imagenes a tensor (torch.FloatTensor)
training_data = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testing_data = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Creamos datasets con la informacion descargada y transformada, otra vez uno para entrenamiento y otro para test
trainset = DataLoader(training_data, batch_size=8, shuffle=True)    # Batch size & shuffle
testset = DataLoader(testing_data, batch_size=8, shuffle=True)      # GENERALIZATION (Evitar que la red "tome atajos")

