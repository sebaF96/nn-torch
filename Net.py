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


class Net(nn.Module):   # Creamos nuestra red como una clase normal de Python que hereda de nn.Module

    def __init__(self):
        # Inicializar nn.Module
        super().__init__()

        """
        definimos nuestras "fully connected" layers igualandolas a nn.Linear(input, output).
        para terminar de entender esto, se piensa a las layers como "en el medio" de input/output para cada
        llamada a nn.Linear(i, o)
        """
        self.fc1 = nn.Linear(28 * 28, 64)   # 28x28 = tamaño de input layer. 64 = tamaño de la siguiente hidden layer.
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)    # Output layer = 10 neuronas (1 para cada digito)


    def forward(self, data):
        # Metodo que define como la data "va a fluir" por nuestra red

        """
        Una de las ventajas de Pytorch es la libertad para personalizar esta parte.
        En este caso pasamos la informacion por todas las layers en el orden que las definimos
        y para cada uno llamamos a torch.nn.functional.relu (rectified linear unit function element-wise)
        que sería nuestra Funcion de activacion. (corre en el lado de 'output' de nuestras self.fcs)
        """
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = F.relu(self.fc3(data))
        data = self.fc4(data)

        return F.log_softmax(data, dim=1)




net = Net()
print(net)
