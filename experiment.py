from Net import trainset
from matplotlib import pyplot


def print_dataset():
    # Imprimimos el primer batch del trainset (8 imagenes, cada una con su etiqueta)
    for data in trainset:
        print(data)
        break

    # img_tensor es el tensor de la primer imagen del primer batch, label es su respectiva etiqueta
    input()
    img_tensor, label = data[0][0], data[1][0]
    print(img_tensor)
    print(label)

    # mostramos la imagen
    input()
    pyplot.imshow(img_tensor.view(28, 28))
    pyplot.show()


# Que tan balanceado esta nuestro dataset?

def show_balance():
    total = 0
    counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    for data in trainset:
        imgs, labels = data

        for lab in labels:
            counter[int(lab)] += 1
            total += 1

    for label, quantity in counter.items():
        print(f"El numero {label} aparece {quantity} veces ({str(quantity/total * 100)[:4]} %)")


if __name__ == '__main__':
    #  print_dataset()
    #  show_balance()

    pass
