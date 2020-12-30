import matplotlib.pyplot as plt
from keras.callbacks import History

from config.config import train_config as config


def visualize(history: History, name: str = ''):
    acc = [i * 100 for i in history.history['acc']]
    val_acc = [i * 100 for i in history.history['val_acc']]
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    train_line_style = config.get('train_line_style')
    validation_line_style = config.get('validation_line_style')

    plt.plot(epochs, acc, train_line_style, label='Training accuracy')
    plt.plot(epochs, val_acc, validation_line_style, label='Validation accuracy')
    plt.title('Training and validation accuracy' + (' of ' + name if name != '' else ''))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy [%]')

    plt.legend()
    plt.grid(which='both', axis='both')
    plt.minorticks_on()
    plt.grid(which='minor', axis='both', alpha=0.2)

    plt.figure()

    plt.plot(epochs, loss, train_line_style, label='Training loss')
    plt.plot(epochs, val_loss, validation_line_style, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.grid(which='both', axis='both')
    plt.minorticks_on()
    plt.grid(which='minor', axis='both', alpha=0.2)

    plt.show()
