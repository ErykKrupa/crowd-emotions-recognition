import matplotlib.pyplot as plt
from keras.callbacks import History

from config.config import Config


def visualize(history: History, name: str = ''):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    train_line_style = Config.get('train_line_style')
    validation_line_style = Config.get('validation_line_style')

    plt.plot(epochs, acc, train_line_style, label='Training accuracy')
    plt.plot(epochs, val_acc, validation_line_style, label='Validation accuracy')
    plt.title('Training and validation accuracy' + (' of ' + name if name != '' else ''))
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, train_line_style, label='Training loss')
    plt.plot(epochs, val_loss, validation_line_style, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
