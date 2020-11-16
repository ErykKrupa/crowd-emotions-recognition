import matplotlib.pyplot as plt
from keras.callbacks import History

from config.config import TRAIN_LINE_STYLE, VALIDATION_LINE_STYLE


def visualize(history: History):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, TRAIN_LINE_STYLE, label='Training accuracy')
    plt.plot(epochs, val_acc, VALIDATION_LINE_STYLE, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, TRAIN_LINE_STYLE, label='Training loss')
    plt.plot(epochs, val_loss, VALIDATION_LINE_STYLE, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
