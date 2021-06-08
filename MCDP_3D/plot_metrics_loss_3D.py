import matplotlib.pyplot as plt


def plot_crossentropy_loss_dice_coefficient_3D(history, name):
    fig = plt.figure(num=None, figsize=(20, 6), dpi=150, facecolor='w', edgecolor='k',
                     tight_layout=True)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['dice_coefficient'], color='b', label="train_coefficient")
    plt.plot(history.history['val_dice_coefficient'], color='r', label="validation_coefficient")
    plt.title('dice coefficient')
    plt.ylabel('coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('crossentropy_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)
