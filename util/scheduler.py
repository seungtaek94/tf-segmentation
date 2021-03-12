import math
import tensorflow as tf
import tensorflow.keras.backend as backend
import matplotlib.pyplot as plt


class CosineAnnealingLearningRateSchedule(tf.keras.callbacks.Callback):
    # constructor
    def __init__(self, end_Epochs, steps, min_finalized_epochs, max_lr, min_lr, max_lr_decay_rate=4e-5):
        self.end_Epochs = end_Epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.steps = steps
        self.lrates = list()
        self.max_lr_decay_rate = max_lr_decay_rate

    # caculate learning rate for an epoch
    def cosine_annealing(self, epoch):
        cos_inner = (math.pi * (epoch % self.steps)) / (self.steps)
        return (self.max_lr / 2 * (math.cos(cos_inner) + 1)) + self.min_lr

    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs=None):
        if ((self.end_Epochs - epoch) > self.steps):
            # calculate learning rate
            if epoch % self.steps == 0:
                self.max_lr -= self.max_lr_decay_rate

            lr = self.cosine_annealing(epoch)
            print('\nEpoch %05d: CosineAnnealingScheduler setting learng rate to %s.' % (epoch + 1, lr))
        # 남은 epoch이 step보다 작으면 min_lr 적용.
        else:
            lr = self.min_lr

        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)


def decay(epoch, lr, lr_decay, lr_decay_step):
    if epoch < lr_decay_step[0]:
        learning_rate = lr
    elif lr_decay_step[0] <= epoch < lr_decay_step[1]:
        learning_rate = lr * lr_decay
    else:
        learning_rate = lr * (lr_decay ** 2)

    print(f'LR[EPOCH_{epoch:0>3}]: {learning_rate}')
    return learning_rate



if __name__ =="__main__":
    cosine_schedule = CosineAnnealingLearningRateSchedule(end_Epochs=120, steps=20, min_finalized_epochs=20, max_lr=2e-4, min_lr=1e-6)

    for i in range(1, 120):
        cosine_schedule.on_epoch_begin(i)

    import matplotlib.pyplot as plt

    plt.plot(cosine_schedule.lrates)
    plt.title('Cosine Annealing')
    plt.xlabel('epochs')
    plt.ylabel('learning_rate')
    plt.grid()
    plt.show()