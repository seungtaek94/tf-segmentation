import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf

import time, logging

import argparse
from models.pspunet import pspunet
from util.dataloader.customRoad import get_dataset
from util.func import display_sample, create_mask
from util.losses import get_loss
from util.scheduler import decay, CosineAnnealingScheduler
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--batch-size', type=int, default=8, help='training batch size per device (CPU/GPU).')
    #parser.add_argument('--num-gpus', type=int, default=2, help='number of gpus to use.')
    parser.add_argument('--num-epochs', type=int, default=120, help='number of training epochs.')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. default is 0.1.')

    # linear lr decay
    parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-step', type=str, default='80,100',
                        help='epochs at which learning rate decays. default is 40,60')

    # Cosine Annealing
    parser.add_argument('--use-ca', type=bool, default=False, help='Use Cosine Annealing')
    parser.add_argument('--ca-max-lr', type=float, default=2e-4, help='each steps initial learning rate')
    parser.add_argument('--ca-min-lr', type=float, default=1e-6, help='each steps final learning rate')
    parser.add_argument('--ca-step', type=int, default=20, help='Annealing Step')

    # Loss
    parser.add_argument('--loss', type=str, default='SCCE', help='SCCE, BCE, focal, dice, soft_dice')

    parser.add_argument('--optimizer', type=str, default='adam', help='sgd, nag')
    parser.add_argument('--model-name', type=str, default='pspunet',
                        help='model-name should be same with <models/{name}.py>')

    # Dataset
    parser.add_argument('--dataset', type=str, default='customRoad', help='cityscapes, vitas, customRoad')
    parser.add_argument('--dataset-path', type=str, default='/home/seungtaek/ssd1/datasets/road_seg_v1',
                        help='dataset path')
    parser.add_argument('--image-height', type=int , default=512, help='image height')
    parser.add_argument('--image-width', type=int, default=512, help='image width')


    return parser.parse_args()



def show_predictions(img, gt, epoch, plot_dir):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    pred_mask = model.predict(img)
    display_sample([img[0], gt[0], create_mask(pred_mask[0])], epoch, plot_dir)


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, img, gt, plot_dir):
        super(DisplayCallback, self).__init__()
        self.img=img
        self.gt=gt
        self.plot_dir=plot_dir

    def on_epoch_end(self, epoch, logs=None):
        show_predictions(self.img, self.gt, epoch, self.plot_dir)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, base_dir):
        super(SaveCallback, self).__init__()
        self.base_dir = base_dir
        self.save_dir = f'{self.base_dir}/save_model/'
        os.mkdir(self.save_dir)

    def on_epoch_end(self, epoch, logs=None):
        model_save_dir = os.path.join(self.save_dir, f'{epoch:0>3}')
        model.save(model_save_dir)


if __name__ == '__main__':
    opt = parse_args()

    strategy = tf.distribute.MirroredStrategy()

    now = time.localtime(time.time())
    timePrefix = f'{now.tm_year}{now.tm_mon:0>2}{now.tm_mday:0>2}{now.tm_hour:0>2}{now.tm_min:0>2}{now.tm_sec:0>2}'

    ignore_classes = [-1]

    num_classes = 2

    if not os.path.isdir('./log'):
        os.mkdir('./log')

    model_name = opt.model_name
    filehandler = logging.FileHandler(f'./log/{opt.model_name}_{timePrefix}_c{num_classes}_{opt.loss}_{opt.lr}_{opt.dataset}.log')
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)
    logger.info(f'#Classes: {num_classes}')

    if not os.path.isdir('./exp'):
        os.mkdir('./exp')

    exp_base_dir = f'./exp/{opt.model_name}_{timePrefix}_c{num_classes}_{opt.loss}_{opt.lr}_{opt.dataset}'
    checkpoints_dir = exp_base_dir + '/checkpoints'
    plot_dir = exp_base_dir + '/plot'

    os.mkdir(exp_base_dir)
    os.mkdir(checkpoints_dir)
    os.mkdir(plot_dir)

    copyfile(f'./models/{opt.model_name}.py', f'{exp_base_dir}/{opt.model_name}.py')

    logger.info(f'Exp Dir: {exp_base_dir}')

    dataset = get_dataset(opt.dataset_path, opt.image_height, opt.image_width, batch_size=opt.batch_size)

    for image, mask in dataset['train'].take(1):
        sample_image, sample_mask = image, mask

    print("Train Shape: ", dataset['train'])
    print("Val Shape:", dataset['val'])

    with strategy.scope():
        #model = Unet(opt.image_height, opt.image_width, num_classes)
        model = pspunet((opt.image_height, opt.image_width, 3), num_classes)

        train_ds = strategy.experimental_distribute_dataset(dataset['train'])
        val_ds = strategy.experimental_distribute_dataset(dataset['val'])

        loss_fn = get_loss(opt.loss)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=opt.lr),
            loss=loss_fn,
            metrics=['accuracy']
        )

    cp_prefix = os.path.join(checkpoints_dir, "ckpt_{epoch:0>3}.ckpt")
    tensorboar_logs_dir = f'{exp_base_dir}/tblogs'
    logger.info(f'Tensorboard: /home/seungtaek/project/segmentation/{tensorboar_logs_dir}')

    if opt.use_ca:
        lr_scheduler = CosineAnnealingScheduler(T_max=opt.ca_step, eta_max=opt.ca_max_lr, eta_min=opt.ca_min_lr)
    else:
        lr_decay_step = list(map(int, opt.lr_decay_step.split(',')))
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: decay(epoch, opt.lr, opt.lr_decay, lr_decay_step))

    callbacks = [
        DisplayCallback(sample_image, sample_mask, plot_dir),
        SaveCallback(exp_base_dir),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboar_logs_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath=cp_prefix),
        lr_scheduler
    ]

    '''
    steps_per_epoch = 2975 // opt.batch_size
    val_step = 500 // opt.batch_size
    '''

    #steps_per_epoch = 18000 // opt.batch_size
    #val_step = 2000 // opt.batch_size

    steps_per_epoch = 500
    val_step = 116

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=opt.num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_step,
        callbacks=callbacks
    )

