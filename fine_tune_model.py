import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
import time, logging
import argparse

from util.dataloader.customRoad import get_dataset

'''

from util.func import *

from util.losses import *
'''

#from models.pspunet import Unet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project-base-dir', type=str, default='/home/seungtaek/project/segmentation',
                        help='project base directory')

    # Training
    parser.add_argument('--batch-size', type=int, default=8, help='training batch size per device (CPU/GPU).')
    # parser.add_argument('--num-gpus', type=int, default=2, help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='unet',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('--num-epochs', type=int, default=200, help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate. default is 0.1.')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-step', type=str, default='150,180',
                        help='epochs at which learning rate decays. default is 80,120')
    parser.add_argument('--loss', type=str, default='SCCE', help='SCCE, BCE, focal, dice, soft_dice')

    parser.add_argument('--optimizer', type=str, default='adam', help='sgd, nag')
    parser.add_argument('--model-name', type=str, default='pspunet', help='model name')

    # Fine tune with pre-trained model
    parser.add_argument('--fine-tune', type=bool, default=True, help='')
    parser.add_argument('--pre-trained-model', type=str,
                        default='/exp/pspunetv0_20210308120957_c2_SCCE_0.0001/save_model/000',
                        help='load pre-trained model')

    # Dataset
    parser.add_argument('--dataset-path', type=str, default='/home/seungtaek/ssd1/datasets/road_seg_v1',
                        help='dataset path')
    parser.add_argument('--image-height', type=int, default=512, help='image height')
    parser.add_argument('--image-width', type=int, default=1024, help='image width')

    return parser.parse_args()

'''
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
        self.img = img
        self.gt = gt
        self.plot_dir = plot_dir

    def on_epoch_end(self, epoch, logs=None):
        show_predictions(self.img, self.gt, epoch, self.plot_dir)
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))
'''

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

    num_classes = 2

    if not os.path.isdir('./log'):
        os.mkdir('./log')

    model_name = opt.model_name
    filehandler = logging.FileHandler(f'./log/{model_name}_c{num_classes}_{opt.loss}_{timePrefix}.log')
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)
    logger.info(f'#Classes: {num_classes}')

    if not os.path.isdir('./exp'):
        os.mkdir('./exp')

    exp_base_dir = f'./exp/{opt.model_name}_{timePrefix}_c{num_classes}_{opt.loss}_{opt.lr}_f-{opt.fine_tune}'
    checkpoints_dir = exp_base_dir + '/checkpoints'
    plot_dir = exp_base_dir + '/plot'

    os.mkdir(exp_base_dir)
    os.mkdir(checkpoints_dir)
    os.mkdir(plot_dir)

    logger.info(f'Exp Dir: {exp_base_dir}')

    dataset = get_dataset(opt.dataset_path, opt.image_height, opt.image_width, batch_size=opt.batch_size)

    for image, mask in dataset['train'].take(1):
        sample_image, sample_mask = image, mask

    print("Train Shape: ", dataset['train'])
    print("Val Shape:", dataset['val'])

    with strategy.scope():
        if opt.fine_tune:
            pre_trained_path = f'{opt.project_base_dir}{opt.pre_trained_model}'

            inp = tf.keras.Input(shape=(512, 1024, 3))
            base_model = tf.keras.models.load_model(pre_trained_path)
            #base_model.layers[-1].summary()

            layers = [l for l in base_model.layers[:-1]]

            print(layers)

            classifier = tf.keras.layers.Conv2D(10, 1, padding='same', activation='softmax', name='classifier')(layers[-1].output)


            for layer in layers:
                layer.trainable = False

            x = layers[-1](classifier)

            result = tf.keras.Model(inputs=layers[0].input, outputs=x)

            result.summary()

            '''
            out = base_model(inp)
            print(type(base_model))
            model = tf.keras.Model(inputs=inp, outputs=out)
            print(type(model))
            model.summary()


            #print(model.layers[-1].layers)

            out = tf.keras.layers.Conv2D(5, 1, padding='same', activation='softmax')(model.layers[-1].output)
            model2 = tf.keras.Model(inputs=inp, outputs=out)
            model2.summary()
            #print(base_model.layers)
            '''

            '''
            for layer in base_model.layers[:-1]:
                layer.trainable = False

            for layer in base_model.layers:
                sp = '     '[len(layer.name) - 9:]
                print(layer.name, sp, layer.trainable)
            '''
        #else:
            #model = Unet(opt.image_height, opt.image_width, num_classes)
