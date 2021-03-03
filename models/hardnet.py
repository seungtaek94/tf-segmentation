import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, UpSampling2D, Conv2D, BatchNormalization, ReLU, AveragePooling2D


class TransitionUp(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, skip=None, concat=True, **kwargs):
        out = UpSampling2D(interpolation='bilinear')(inputs)

        if concat:
            skip = tf.dtypes.cast(skip, tf.float32)
            out = tf.concat([out, skip], 3)

        return out


class ConvLayer(Layer):
    def __init__(self, out_channels, kernel=3, stride=1, **kwargs):
        super().__init__(**kwargs)

        self.conv = Conv2D(out_channels,
                           kernel_size=kernel,
                           strides=stride,
                           padding=[[0, 0], [kernel // 2, kernel // 2], [kernel // 2, kernel // 2], [0, 0]],
                           use_bias=False)
        self.norm = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.relu(out)

        return out


class HardBlock(Layer):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, **kwargs):
        super().__init__(**kwargs)
        self.keepBase = keepBase
        self.links = []
        self.out_channels = 0
        self.layers = []

        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            self.layers.append(ConvLayer(outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

    def get_out_ch(self):
        return self.out_channels

    def call(self, inputs, **kwargs):
        layers_ = [inputs]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                inputs = tf.concat(tin, 3)
            else:
                inputs = tin[0]
            out = self.layers[layer](inputs)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = tf.concat(out_, 3)
        return out


class HardNet(Model):
    def __init__(self, n_classes=19, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_ch = [16, 24, 32, 48]
        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)

        self.shortcut_layers = []

        self.base = []
        self.base.append(ConvLayer(out_channels=first_ch[0], kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[1], kernel=3))
        self.base.append(ConvLayer(first_ch[2], kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[3], kernel=3))

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HardBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(AveragePooling2D(strides=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = []
        self.denseBlocksUp = []
        self.conv1x1_up = []

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp())
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HardBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            blk.trainable = False
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()

        self.finalConv = Conv2D(n_classes, kernel_size=1, strides=1, padding='valid')

    def call(self, x, training=None, mask=None):
        skip_connections = []

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = UpSampling2D(size=(4, 4), interpolation='bilinear')(out)
        return out

if __name__ == "__main__":
    model = HardNet(n_classes=5)

    x = tf.random.uniform([2, 512, 1024, 3], 0, 1)
    y = model(x)
    print(y.shape)

    model.summary()