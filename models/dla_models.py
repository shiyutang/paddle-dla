import paddle


class Mnist(paddle.nn.Layer):
    def __init__(self):
        super(Mnist, self).__init__()

        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(784, 512)
        self.linear_2 = paddle.nn.Linear(512, 10)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, inputs):
        y = self.flatten(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)

        return y


class BasicBlock(paddle.nn.Layer):
    def __init__(self, inplanes, planes, stride=1, dialation=1):
        super(BasicBlock, self).__init__()
        self.conv = paddle.nn.con


def get_model(cfg):
    if cfg.Net.arch == 'dla34':
        model = DLA([1, 1, 1, 2, 2, 1],
                    [16, 32, 64, 128, 256, 512],
                    block=BasicBlock, **kwargs)
    elif cfg.Net.arch == 'dla60':
        Bottleneck.expansion = 2
        model = DLA([1, 1, 1, 2, 3, 1],
                    [16, 32, 128, 256, 512, 1024],
                    block=Bottleneck, **kwargs)
    elif cfg.Net.arch == 'dla60x':
        BottleneckX.expansion = 2
        model = DLA([1, 1, 1, 2, 3, 1],
                    [16, 32, 128, 256, 512, 1024],
                    block=BottleneckX, **kwargs)
    elif cfg.Net.arch == 'dla102x':  # DLA-X-102
        BottleneckX.expansion = 2
        model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                    block=BottleneckX, residual_root=True, **kwargs)
    elif cfg.Net.arch == 'dla102x2':  # DLA-X-102 64
        BottleneckX.cardinality = 64
        model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                    block=BottleneckX, residual_root=True, **kwargs)
    elif cfg.Net.arch == 'dla169':
        Bottleneck.expansion = 2
        model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                    block=Bottleneck, residual_root=True, **kwargs)
    else:
        raise '{} is not implemented'.format(cfg.Net.arch)

    if cfg.Train.pretrained is not None:
        model.load_pretrained_model(cfg.Train.pretrained, cfg.Net.arch)

    return model
