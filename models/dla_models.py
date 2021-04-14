import math

import paddle
import paddle.nn as nn


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
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2D(inplanes, planes, kernel_size=3, stride=stride, padding=dilation,
                              dilation=dilation, bias_attr=False,
                              weight_attr=paddle.framework.ParamAttr(
                                  initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. /(3*3*planes)))))
        self.bn1 = nn.BatchNorm2D(planes, weight_attr=nn.initializer.Constant(value=1),
                                  bias_attr=nn.initializer.Constant(value=0))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=dilation,
                               bias_attr=False,dilation=dilation,
                               weight_attr=paddle.framework.ParamAttr(
                                   initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. /(3*3*planes)))))
        self.bn2 = nn.BatchNorm2D(planes, weight_attr=nn.initializer.Constant(value=1),
                                  bias_attr=nn.initializer.Constant(value=0))

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(paddle.nn.Layer):
    expansion = 2

    def __init__(self, inplanes,planes,stride=1,dilation=1)
        super(Bottleneck, self).__init__()
        bottle_planes = planes//Bottleneck.expansion
        self.conv1 = nn.Conv2D(inplanes, bottle_planes, kernel_size=1, bias_attr=False,
                               weight_attr=paddle.framework.ParamAttr(
                                   initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. /bottle_planes))))
        self.bn1 = nn.BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2D(bottle_planes, bottle_planes, kernel_size=3, stride=stride,
                               padding=dilation,bias_attr=False,dilation=dilation,
                               weight_attr=paddle.framework.ParamAttr(
                                   initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. /(3*3*bottle_planes)))))
        self.bn2 = nn.BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2D(bottle_planes, planes, kernel_size=1,bias_attr=False,
                               weight_attr=paddle.framework.ParamAttr(
                                   initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. /planes))))
        self.bn3 = nn.BatchNorm(planes)
        self.relu = nn.ReLU()

    def forward(self,x,residual=None):
        if residual is None:
            residual=x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class BottleneckX(paddle.nn.Layer):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        bottle_planes = planes*BottleneckX.cardinality//32
        self.conv1 = nn.Conv2D(inplanes, bottle_planes, kernel_size=1, bias_attr=False,
                               weight_attr=paddle.framework.ParamAttr(
                                   initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. /bottle_planes))))
        self.bn1 = nn.BatchNorm2D(bottle_planes, weight_attr=nn.initializer.Constant(value=1),
                                  bias_attr=nn.initializer.Constant(value=0))
        self.conv2 = nn.Conv2D(bottle_planes,bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias_attr=False,
                               dilation=dilation,groups=BottleneckX.cardinality,
                               weight_attr=paddle.framework.ParamAttr(
                                   initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. /(3*3*bottle_planes)))))
        self.bn2 = nn.BatchNorm2D(bottle_planes, weight_attr=nn.initializer.Constant(value=1),
                                  bias_attr=nn.initializer.Constant(value=0))
        self.conv3 = nn.Conv2D(bottle_planes, planes, kernel_size=1, bias_attr=False,
                               weight_attr=paddle.framework.ParamAttr(
                                   initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. /planes))))
        self.bn3 = nn.BatchNorm2D(planes, weight_attr=nn.initializer.Constant(value=1),
                                  bias_attr=nn.initializer.Constant(value=0))
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        if residual is None:
            residual=x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Layer):
    def __init__(self, inplanes, outplanes, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2D(inplanes,outplanes, kernel_size, stride=1,
                              bias_attr=False, padding=(kernel_size-1)//2,
                              weight_attr=paddle.framework.ParamAttr(
                                  initializer=paddle.nn.initializer.Normal(
                                      mean=0.0, std=math.sqrt(2. /(kernel_size*kernel_size*outplanes)))))
        self.bn = nn.BatchNorm2D(outplanes, weight_attr=nn.initializer.Constant(value=1),
                                 bias_attr=nn.initializer.Constant(value=0))
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(paddle.concat(x, 1)) # 按行拼接
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)


class Tree(nn.Layer):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if stride > 1:
            self.downsample = nn.MaxPool2D(stride,stride=stride)
        else:
            self.downsample = None

        if in_channels == out_channels:
            self.project = None
        else:
            self.project = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1, bias_attr=False,
                          weight_attr=paddle.framework.ParamAttr(
                              initializer=paddle.nn.initializer.Normal(
                                  mean=0.0, std=math.sqrt(2. /out_channels)))),
                nn.BatchNorm2D(out_channels, weight_attr=nn.initializer.Constant(value=1),
                               bias_attr=nn.initializer.Constant(value=0)))

        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Layer):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7):
        super(DLA, self).__init__()
        self.base_layer = nn.Sequential(
            nn.Conv2D(3, channels[0], kernel_size=7, stride=1, padding=3, bias_attr=False,
                      weight_attr=paddle.framework.ParamAttr(
                          initializer=paddle.nn.initializer.Normal(
                              mean=0.0, std=math.sqrt(2. /49*channels[0])))),
            nn.BatchNorm2D(channels[0], weight_attr=nn.initializer.Constant(value=1),
                           bias_attr=nn.initializer.Constant(value=0)),
            nn.ReLU()
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)
        self.avgpool = nn.AvgPool2D(kernel_size=pool_size)
        
        self.fc = nn.Conv2D(channels[-1], num_classes, kernel_size=1, stride=1, 
                            padding=0, bias_attr=False,
                            weight_attr=paddle.framework.ParamAttr(
                                initializer=paddle.nn.initializer.Normal(
                                    mean=0.0, std=math.sqrt(2. /num_classes))))
        
        for m in self.modules:


        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes

    def _make_conv_level(self, inplanes, outplanes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2D(inplanes, outplanes, kernel_size=3,
                          stride=stride if i==0 else 1,
                          padding=dilation, bias_attr=False, dilation=dilation,
                          weight_attr=paddle.framework.ParamAttr(
                              initializer=paddle.nn.initializer.Normal(
                                  mean=0.0, std=math.sqrt(2. /(9*outplanes))))),
                nn.BatchNorm2D(outplanes, weight_attr=nn.initializer.Constant(value=1),
                               bias_attr=nn.initializer.Constant(value=0)),
                nn.ReLU()])
            inplanes = outplanes

        return nn.Sequential(*modules)


    def forward(self,x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.reshape(x.shape[0], -1)

            return x


def get_model(cfg=None, **kwargs):
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

if __name__ == '__main__':
    mnist2 = get_model(cfg)
    paddle.summary(mnist2, (64,3,224,224))
