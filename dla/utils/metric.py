import paddle


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]  # 256

    _, pred = paddle.topk(output, maxk, 1, True, True)  # 256, 5
    pred = paddle.t(pred)  # 5,256
    correct = paddle.equal(pred, paddle.expand_as(target.reshape([1, -1]), pred)).astype('float32')  # 5, 256

    res = []
    for k in topk:
        correct_k = paddle.flatten(correct[:k], start_axis=0, stop_axis=-1).sum(0)
        res.append(correct_k*(100.0 / batch_size))
    return res
