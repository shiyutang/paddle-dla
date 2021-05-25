import pprint
import shutil
from argparse import ArgumentParser
from pathlib import Path

import paddle
from tqdm import tqdm
from visualdl import LogWriter

from config import cfg_from_file, cfg
from dataset.imagenet import imagenet, get_dataloader
from models.dla_models import get_model
from utils.logger import init_logger
from utils.metric import AverageMeter, accuracy


class Trainer:
    def __init__(self):
        self.cfg = self.init_config()
        print('Using config:')
        pprint.pprint(self.cfg)

        self.logDir = Path(self.cfg.root).joinpath('experiment/log').joinpath(self.cfg.name)
        self.logDir.mkdir(exist_ok=True)
        self.logger = init_logger(cfg, self.logDir)

        self.best_prec1 = 0

        # model
        self.model = get_model(self.cfg, num_classes=self.cfg.nbr_class, pool_size=self.cfg.Trans.crop_size // 32)
        # todo: model parallel

        # data
        self.train_loader, self.val_loader = get_dataloader(self.cfg)

        # optimizer
        self.optimizer = paddle.optimizer.Momentum(learning_rate=self.cfg.Train.lr,
                                                   parameters=self.model.parameters(),
                                                   weight_decay=self.cfg.Train.weight_decay,
                                                   momentum=self.cfg.Train.momentum)

        # self.scheduler = paddle.optimizer.lr.StepDecay(learning_rate=self.cfg.Train.lr, step_size=30,
        #                                                gamma=self.cfg.Train.step_ratio, verbose=True)

        self.scheduler = paddle.optimizer.lr.NaturalExpDecay(learning_rate=self.cfg.Train.lr, gamma=0.95, verbose=True)
        # criterion
        self.criterion = paddle.nn.CrossEntropyLoss()

        if self.cfg.Train.resume is not None:
            # load
            layer_state_dict = paddle.load(self.cfg.Train.resume[0])
            opt_state_dict = paddle.load(self.cfg.Train.resume[1])

            self.model.set_state_dict(layer_state_dict)
            self.optimizer.set_state_dict(opt_state_dict)

        # log files
        shutil.copytree('../dla/', self.logDir, dirs_exist_ok=True)

    def train(self):
        for epoch in tqdm(range(self.cfg.Train.start_epoch, self.cfg.Train.epochs),
                          desc='Training epochs'):

            prec1, prec5, loss = self.training(epoch)

            if prec1 > self.best_prec1:
                self.best_prec1 = prec1
                # save
                paddle.save(self.model.state_dict(),
                            self.logDir.joinpath("Epoch{}_prec1{.2f}_prec5{.2f}.pdparams".format(epoch, prec1, prec5)))
                paddle.save(self.optimizer.state_dict(),
                            self.logDir.joinpath("Epoch{}_prec1{.2f}_prec5{.2f}.pdopt".format(epoch, prec1, prec5)))

            self.scheduler.step()

    def training(self, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()

        for i, (input, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            output = self.model(input)
            loss = self.criterion(output, target)

            # print(output) # Tensor(shape=[256, 1000]
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.numpy()[0], input.shape[0])
            top1.update(prec1.numpy()[0], input.shape[0])
            top5.update(prec5.numpy()[0], input.shape[0])

            self.optimizer.clear_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.cfg.Log_print_freq == 0:
                self.logger.info('Epoch: [{0}][{1}/{2}]\t'
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(self.train_loader), loss=losses, top1=top1, top5=top5))

        prec1, prec5 = self.validate()

        if self.cfg.visualDL:
            with LogWriter(logdir=self.logDir) as writer:
                # 使用scalar组件记录一个标量数据
                writer.add_scalar(tag="loss", step=epoch, value=losses.avg)
                writer.add_scalar(tag="prec1", step=epoch, value=prec1)
                writer.add_scalar(tag="prec5", step=epoch, value=prec5)

        self.logger.info("Epoch {}: prec1: {} prec5: {}".format(epoch, prec1, prec5))

        return prec1, prec5, losses

    def validate(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        for i, (input, target) in enumerate(self.val_loader):
            output = self.model(input)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.numpy()[0], input.shape[0])
            top1.update(prec1.numpy()[0], input.shape[0])
            top5.update(prec5.numpy()[0], input.shape[0])

        self.logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                         .format(top1=top1, top5=top5))

        return top1.avg, top5.avg

    @staticmethod
    def init_config():
        parser = ArgumentParser(description='configs of DLA')
        parser.add_argument('--cfg', type=str, default='./cfg.yml')
        parser.add_argument("--random-train", action="store_true",
                            help="not fixing random seed.")
        parser.add_argument("--visaulDL", action="store_true",
                            help="visualize training loss with visualDL.")

        args = parser.parse_args()
        print('Called with args:')
        print(args)

        assert args.cfg is not None, 'Missing cfg file'
        cfg_from_file(args.cfg)

        return cfg


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
