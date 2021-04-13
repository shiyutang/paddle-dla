from argparse import ArgumentParser

from config import cfg_from_file, cfg


class Trainer:
    def __init__(self):
        self.init_config()

    def init_config(self):
        parser = ArgumentParser(description='configs of DLA')
        parser.add_argument('cfg', type=str, default=None)
        parser.add_argument("--random-train", action="store_true",
                            help="not fixing random seed.")
        parser.add_argument("--visaulDL", action="store_true",
                            help="visualize training loss with visualDL.")

        args = parser.parse_args()
        print('Called with args:')
        print(args)

        assert args.cfg is not None, 'Missing cfg file'
        cfg_from_file(args.cfg)


if __name__ == '__main__':
    trainer = Trainer()
