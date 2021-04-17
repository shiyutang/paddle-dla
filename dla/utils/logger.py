import logging
import os


def init_logger(cfg, logdir):
    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(logdir, cfg.name + "_log.txt"))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
