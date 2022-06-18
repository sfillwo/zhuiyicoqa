import logging
import logging.handlers
import os

from config import args

def config_logger(name, level, f=os.path.join(args.output_dir, 'train.log')):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    LEVEL = getattr(logging,level.upper(), None)
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(LEVEL)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if f:
        fh = logging.handlers.RotatingFileHandler(f, maxBytes=1024*1024*10, backupCount=3, encoding='utf-8')
        fh.setLevel(LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(LEVEL)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger