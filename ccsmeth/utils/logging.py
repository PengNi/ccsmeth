import logging
import multiprocessing

log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
log_datefmt = "%Y-%m-%d %H:%M:%S"
log_formatter = logging.Formatter(log_fmt, log_datefmt)
LOG_FN = '/tmp/ccsmeth.log'


def _level_(levelname):
    if levelname == 'DEBUG':
        return logging.DEBUG
    elif levelname == 'INFO':
        return logging.INFO
    elif levelname == 'WARNING':
        return logging.WARNING
    elif levelname == 'ERROR':
        return logging.ERROR
    elif levelname == 'CRITICAL':
        return logging.CRITICAL
    else:
        # default
        return logging.INFO


def mylogger(name='', ch=True, fh=False, levelname='INFO'):
    loglevel = _level_(levelname)

    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    # handler
    if ch:
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        ch.setFormatter(log_formatter)
        logger.addHandler(ch)
    if fh:
        fh = logging.FileHandler(LOG_FN, mode='w')
        fh.setLevel(loglevel)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
    return logger


def get_mp_logger(levelname='INFO'):
    loglevel = _level_(levelname)
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(loglevel)
    return logger
