import os
import logging


def create_logger(logs_dir, saved_fn):
    """
    Create logger to save logs during training
    Args:
        logs_dir:
        saved_fn:

    Returns:

    """
    logger_fn = 'logger_{}.txt'.format(saved_fn)
    logger_path = os.path.join(logs_dir, logger_fn)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # formatter = logging.Formatter('%(asctime)s:File %(module)s.py:Func %(funcName)s:Line %(lineno)d:%(levelname)s: %(message)s')
    formatter = logging.Formatter(
        '%(asctime)s: %(module)s.py - %(funcName)s(), at Line %(lineno)d:%(levelname)s:\n%(message)s')

    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
