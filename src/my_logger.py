import logging
import sys
from typing import Union


def setup_logger(logger: Union[str, logging.Logger]) -> logging.Logger:
    assert type(logger) in [str, logging.Logger], "Provided logger not "\
                                                      "correct type"
    if type(logger) is str:
        logger = logging.getLogger(logger)

    line = '-' * 80
    fmt = f'{line}\n\n%(asctime)s    %(threadName)s    ' \
          f'%(levelname)s:    %(message)s'
    formatter = logging.Formatter(fmt=fmt)
    cli_err = logging.StreamHandler(stream=sys.stderr)
    cli_err.setLevel(logging.ERROR)
    cli_err.setFormatter(formatter)

    class StdOutFilter(logging.Filter):
        def filter(self, record: logging.LogRecord):
            return record.levelno in (logging.DEBUG,
                                      logging.INFO, logging.WARNING)

    cli_out = logging.StreamHandler(stream=sys.stdout)
    cli_out.setLevel(logging.DEBUG)
    cli_out.setFormatter(formatter)
    cli_out.addFilter(StdOutFilter())

    file_log = logging.FileHandler(filename='log.txt', mode='w+')
    file_log.setLevel(logging.DEBUG)
    file_log.setFormatter(formatter)

    root_logger = logging.root
    root_logger.setLevel(logging.ERROR)
    root_logger.addHandler(cli_err)

    # logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(cli_out)

    num_args = len(sys.argv) - 1
    if num_args == 0 or (num_args > 0 and '--dev' not in sys.argv):
        root_logger.addHandler(file_log)

    # logger.debug('Debug test')
    # logger.info('Info test')
    # logger.warning('Warning test')
    # logger.error('Error test')
    # logger.critical('Critical test')

    return logger