import multiprocessing

from manager import Manager
import os
import logging
from logging import config as logging_config, Logger

config_file_path = os.path.abspath(os.path.dirname(__file__))
LOGGING_CONFIG = os.path.abspath(os.path.join(config_file_path, 'log_config.ini'))


def setup_logger():
    logging_config.fileConfig(LOGGING_CONFIG, disable_existing_loggers=False)
    log: Logger = logging.getLogger('root')
    return log


def start(processId):
    manager_obj = Manager(num_batch_workers=4, num_infer_workers=1, num_post_workers=3)
    manager_obj.process()


if __name__ == '__main__':
    log = setup_logger()
    log.info('Initializing Manager')
    for i in range(2):
        p = multiprocessing.Process(target=start, args=(i,))
        p.start()
    # manager_obj = Manager(num_batch_workers=4, num_infer_workers =1, num_post_workers=3)
    # manager_obj.process()
    log.info('Program gracefully exited')


