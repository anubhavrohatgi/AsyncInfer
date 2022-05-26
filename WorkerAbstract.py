import logging
import multiprocessing
from abc import ABC, abstractmethod


class IWorker(multiprocessing.Process, ABC):
    def __init__(self, _quit_event, input_queue, output_queue):
        multiprocessing.Process.__init__(self)

        # Register Logger
        self.log = logging.getLogger('workers')

        # Task queues and events
        self.task_queue = input_queue
        self.result_queue = output_queue
        self.shutdown_flag = _quit_event

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def generate_dummy_record(self, record_key) -> dict:
        pass
