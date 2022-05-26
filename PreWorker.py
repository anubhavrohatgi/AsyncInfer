import multiprocessing
import queue
import sys
import time
import logging
from abc import ABC

import cv2
import os
import numpy as np
from WorkerAbstract import IWorker

# InputResolution = eval(os.environ.get("InputResolution"))
InputResolution = 416
BATCH_SIZE = 8  # eval(os.environ.get("BATCH_SIZE"))


class BatchWorker(IWorker):
    def __init__(self, _quit_event, input_queue, output_queue):
        super().__init__(_quit_event, input_queue, output_queue)
        self.batch_counter = 0

    def run(self):
        # This method implements batch maker
        # set the lists -> output on result_queue is dict {batch_key : batch_value}
        one_batch_values = []
        one_batch_keys = []
        while not self.shutdown_flag.is_set():
            # before querying the task_queue for next task, check if the batch is complete
            # if batch complete extract the blob and push it to output queue.
            # we should not limit the output queue size

            # in the next statement we check the batch size, if done then push this batch to result queue
            # in case if result queue is full then continue else reset the batch and fall to the next line of code
            # this way we are not blocking the process to check for shutdown flag
            if len(one_batch_values) == BATCH_SIZE:
                # push the batch on the queue
                try:
                    self.result_queue.put(self.blob_maker(one_batch_values, one_batch_keys), block=True, timeout=0.1)
                    time.sleep(0.01)  # custom delay for system -> 10ms for lock free
                    self.log.debug('Pushed batch_id <{}> to queue **'.format(self.batch_counter))

                    if self.batch_counter == sys.maxsize:
                         self.batch_counter = 0

                    self.batch_counter += 1

                    # reset lists and move on to build next batch
                    one_batch_values = []
                    one_batch_keys = []
                except queue.Full:
                    self.log.warning("Pre-Process result queue is full, cant push batch ...")
                    continue

            # query task from input queue-> get a new task from the queue
            try:
                next_task = self.task_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                self.log.debug('PreProcess Input Queue is empty')
                # push whatever was in the batches since the input queue has been empty for longer time
                if len(one_batch_values) > 0:
                    # push the batch on the queue
                    try:
                        self.result_queue.put(self.blob_maker(one_batch_values, one_batch_keys), block=True,
                                              timeout=0.1)
                        self.log.info("Pre-Process pushed incomplete batch ...")
                        time.sleep(0.01)  # custom delay for system -> 10ms for lock free

                        self.log.debug('Pushed batch_id <{}> to queue **'.format(self.batch_counter))

                        if self.batch_counter == sys.maxsize:
                            self.batch_counter = 0

                        self.batch_counter += 1

                        # reset lists and move on to build next batch
                        one_batch_values = []
                        one_batch_keys = []
                    except queue.Full:
                        self.log.warning("Pre-Process result queue is full, cant push batch looping back...")
                continue

            # unlikely - but taken up
            if next_task is None:
                self.log.warning('PreProcess Input Queue had <None> fetched, trying to fetch again')
                continue

            # Below code just adds one image and corresponding hashed key to batch
            h, w = 0, 0
            image_path = next_task['frame_url']
            record_key = ["-".join([next_task['userId'], next_task['videoId'], next_task['frameId']]), (h, w)]

            try:
                frame = cv2.imread(image_path)
                if frame is None:
                    raise cv2.error('Image reading error')

                record_key[1] = frame.shape[:2]

                # add to batch
                one_batch_keys.append(record_key)
                one_batch_values.append(frame)

            except cv2.error as ex:
                self.log.error("Could not read image from {} with error {}".format(image_path, str(ex)))
                # as an exception handle we add a dummy record to the batch
                dummy = self.generate_dummy_record(record_key)
                one_batch_keys.append(dummy["key"])
                one_batch_values.append(dummy["image"])

        # Shutdown gracefully
        self.log.info('Exiting BatchWorker subprocess {} xxx'.format(self.name))

    def generate_dummy_record(self, record_key) -> dict:
        dummy_record_key = record_key
        # dummy_record_key[1] = (InputResolution, InputResolution)
        return {
            "key": dummy_record_key,
            "image": np.zeros((InputResolution, InputResolution, 3), dtype=np.uint8)
        }

    def blob_maker(self, _one_batch_values, _one_batch_keys) -> dict:
        try:
            blobs = cv2.dnn.blobFromImages(_one_batch_values, 1 / 255.0,
                                           (InputResolution, InputResolution), swapRB=True,
                                           crop=False)
        except cv2.error as ex:
            self.log.warning("Error in Blob creation {}, preparing dummy blobs of len <{}>"
                             .format(ex, len(_one_batch_keys)))
            # generating a dummy blob
            return {"record_keys": _one_batch_keys, "image_blob": "DUMMY"}

        return {"record_keys": _one_batch_keys, "image_blob": blobs}

