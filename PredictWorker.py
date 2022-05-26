import queue
import time
import cv2
import numpy as np
from WorkerAbstract import IWorker
from ModelLoader import load_model


class InferWorker(IWorker):
    def __init__(self, _quit_event, input_queue, output_queue):
        super().__init__(_quit_event, input_queue, output_queue)
        self.net = None
        # self.dummy_image_count =0

    def init_model(self) -> bool:
        # load the ai model
        self.net = load_model()
        if self.net is None:
            self.log.error('Model could not be loaded')
            return False
        else:
            self.log.info('Model was loaded ...')
            return True

    def run(self):
        # This method implements batch inference
        # ------------ Model was loaded proceed to start the loop -------------
        # N.B this loop will disintegrate the batch into individual frame/image detections
        while not self.shutdown_flag.is_set() and self.net is not None:
            # query task from input queue-> get a new task from the queue
            try:
                next_task = self.task_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                self.log.debug('Inference Input Queue is empty')
                time.sleep(0.01)
                continue

            # unlikely - but taken up
            if next_task is None:
                self.log.warning('Inference Input Queue had <None> fetched, trying to fetch again')
                continue

            # incoming packet {"record_keys": one_batch_keys, "image_blob": blobs} <- one batch of images
            image_blob = next_task['image_blob']
            if type(image_blob) == str:
                if image_blob == "DUMMY":
                    self.log.debug('Dummy record was encountered in batch_queue, generating dummy infer batch size<{}>'
                                   .format(len(next_task["record_keys"])))
                    # straightaway assume an exception and send a dummy record forward
                    # push the batch on the queue
                    self.try_push_dummy(next_task)
                    continue

            detections = []
            try:
                # determine the output layer
                ln = self.net.getLayerNames()
                ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
                self.net.setInput(image_blob)
                detections = self.net.forward(ln)
            except cv2.error as ex:
                self.log.error("OpenCV net.forward failed in infer process, generating dummy infer batch for"
                               " images {}".format(len(next_task['record_keys'])))
                self.try_push_dummy(next_task)
                continue

            # if detection is successful then proceed to the next line

            if len(detections[0].shape) == 2:
                # if only 1 image sent for infer,
                detections[0] = detections[0][np.newaxis, :]
                detections[1] = detections[1][np.newaxis, :]
                detections[2] = detections[2][np.newaxis, :]

            try:
                for idx, (det1, det2, det3) in enumerate(zip(detections[0], detections[1], detections[2])):
                    record_key = next_task['record_keys'][idx]
                    detection_stacked = np.vstack(np.array([det1, det2, det3]))
                    answer = {
                        "detections": detection_stacked,
                        "record_key": record_key
                    }
                    is_pushed = False
                    while not is_pushed:
                        try:
                            self.result_queue.put(answer, block=True, timeout=0.05)
                            time.sleep(0.02)
                            is_pushed = True
                        except queue.Full:
                            time.sleep(0.02)
                            self.log.debug('Waiting for the result queue to get free...')
                            is_pushed = False
                            if self.shutdown_flag.is_set():
                                break
                    if self.shutdown_flag.is_set():
                        self.log.warning('[try_push_result] Quit event is set breaking out '
                                         'from item iterator in try_push_result')
                        break

            except Exception as ex:
                self.log.warning("Stacking and Unzipping of detections failed, generating dummy infer batch")
                self.log.error(ex)
                self.try_push_dummy(next_task)
                continue
            # self.log.info('-- Dummy Image Count <{}> --- '.format(self.dummy_image_count))
            # test by commenting above lines < by pass
            # for x in next_task["record_keys"]:
            #     self.result_queue.put(x, block=True, timeout=0.2)
            #     time.sleep(0.01)

        # Shutdown gracefully
        self.log.info('Exiting InferWorker subprocess {} xxx'.format(self.name))

    def generate_dummy_record(self, record_key) -> dict:
        return {"detections": "DUMMY", "record_key": record_key}

    def try_push_dummy(self, task={}) -> bool:
        for item in list(task["record_keys"]):
            # self.dummy_image_count += 1
            # Note :if the put method is blocking then this thread will be in wait forever
            # if the consumer for task_queue gets killed.
            is_pushed = False
            while not is_pushed:
                try:
                    self.result_queue.put(self.generate_dummy_record(item), block=True, timeout=0.05)
                    time.sleep(0.002)
                    is_pushed = True
                except queue.Full:
                    time.sleep(0.02)
                    self.log.debug('Waiting for the result queue to get free...')
                    is_pushed = False
                    if self.shutdown_flag.is_set():
                        break
            if self.shutdown_flag.is_set():
                self.log.warning('[try_push_dummy] Quit event is set breaking out '
                                 'from item iterator in try_push_dummy')
                return False
        return True
