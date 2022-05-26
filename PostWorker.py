import queue
import time

import cv2
import numpy as np
from WorkerAbstract import IWorker
from ModelLoader import load_classes

DETECTION_THRESHOLD = 0.5  # eval(os.environ.get("DETECTION_THRESHOLD"))
NMS_THRESHOLD = 0.35


class PostPWorker(IWorker):
    def __init__(self, _quit_event, input_queue, output_queue):
        super().__init__(_quit_event, input_queue, output_queue)
        self.class_names = []
        self.allowed_classes = ["car", "motorbike", "truck", "boat", "person", "bicycle", "bus"]
        # self.dummy_image_count =0

    def init_model_classes(self):
        # load the ai model classes
        self.class_names = load_classes()
        if bool(self.class_names):
            self.log.info('Model class_names list was loaded ...')
            return True
        else:
            self.log.error('Model class_names list could not be loaded or was empty')
            return False

    def run(self):
        # This method implements nms and inference extractions/normalization scaling etc.
        result_answer = {}
        while not self.shutdown_flag.is_set() and bool(self.class_names):
            # check if there is any stuck answer, if yes then produce it
            if bool(result_answer):
                try:
                    self.result_queue.put(result_answer, block=True, timeout=0.05)
                    result_answer = {}  # Reset
                    time.sleep(0.01)
                except queue.Full:
                    time.sleep(0.01)
                    self.log.debug('Waiting for the result queue to get free...')
                    continue

            # query task from input queue-> get a new task from the queue
            try:
                next_task = self.task_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                self.log.debug('Post Process Queue is empty')
                time.sleep(0.01)
                continue

            # unlikely - but taken up
            if next_task is None:
                self.log.warning('Post Process Input Queue had <None> fetched, trying to fetch again')
                continue

            # incoming packet {"detections": "DUMMY", "record_key": record_key} <- detection / image
            detections = next_task['detections']
            record_key, (h, w) = next_task['record_key']

            if type(detections) == str or ((h == 0) and (w == 0)):
                if detections == "DUMMY" or ((h == 0) and (w == 0)):
                    self.log.debug('Dummy record was encountered in input queue, generating dummy post worker'
                                   ' record<{}>'.format(next_task["record_key"]))
                    # straightaway assume an exception and send a dummy record forward
                    # push the batch on the queue, this will be produced at the beginning of next iteration
                    result_answer = self.generate_dummy_record(record_key)
                    continue

            boxes_list = []
            boxes_list = self.detections_to_boxes(detections, h, w)

            result_answer = {
                "record_key": record_key,
                "boxes_list": boxes_list
            }

            # self.log.info('-- Dummy Image Count <{}> --- '.format(self.dummy_image_count))
            # test by commenting above lines < by pass
            # for x in next_task["record_key"]:
            #     self.result_queue.put(x, block=True, timeout=0.2)
            #     time.sleep(0.01)

        # Shutdown gracefully
        self.log.info('Exiting PostWorker subprocess {} xxx'.format(self.name))

    def generate_dummy_record(self, record_key) -> dict:
        return {"boxes_list": [], "record_key": record_key}

    def detections_to_boxes(self, detections, h, w) -> list:
        # box postprocessing and thresholding
        results_list = []
        try:
            scores = detections[:, 5:]
            class_id = np.argmax(scores, axis=1)
            n_detections = scores.shape[0]
            Iax = np.ogrid[:n_detections]
            confidence = scores[Iax, class_id]
            boxes = detections[confidence > DETECTION_THRESHOLD][:, :4]
            confidences = confidence[confidence > DETECTION_THRESHOLD]
            class_ids = class_id[confidence > DETECTION_THRESHOLD]
            boxes = boxes * np.array([w, h, w, h])

            centerX = boxes[:, 0].astype(int)
            centerY = boxes[:, 1].astype(int)
            width = boxes[:, 2].astype(int)
            height = boxes[:, 3].astype(int)

            x = np.floor(centerX - (width / 2))
            y = np.floor(centerY - (height / 2))

            boxes[:, 0] = x
            boxes[:, 1] = y
            boxes[:, 2] = width
            boxes[:, 3] = height
            BOXES = boxes.astype(int)

            BOXES = BOXES.tolist()
            confidences = confidences.tolist()
            indices = cv2.dnn.NMSBoxes(BOXES, confidences, DETECTION_THRESHOLD, NMS_THRESHOLD)

            if len(indices) > 0:
                for idx, i in enumerate(indices.flatten()):
                    (x, y) = (int(boxes[i][0]), int(boxes[i][1]))
                    (w, h) = (int(boxes[i][2]), int(boxes[i][3]))

                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    class_name = str(self.class_names[class_ids[i]])
                    if class_name in self.allowed_classes:
                        box_string = str(x) + "," + str(y) + "," + str(x + w) + ',' + str(y + h)
                        results_list.append({
                            "object_id": (str(idx + 1)),
                            "box": box_string,
                            "confidence": confidences[i],
                            "label": class_name
                        })
        except Exception as ex:
            self.log.error("Detections to boxes failed")
            self.log.error(ex)
            return []

        return results_list
