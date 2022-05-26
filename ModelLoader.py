import os
import cv2
import logging

weightsPath = "/home/anubhav/Documents/Documents/models/yolov4/yolov4.weights"
configPath = "/home/anubhav/Documents/Documents/models/yolov4/yolov4.cfg"
namesPath = "/home/anubhav/Documents/Documents/models/yolov4/coco.names"


def load_model():
    log = logging.getLogger('app')

    if not os.path.isfile(weightsPath) or not os.path.isfile(configPath):
        log.error("Model Weights/Config File does not exist")
        return None

    try:
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except cv2.error as ex:
        log.error("Could not load model using opencv")
        return None

    return net


def load_classes() -> list:
    classes = []
    log = logging.getLogger('app')

    if not os.path.isfile(namesPath):
        log.error("Classes File does not exist")
        return classes

    with open(namesPath, 'rt') as f:
        classes = f.read().splitlines()

    if not bool(classes):
        log.error("Classes list is empty")
        return []

    log.info('Loaded classes list : {}'.format(classes))
    return classes


