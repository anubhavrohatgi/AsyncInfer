import json
import logging
import os
import cv2
import subprocess
import glob
import re
import jsonschema
from jsonschema import validate

PATH_TO_VIDEOS = '/home/anubhav/Downloads/videos'
PATH_TO_RESULTS = '/home/anubhav/Downloads/results'


def validate_input_json(msg_json) -> bool:
    # Input Message Schema.
    input_schema = {
        "videoId": "video_1"
    }

    log = logging.getLogger('app')
    # Validate will raise exception if given json is not
    # what is described in schema.
    try:
        validate(instance=msg_json, schema=input_schema)
    except jsonschema.ValidationError as vex:
        log.warning('Invalid input Json message received : {}'.format(vex))
        log.exception(vex)
        return False
    return True


def validate_output_json(msg_json) -> bool:
    # Input Message Schema.
    output_schema = {
        "videoId": "video_1"
    }
    log = logging.getLogger('app')
    # Validate will raise exception if given json is not
    # what is described in schema.
    try:
        validate(instance=msg_json, schema=output_schema)
    except jsonschema.ValidationError as vex:
        log.warning('Invalid output Json message received : {}'.format(vex))
        log.exception(vex)
        return False
    return True


def msg_decode_n_load(msg_json=None) -> list:
    """

    :param msg_json: Input json message received from Kafka
    :return: Output list of dictionaries referring to grains
    """
    log = logging.getLogger('app')

    if msg_json is None:
        return []

    # validate the path
    try:
        if not os.path.exists(os.path.join(PATH_TO_VIDEOS, msg_json["videoId"])):
            log.warning('VideoId folder not present on storage {}'.format(os.path.join(PATH_TO_VIDEOS,
                                                                                       msg_json["videoId"])))
            return []
    except OSError as ex:
        log.warning('Exception found while accessing the videoId {} folder'.format(msg_json["videoId"]))
        log.exception(ex)
        return []

    try:
        files = sorted(glob.glob(os.path.join(PATH_TO_VIDEOS, msg_json["videoId"]) + "/*.jpg"),
                       key=lambda x: float(re.findall("frame_(\\d+)", x)[0]))

        output = [{"videoId": msg_json["videoId"],
                   "frame_url": str(x),
                   "frameId": str(re.findall("frame_(\\d+)", x)[0]),
                   "userId": "None"} for x in files]
        log.debug('Frame_Message  :  {}'.format(output))
        return output
    except IndexError as ex:
        log.error('Issue in gathering filenames for the videoId {}'.format(msg_json["videoId"]))
        return []
    # return {"videoId": msg_json["videoId"], "frames": files, "total_records": len(files)}


def save_results(videoId, frameId, detections=[]) -> bool:
    log = logging.getLogger('app')
    results_path = None
    if not os.path.exists(os.path.join(PATH_TO_RESULTS, videoId)):
        os.makedirs(os.path.join(PATH_TO_RESULTS, videoId))
        results_path = os.path.join(PATH_TO_RESULTS, videoId)
        log.debug('Cannot find the results path, creating folder @ {}'.format(results_path))
    else:
        results_path = os.path.join(PATH_TO_RESULTS, videoId)

    if results_path is None:
        log.error('Error in path creation for saving method')
        return False

    # e.g. video_4_frame_1
    # read file
    frame = None
    im_frame_filename = videoId + '_frame_' + frameId + '.jpg'

    try:
        frame = cv2.imread(os.path.join(PATH_TO_VIDEOS, videoId, im_frame_filename))
        if frame is None:
            raise cv2.error('Image file read error')

    except cv2.error as cve:
        log.error('Cannot read the input filename for plotting the info : {}'
                  .format(os.path.join(PATH_TO_VIDEOS, videoId, im_frame_filename)))
        return False

    # plot
    #  box_string = str(x) + "," + str(y) + "," + str(x + w) + ',' + str(y + h)
    #                         results_list.append({
    #                             "object_id": (str(idx + 1)),
    #                             "box": box_string,
    #                             "confidence": confidences[i],
    #                             "label": class_name
    #                         })

    for item in detections:
        bbox_string = item["box"]
        conf = item["confidence"]
        class_name = item["label"]

        left, top, right, bottom = map(int, bbox_string.split(','))

        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        label = '%s: %s' % (class_name, label)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
                      cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imwrite(os.path.join(results_path, im_frame_filename), frame)
    log.debug('Image written @ {} <<<<<<<'.format(os.path.join(results_path, im_frame_filename)))
