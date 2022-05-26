import multiprocessing
import queue
import threading
import signal
import time
import json
import logging
import traceback
from datetime import datetime
from json import JSONDecodeError

import helpers as hp
from timeit import default_timer as timer

from kafkaConsumer import KafkaConsumer
from kafkaProducer import KafkaProducer
from PreWorker import BatchWorker
from PredictWorker import InferWorker
from PostWorker import PostPWorker

MAX_TASK_QUEUE = 10
MAX_BATCH_QUEUE = 40


class Manager:
    def __init__(self, num_batch_workers, num_infer_workers, num_post_workers, _test=False):
        # private variables
        self.test = _test
        self.num_batch_workers = num_batch_workers
        self.num_infer_workers = num_infer_workers
        self.num_post_workers = num_post_workers

        self.batch_workers = []
        self.infer_workers = []
        self.post_workers = []

        # Events
        self.quit_event = multiprocessing.Event()
        self.quit_event_thread = threading.Event()
        self.quit_event_observer = threading.Event()

        # Locks
        self.thread_lock = threading.Lock()

        # Establish communication queues
        self.task_queue = multiprocessing.Queue(MAX_TASK_QUEUE)
        self.result_queue = multiprocessing.Queue()
        self.batch_queue = multiprocessing.Queue(MAX_BATCH_QUEUE)
        self.infer_queue = multiprocessing.Queue()
        self.video_dict = {}  # would require locking for production and consumption between threads

        # Register the signal handlers
        signal.signal(signal.SIGTERM, self.shutdown_signal)
        signal.signal(signal.SIGINT, self.shutdown_signal)
        signal.signal(signal.SIGHUP, self.shutdown_signal)
        signal.signal(signal.SIGQUIT, self.shutdown_signal)

        # Register Logger
        self.log = logging.getLogger('app')
        self.log.info('Manager Initialization Complete')

        # Register observer thread
        self.thread_observer = threading.Thread(target=self.thread_state_observer)

        # Register kafka consumer producer threads
        self.kafka_consumer_thread = threading.Thread(target=self.message_consumer)
        self.kafka_producer_thread = threading.Thread(target=self.message_producer)

    def shutdown_all(self):
        self.quit_event.set()
        self.quit_event_thread.set()

    def shutdown_signal(self, signum, frame) -> None:
        # exit_signals = ['SIGHUP', 'SIGINT', 'SIGQUIT', 'SIGTERM']
        self.log.warning('Received SIGINT, SIGTERM, shutting down now...')
        self.shutdown_all()

    def thread_state_observer(self):
        thread_state_dict = {"kafka_consumer": "OFF",
                             "kafka_producer": "OFF",
                             "batch_workers": "OFF",
                             "infer_workers": "OFF",
                             "post_workers": "OFF"}

        prev_val = {"kafka_consumer": False,
                    "kafka_producer": False,
                    "batch_workers": False,
                    "infer_workers": False,
                    "post_workers": False}

        while not self.quit_event_observer.is_set():
            time.sleep(1.5)  # add delay else this will continuously poll

            curr_val = {"kafka_consumer": False,
                        "kafka_producer": False,
                        "batch_workers": False,
                        "infer_workers": False,
                        "post_workers": False}

            if self.kafka_consumer_thread.is_alive():
                thread_state_dict["kafka_consumer"] = "ON"
                curr_val["kafka_consumer"] = True
            else:
                thread_state_dict["kafka_consumer"] = "OFF"
                curr_val["kafka_consumer"] = False

            if self.kafka_producer_thread.is_alive():
                thread_state_dict["kafka_producer"] = "ON"
                curr_val["kafka_producer"] = True
            else:
                thread_state_dict["kafka_producer"] = "OFF"
                curr_val["kafka_producer"] = False

            # Check State of all the BatchWorker Subprocesses
            temp = 0
            for c in self.batch_workers:
                if c.is_alive():
                    temp += 1
            if temp > 0:
                thread_state_dict["batch_workers"] = "{}/{}".format(temp, self.num_batch_workers)
                if temp == self.num_batch_workers:
                    curr_val["batch_workers"] = True
                else:
                    curr_val["batch_workers"] = False
            else:
                thread_state_dict["batch_workers"] = "OFF"
                curr_val["batch_workers"] = False

            # Check State of all InferWorker Subprocesses
            temp = 0  # reset
            for p in self.infer_workers:
                if p.is_alive():
                    temp += 1
            if temp > 0:
                thread_state_dict["infer_workers"] = "{}/{}".format(temp, self.num_infer_workers)
                if temp == self.num_infer_workers:
                    curr_val["infer_workers"] = True
                else:
                    curr_val["infer_workers"] = False
            else:
                thread_state_dict["infer_workers"] = "OFF"
                curr_val["infer_workers"] = False

            # Check State of all PostWorker Subprocesses
            temp = 0  # reset
            for p in self.post_workers:
                if p.is_alive():
                    temp += 1
            if temp > 0:
                thread_state_dict["post_workers"] = "{}/{}".format(temp, self.num_post_workers)
                if temp == self.num_post_workers:
                    curr_val["post_workers"] = True
                else:
                    curr_val["post_workers"] = False
            else:
                thread_state_dict["post_workers"] = "OFF"
                curr_val["post_workers"] = False

            try:
                self.log.info('<TaskQueue> Size : {}'.format(self.task_queue.qsize()))
            except ValueError as ex:
                self.log.warning('<TaskQueue> not callable now')

            try:
                self.log.info('<BatchQueue> Size : {}'.format(self.batch_queue.qsize()))
            except ValueError as ex:
                self.log.warning('<BatchQueue> not callable now')

            try:
                self.log.info('<InferQueue> Size : {}'.format(self.infer_queue.qsize()))
            except ValueError as ex:
                self.log.warning('<InferQueue> not callable now')

            try:
                self.log.info('<ResultQueue> Size : {}'.format(self.result_queue.qsize()))
            except ValueError as ex:
                self.log.warning('<ResultQueue> not callable now')

            # ToDo check and terminate -> add terminator if any thread/process dies
            # need to check if the state changes from True to False
            for prev, curr in zip(prev_val, curr_val):
                if prev_val[prev] and not curr_val[prev]:
                    self.log.critical('Triggered Shutdown due to inoperative threads xxxxxxxxx')
                    self.shutdown_all()
                    break

            prev_val = curr_val
            self.log.critical('Curr ---- {}'.format(curr_val))
            self.log.critical('Prev ---- {}'.format(prev_val))
            self.log.critical('Thread and Process States :: {}'.format(thread_state_dict))

        self.log.info('Terminating the Observer thread gracefully')

    def process(self) -> None:
        self.log.info('Creating {} BatchWorker & {} InferWorker & {} PostWorker'
                      .format(self.num_batch_workers, self.num_infer_workers, self.num_post_workers))

        # Start the Observer thread
        self.thread_observer.start()

        # Start Inference Workers
        self.infer_workers = [
            InferWorker(self.quit_event, self.batch_queue, self.infer_queue)
            for i in range(self.num_infer_workers)
        ]

        # model loading
        try:
            for w in self.infer_workers:
                if not w.init_model():
                    raise Exception('DNN Model Could not be loaded')
        except Exception as ex:
            self.log.error('DNN Error : {}'.format(ex))
            self.shutdown_all()

        for w in self.infer_workers:
            w.start()

        # Start Post Process Workers
        self.post_workers = [
            PostPWorker(self.quit_event, self.infer_queue, self.result_queue)
            for i in range(self.num_post_workers)
        ]

        # model classes loading
        try:
            for w in self.post_workers:
                if not w.init_model_classes():
                    raise Exception('DNN Model class labels  Could not be loaded')
        except Exception as ex:
            self.log.error('DNN Error : {}'.format(ex))
            self.shutdown_all()

        for w in self.post_workers:
            w.start()

        # Start consumers and batch workers sub processes order -> producer > consumer
        self.batch_workers = [
            BatchWorker(self.quit_event, self.task_queue, self.batch_queue)
            for i in range(self.num_batch_workers)
        ]
        for w in self.batch_workers:
            w.start()

        # Start Kafka Threads for consuming and producing kafka messages
        self.kafka_consumer_thread.start()
        self.kafka_producer_thread.start()

        # Keep the main thread running, otherwise signals are ignored.
        while not self.quit_event.is_set() or not self.quit_event_thread.is_set():
            time.sleep(0.5)

        # Terminate the running threads.
        # Set the shutdown flag on each thread to trigger a clean shutdown of each thread.
        self.log.info('Set terminate threads and subprocesses')
        self.shutdown_all()

        self.log.debug('Joining threads')
        self.kafka_consumer_thread.join()
        self.kafka_producer_thread.join()
        self.log.debug('Threads joined')

        # Wait for the threads to close...
        self.log.info('---- Wait for termination of  processes ----')

        if self.test:
            with self.thread_lock:
                while not self.batch_queue.empty():
                    batch = self.batch_queue.get()
                    self.log.info(' Batch ----- {}'.format(len(batch["record_keys"])))
                    for x in batch["record_keys"]:
                        self.result_queue.put(x, block=True)
                        time.sleep(0.01)
                    time.sleep(0.01)

        self.flush_queues()
        self.task_queue.close()
        self.task_queue.join_thread()
        self.batch_queue.close()
        self.batch_queue.join_thread()
        self.infer_queue.close()
        self.infer_queue.join_thread()
        self.result_queue.close()
        self.result_queue.join_thread()

        self.log.info('---- Queues Joined Terminating for BatchWorkers Now ----')
        for p in self.batch_workers:
            p.kill()
            p.join()

        self.log.info('---- BatchWorkers Joined Terminating for InferWorkers Now ----')
        for c in self.infer_workers:
            c.kill()
            c.join()

        self.log.info('---- InferWorkers Joined Terminating for PostWorkers Now ----')
        for postw in self.post_workers:
            postw.kill()
            postw.join()

        self.log.info('---- PostWorkers Joined Terminating Observer Now ----')

        # Unregister Observer thread and join
        self.quit_event_observer.set()
        time.sleep(0.5)  # add delay to flush logs and then quit
        self.thread_observer.join()

        self.log.info('Exiting Manager')

    # ----------------------- Below Kafka Consumer & Producer -------------------------------------------

    def message_consumer(self) -> None:
        # Setup Kafka Consumer
        kconsumer = KafkaConsumer('127.0.0.1:9092', ['todo'])
        while not self.quit_event_thread.is_set():
            try:
                msg = kconsumer.consume_one_message()
                if msg is not None:
                    # process the message
                    msg_json = json.loads(msg.decode('utf-8'))
                    self.log.debug('Consumed message : {}'.format(msg_json))

                    # validate json
                    if not hp.validate_input_json(msg_json):
                        continue  # Skip the message if not valid

                    # Todo : Seems like this is not the correct way to query a key, but below does the job
                    is_present = False
                    with self.thread_lock:
                        if self.video_dict.get(msg_json['videoId'], "NONE") == "NONE":
                            is_present = False
                            self.log.info('Adding new video in video dictionary : {}'.format(msg_json['videoId']))
                            # don't add the record yet, extract the frames/vehicles and make the paths ready
                        else:
                            is_present = True
                            self.log.warning('Skipping this video as this is already in video dictionary : {}'.
                                             format(msg_json['videoId']))

                    # If the videoId is not present in the processing list we have to add it and then
                    # push the corresponding data points to the task queue
                    if not is_present:
                        video_record = hp.msg_decode_n_load(msg_json)

                        # We have to ensure that the next process worker is not dependant on any video
                        # level information. It only creates batches and propagates batches to the
                        # inference block. Also this put method should be blocking in the sense it should
                        # not throw exception if the queue is full. The next process worker step just loads
                        # up the image and pushes it to batch_queue for inference.
                        # each index should be a dictionary
                        if len(video_record) > 0:
                            # Add the videoId entry in the dictionary
                            with self.thread_lock:
                                now = datetime.datetime.now()
                                self.video_dict.update({msg_json['videoId']: {"frames": len(video_record),
                                                                              "start_time": now,
                                                                              "stop_time": now,
                                                                              "time_taken_s": 0.0}})

                            self.log.info('----- Video <{}> Record Count = {} -----'.format(msg_json['videoId'],
                                                                                            len(video_record)))
                            data_count = 0
                            for item in video_record:
                                # Note :if the put method is blocking then this thread will be in wait forever
                                # if the consumer for task_queue gets killed.
                                is_pushed = False
                                while not is_pushed:
                                    try:
                                        self.task_queue.put(item, block=True, timeout=0.05)
                                        data_count += 1
                                        is_pushed = True
                                    except queue.Full:
                                        time.sleep(0.1)
                                        self.log.debug('Waiting for the task_queue to get free ...')
                                        is_pushed = False
                                        if self.quit_event_thread.is_set():
                                            break
                                if self.quit_event_thread.is_set():
                                    self.log.warning('[message_consumer] Quit event is set breaking out '
                                                     'from item iterator in message_consume_one')
                                    break

                            self.log.info('Items for videoId <{}> pushed on task_queue with records = {}'
                                          .format(msg_json['videoId'], data_count))
                else:
                    time.sleep(0.05)
            except JSONDecodeError as js:
                self.log.error("Kafka Input JSON Message exception: {}, skipping the received message".format(js))
                continue
            except:
                self.log.error("Uncaught exception: %s", traceback.format_exc())
                self.shutdown_all()

        kconsumer.disconnect()
        self.log.info('Kafka Consumer Thread Disconnected')

    def message_producer(self) -> None:
        # Setup kafka producer
        kproducer = KafkaProducer('127.0.0.1:9092')
        count = 100  # just for sake of testing
        local_video_dict = {}
        while not self.quit_event_thread.is_set():
            # time.sleep(0.1)

            # try:
            #     msg = kproducer.produce_one_message('output', 'operator_{}'.format(count))
            #     self.log.info('Producing Message : {}'.format(str(msg)))
            #     count += 1
            # except:
            #     self.log.error("Uncaught exception: %s", traceback.format_exc())
            #     self.shutdown_all()

            # query task from input queue-> get a new task from the queue

            try:
                next_task = self.result_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                # self.log.debug('Result Queue is empty')
                time.sleep(0.01)
                continue

            # unlikely - but taken up
            if next_task is None:
                self.log.warning('Post Process Input Queue had <None> fetched, trying to fetch again')
                continue

            try:
                self.log.debug("Received msg from PostWorker : {}".format(next_task['record_key']))
                record_key = next_task['record_key']
                boxes_list = next_task['boxes_list']
                userId, videoId, frameId = record_key.split("-")
            except KeyError as ke:
                self.log.error("Issue in reading keys in received record from post worker: {}".format(ke))
                self.shutdown_all()
                continue
            except ValueError as ve:
                self.log.error("Issue in splitting record key: {}".format(ve))
                self.shutdown_all()
                continue

            # process the boxes
            hp.save_results(videoId, frameId, detections=boxes_list)

            # check in local video dictionary if the video is present then we reduce there,
            # it is is not present we add the videoid to dictionary
            # we maintain local dictionary so as to we dont intermittently block the consumer thread
            if local_video_dict.get(videoId, "NONE") == "NONE":
                self.log.error('Key VideoID <{}> is not present in local video dictionary hence adding from global'
                               .format(videoId))

                # get videoId from global
                with self.thread_lock:
                    video_record_d = self.video_dict.get(videoId, "NONE")

                if video_record_d == "NONE":
                    self.log.error('Key VideoID <{}> is not present in global dictionary also xxx'.format('videoId'))
                    self.shutdown_all()  # come out of the loop
                    continue
                else:
                    # key present in global, lets copy that to local
                    local_video_dict.update({videoId: video_record_d})
                    self.log.warning('Key VideoID <{}> added to local dictionary'.format(videoId))

            self.log.debug('Local Dict value : {}'.format(local_video_dict))
            # self.log.debug('Global Dict value : {}'.format(self.video_dict))

            # At this point we have the video present in global replicated to local, hence we now only update local,
            # Once the entire video seq is processed we erase the key from global, hence reducing the entire
            # intermittent blocking to happen only twice per video

            # This is how key value looks like
            # {msg_json['videoId']: {"frames": len(video_record),
            #                        "start_time": datetime.timestamp(now),
            #                        "stop_time": datetime.timestamp(now),
            #                        "time_taken_s": 0.0}}

            try:
                local_video_dict[videoId]["frames"] -= 1
                # check if the entire video was processed
                if local_video_dict[videoId]["frames"] <= 0:
                    # indicates the video is processed, now we need to eject this key from global and then local
                    # remove from global video dict
                    with self.thread_lock:
                        self.video_dict.pop(videoId)

                    # copy the value since we have to produce this
                    # now = datetime.now()
                    # record_msg = local_video_dict[videoId]
                    # record_msg["stop_time"] = datetime.timestamp(now)
                    # record_msg["time_taken_s"] = (datetime.timestamp(now) -
                    #                               record_msg.get("start_time", 0.0)).milliseconds()/1000

                    end = datetime.datetime.now()
                    record_msg = local_video_dict[videoId]
                    record_msg["stop_time"] = end
                    c = (end - record_msg.get("start_time"))
                    record_msg["time_taken_s"] = int(c.total_seconds()*1000)  # in milliseconds

                    # pop from local dictionary
                    local_video_dict.pop(videoId)

                    # # produce message on kafka
                    msg = kproducer.produce_one_message('output', json.dumps({videoId: record_msg}))
                    self.log.info('Producing Message : {}'.format(json.dumps(record_msg)))

            except KeyError as ke:
                self.log.warning('In local_video_dict, key frames is not found : {}'.format(ke))
                self.shutdown_all()
                continue
            except ValueError as ve:
                self.log.warning('In local_video_dict, key frames may not be appropriate data type : {}'.format(ve))
                self.shutdown_all()
                continue
            except Exception as ex:
                self.log.error("Uncaught exception: %s", traceback.format_exc())
                self.shutdown_all()

        self.log.info('Kafka Producer Thread Disconnected')

    def flush_queues(self):
        is_not_empty = True
        while is_not_empty:
            try:
                next_task = self.task_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                self.log.debug('Task Queue Emptied')
                is_not_empty = False

        is_not_empty = True
        while is_not_empty:
            try:
                next_task = self.batch_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                self.log.debug('Batch Queue Emptied')
                is_not_empty = False

        is_not_empty = True
        while is_not_empty:
            try:
                next_task = self.infer_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                self.log.debug('Infer Queue Emptied')
                is_not_empty = False

        is_not_empty = True
        while is_not_empty:
            try:
                next_task = self.result_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                self.log.debug('Result Queue Emptied')
                is_not_empty = False



