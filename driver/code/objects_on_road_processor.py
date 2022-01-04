import cv2
import logging
import time
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
from traffic_objects import *


class ObjectsOnRoadProcessor(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(
            self,
            car=None,
            speed_limit=40,
            model='/home/pi/DeepPiCar/models/object_detection/data/model_result/road_signs_quantized_edgetpu.tflite',
            label='/home/pi/DeepPiCar/models/object_detection/data/model_result/road_sign_labels.txt',
            width=640,
            height=480):
        # model: This MUST be a tflite model that was specifically compiled for Edge TPU.
        # https://coral.withgoogle.com/web-compiler/
        logging.info('Creating a ObjectsOnRoadProcessor...')
        self.width = width
        self.height = height

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit

        # initialize TensorFlow models
        with open(label, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

        # initial edge TPU engine
        logging.info('Initialize Edge TPU with model %s...' % model)
        self.engine = make_interpreter(model)
        self.engine.allocate_tensors()

        self.min_confidence = 0.30
        self.num_of_objects = 3
        logging.info('Initialize Edge TPU with model done.')

        # initialize open cv for drawing boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)  # white
        self.boxColor = (0, 0, 255)  # RED
        self.boxLineWidth = 1
        self.lineType = 2
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0  # ms

        self.traffic_objects = {
            0: GreenTrafficLight(),
            1: Person(),
            2: RedTrafficLight(),
            3: SpeedLimit(25),
            4: SpeedLimit(40),
            5: StopSign()
        }

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        logging.debug('Processing objects.................................')
        objects, final_frame = self.detect_objects(frame)
        self.control_car(objects)
        logging.debug('Processing objects END..............................')

        return final_frame

    def control_car(self, objects):
        car_state = {
            "speed": self.speed_limit,
            "speed_limit": self.speed_limit
        }
        car_state = {
            "speed": self.speed_limit,
            "speed_limit": self.speed_limit
        }

        logging.debug('No objects detected, drive at speed limit of %s.' %
                      self.speed_limit)

        contain_stop_sign = False
        for obj in objects:
            obj_label = self.labels[obj.id]
            processor = self.traffic_objects[obj.id]
            if processor.is_close_by(obj, self.height):
                processor.set_car_state(car_state)
                logging.debug(
                    "[%s] object detected, but it is too far, ignoring. " %
                    obj_label)
                logging.debug(
                    "[%s] object detected, but it is too far, ignoring. " %
                    obj_label)
            if obj_label == 'Stop':
                contain_stop_sign = True

        if not contain_stop_sign:
            self.traffic_objects[5].clear()

        self.resume_driving(car_state)

    def resume_driving(self, car_state):
        old_speed = self.speed
        self.speed_limit = car_state['speed_limit']
        self.speed = car_state['speed']

        if self.speed == 0:
            self.set_speed(0)
        else:
            logging.debug('Current Speed = %d, New Speed = %d' %
                          (old_speed, self.speed))
        logging.debug('Current Speed = %d, New Speed = %d' %
                      (old_speed, self.speed))

        if self.speed == 0:
            logging.debug('full stop for 1 seconds')
            time.sleep(1)

    def set_speed(self, speed):
        # Use this setter, so we can test this class without a car attached
        self.speed = speed
        if self.car is not None:
            logging.debug("Actually setting car speed to %d" % speed)
            self.car.back_wheels.speed = speed

    ############################
    # Frame processing steps
    ############################
    def detect_objects(self, frame):
        logging.debug('Detecting objects...')

        # call tpu for inference
        start_ms = time.time()
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_RGB)
        _, scale = common.set_resized_input(
            self.engine, image.size,
            lambda size: image.resize(size, Image.ANTIALIAS))
        self.engine.invoke()
        objects = detect.get_objects(self.engine,
                                     score_threshold=self.min_confidence,
                                     image_scale=scale)
        if objects:
            for obj in objects:
                box = obj.bbox
                height = box.height
                width = box.width
                logging.debug("%s, %.0f%% w=%.0f h=%.0f" %
                              (self.labels[obj.id], obj.score * 100,
                               width, height))
                logging.debug("%s, %.0f%% w=%.0f h=%.0f" %
                              (self.labels[obj.id], obj.score * 100,
                               width, height))                
                coord_top_left = (int(box.xmin), int(box.ymax))
                coord_bottom_right = (int(box.xmax), int(box.ymin))
                cv2.rectangle(frame, coord_top_left, coord_bottom_right,
                              self.boxColor, self.boxLineWidth)
                annotate_text = "%s %.0f%%" % (self.labels[obj.id],
                                               obj.score * 100)
                annotate_text = "%s %.0f%%" % (self.labels[obj.id],
                                               obj.score * 100)
                cv2.putText(frame, annotate_text, coord_top_left, self.font,
                            self.fontScale, self.boxColor, self.lineType)
                cv2.putText(frame, annotate_text, coord_top_left, self.font,
                            self.fontScale, self.boxColor, self.lineType)
        else:
            logging.debug('No object detected')

        elapsed_ms = time.time() - start_ms
        annotate_summary = "%.1f FPS" % (1.0 / elapsed_ms)
        annotate_summary = "%.1f FPS" % (1.0 / elapsed_ms)
        cv2.putText(frame, annotate_summary, self.bottomLeftCornerOfText,
                    self.font, self.fontScale, self.fontColor, self.lineType)
        cv2.putText(frame, annotate_summary, self.bottomLeftCornerOfText,
                    self.font, self.fontScale, self.fontColor, self.lineType)
        #cv2.imshow('Detected Objects', frame)

        return objects, frame
