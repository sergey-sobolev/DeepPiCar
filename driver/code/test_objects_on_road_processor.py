import cv2
import pytest
import logging
import time
from objects_on_road_processor import ObjectsOnRoadProcessor

_SHOW_IMAGE = True


############################
# Utility Functions
############################
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)



############################
# Test Functions
############################
@pytest.mark.parametrize("file", [
    '/home/pi/DeepPiCar/driver/data/objects/red_light.jpg',
    '/home/pi/DeepPiCar/driver/data/objects/person.jpg',
    '/home/pi/DeepPiCar/driver/data/objects/limit_40.jpg',
    '/home/pi/DeepPiCar/driver/data/objects/limit_25.jpg',
    '/home/pi/DeepPiCar/driver/data/objects/green_light.jpg',
    '/home/pi/DeepPiCar/driver/data/objects/no_obj.jpg'])
def test_photo(file):
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread(file)
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Detected Objects', combo_image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def test_stop_sign():
    # this simulates a car at stop sign
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 1', combo_image)
    time.sleep(1)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 2', combo_image)
    time.sleep(2)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 3', combo_image)
    frame = cv2.imread(
        '/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    frame = cv2.imread(
        '/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 4', combo_image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def test_video(video_file):
    object_processor = ObjectsOnRoadProcessor()
    cap = cv2.VideoCapture(video_file + '.avi')

    # skip first second of video.
    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    video_overlay = cv2.VideoWriter(
        "%s_overlay_%s.avi" % (video_file, date_str), video_type, 20.0,
        (320, 240))
    video_overlay = cv2.VideoWriter(
        "%s_overlay_%s.avi" % (video_file, date_str), video_type, 20.0,
        (320, 240))
    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            cv2.imwrite("%s_%03d.png" % (video_file, i), frame)

            combo_image = object_processor.process_objects_on_road(frame)
            cv2.imwrite("%s_overlay_%03d.png" % (video_file, i), combo_image)
            video_overlay.write(combo_image)

            cv2.imshow("Detected Objects", combo_image)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-5s:%(asctime)s: %(message)s')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-5s:%(asctime)s: %(message)s')

    # These processors contains no state
    test_photo('/home/pi/DeepPiCar/driver/data/objects/red_light.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/person.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/limit_40.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/limit_25.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/no_obj.jpg')

    # test stop sign, which carries state
    test_stop_sign()