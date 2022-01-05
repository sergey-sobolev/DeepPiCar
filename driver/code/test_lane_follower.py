import logging
import pytest
import cv2

from hand_coded_lane_follower import HandCodedLaneFollower, show_image


############################
# Test Functions
############################
@pytest.mark.parametrize("file", [  
    '/home/pi/DeepPiCar/models/lane_navigation/data/images/lanes.jpeg',
    '/home/pi/DeepPiCar/models/lane_navigation/data/images/video01_000_085.png',
    '/home/pi/DeepPiCar/models/lane_navigation/data/images/video01_001_080.png',
    '/home/pi/DeepPiCar/models/lane_navigation/data/images/video01_002_077.png',
    '/home/pi/DeepPiCar/models/lane_navigation/data/images/video01_003_075.png',
    '/home/pi/DeepPiCar/models/lane_navigation/data/images/video01_004_072.png',    
    '/home/pi/DeepPiCar/models/lane_navigation/data/images/video01_005_073.png'
    ])
def test_photo(file):
    land_follower = HandCodedLaneFollower()
    frame = cv2.imread(file)
    combo_image = land_follower.follow_lane(frame)
    show_image('final', combo_image, True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_video(video_file):
    lane_follower = HandCodedLaneFollower()
    cap = cv2.VideoCapture(video_file + '.avi')

    # skip first second of video.
    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    video_overlay = cv2.VideoWriter("%s_overlay.avi" % (video_file), video_type, 20.0, (320, 240))
    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            print('frame %s' % i )
            combo_image= lane_follower.follow_lane(frame)
            
            cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_follower.curr_steering_angle), frame)
            
            cv2.imwrite("%s_overlay_%03d.png" % (video_file, i), combo_image)
            video_overlay.write(combo_image)
            cv2.imshow("Road with Lane line", combo_image)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    test_video('/home/pi/DeepPiCar/driver/data/tmp/video01')
    #test_photo('/home/pi/DeepPiCar/driver/data/video/car_video_190427_110320_073.png')
    #test_photo(sys.argv[1])
    #test_video(sys.argv[1])