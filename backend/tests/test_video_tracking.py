import cv2
import numpy as np

from app.video_tracking import PersonBoxTracker, tracking_gray
from app.video_people import VideoPoseWorker


def test_boxes_follow_real_image_motion_then_expire():
    rng = np.random.default_rng(4)
    frame = np.zeros((240,320,3),np.uint8)
    frame[70:170,100:180] = rng.integers(30,255,(100,80,3),dtype=np.uint8)
    person = {'index':1,'bbox':[100/320,70/240,180/320,170/240]}
    reference = (tracking_gray(frame),1.,[person])
    tracker = PersonBoxTracker()
    assert len(tracker.update(frame,reference,1.)) == 1
    moved = cv2.warpAffine(frame,np.float32([[1,0,8],[0,1,5]]),(320,240))
    boxes = tracker.update(moved,reference,1.1)
    assert len(boxes) == 1
    assert abs(boxes[0]['bbox'][0]-108/320)<.008
    assert abs(boxes[0]['bbox'][1]-75/240)<.008
    assert tracker.update(moved,reference,3.) == []
    assert tracker.update(moved,None,3.1) == []


def test_tracking_does_not_invent_boxes_without_texture_and_reset_clears_reference():
    frame = np.zeros((120,160,3),np.uint8)
    reference = (tracking_gray(frame),1.,[{'index':1,'bbox':[.1,.1,.9,.9]}])
    assert PersonBoxTracker().update(frame,reference,1.) == []
    worker = VideoPoseWorker()
    prediction = {'state':'ready','people':[{'index':1,'bbox':[.1,.1,.9,.9]}]}
    assert worker._accept(prediction,b'people',frame,2,0,1,12)
    assert worker.tracking_detection()[1] == 2
    worker.reset()
    assert worker.tracking_detection() is None
    assert not worker._accept(prediction,b'people',frame,3,0,1,12)
