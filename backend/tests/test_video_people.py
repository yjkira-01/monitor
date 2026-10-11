from pathlib import Path
from types import SimpleNamespace
import time
import threading

import cv2
import numpy as np

from app import data_source
from app.sources.mmvital_source import RadarPhaseProvider
from app.video_model import VideoModelWorker
from app.video_people import annotate_people, serialize_people, match_people, VideoPoseWorker


def pose(visibility=1.):
    return [SimpleNamespace(x=.2+i*.01, y=.2+i*.015, z=-.1, visibility=visibility) for i in range(33)]


def test_pose_serialization_keeps_relative_world_data_and_rejects_invalid_targets():
    good, hidden, invalid = pose(), pose(.1), pose()
    invalid[10].x = float('nan')
    result = SimpleNamespace(pose_landmarks=[good, hidden, invalid], pose_world_landmarks=[pose(), pose(), pose()])
    people = serialize_people(result)
    assert len(people) == 1 and people[0]['index'] == 1
    assert len(people[0]['world_landmarks']) == 33
    assert all(0 <= v <= 1 for v in people[0]['bbox'])
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    jpeg = annotate_people(frame, people)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == frame.shape and decoded.max() > 0
    assert decoded[40:70, 50:75].mean() < 2
    assert frame.max() == 0


def test_detector_box_association_does_not_assign_a_pose_to_another_person():
    pose = {'bbox': [.1, .1, .4, .8], 'landmarks': ['pose'], 'world_landmarks': ['world'], 'visibility': .8}
    boxes = [{'bbox': [.09, .1, .42, .82]}, {'bbox': [.6, .1, .9, .8]}]
    people = match_people(boxes, [pose])
    assert people[0]['world_landmarks'] == ['world']
    assert people[1]['world_landmarks'] == []
    assert [p['index'] for p in people] == [1, 2]
    assert match_people([], [pose])[0]['box_source'] == 'pose_extent'


def test_nested_duplicate_detection_is_not_a_second_overlapping_person():
    pose = {'bbox':[.42,.08,.53,.53],'landmarks':['pose'],'world_landmarks':['world'],'visibility':.9}
    boxes = [{'bbox':[.42,.08,.53,.53],'detector_score':.9},
             {'bbox':[.432,.15,.518,.42],'detector_score':.4},
             {'bbox':[.7,.08,.85,.53],'detector_score':.8}]
    people = match_people(boxes,[pose])
    assert len(people)==2 and people[0]['landmarks']==['pose']


def test_small_head_box_inside_one_body_is_not_a_second_target():
    pose = {'bbox':[.1,.1,.5,.95],'landmarks':['pose'],'world_landmarks':['world'],'visibility':.9}
    boxes = [{'bbox':[.1,.1,.5,.95],'detector_score':.95}, {'bbox':[.24,.1,.38,.3],'detector_score':.6}]
    people = match_people(boxes,[pose])
    assert len(people)==1 and people[0]['landmarks']==['pose']


def test_two_independent_poses_survive_overlapping_detection_boxes():
    poses = [{'bbox':box,'landmarks':[name],'world_landmarks':[name],'visibility':.9}
             for name,box in [('a',[.1,.1,.5,.95]),('b',[.18,.1,.58,.95])]]
    boxes = [{'bbox':p['bbox'],'detector_score':.9} for p in poses]
    assert {p['landmarks'][0] for p in match_people(boxes,poses)}=={'a','b'}


def test_duplicate_pose_is_merged_even_without_detector_boxes():
    points = [{'x':.3+i*.001,'y':.2+i*.015,'visibility':1.} for i in range(33)]
    first = {'bbox':[.2,.1,.5,.8],'landmarks':points,'world_landmarks':[],'visibility':.9}
    second = {**first,'landmarks':[{**p,'x':p['x']+.003} for p in points],'visibility':.8}
    separate = {**first,'bbox':[.6,.1,.9,.8],'landmarks':[{**p,'x':p['x']+.4} for p in points]}
    assert len(match_people([], [first,second,separate]))==2


def test_reset_drops_detection_frame_and_inflight_person_results():
    worker = VideoPoseWorker()
    frame = np.zeros((16, 16, 3), np.uint8)
    prediction = {'state': 'ready', 'people': [{'index': 1, 'bbox': [.1,.1,.9,.9]}]}
    assert worker._accept(prediction, b'people', frame, 2, 0, 1, 12)
    assert worker.observation() == b'people'
    worker.reset()
    assert worker.observation() is None and worker.snapshot()['people'] == []
    assert worker.tracking_detection() is None
    assert not worker._accept(prediction, b'people', frame, 3, 0, 1, 12)
    assert worker._accept(prediction, b'people', frame, 3, 1, 1, 12)


def test_observation_preview_is_cached_and_a_source_reset_rejects_inflight_encoding(monkeypatch):
    from app import video_people
    worker=VideoPoseWorker();frame=np.zeros((1080,1920,3),np.uint8);calls=[]
    worker._accept({'state':'ready','people':[]},None,frame,1,0,1,10)
    monkeypatch.setattr(video_people,'annotate_people',lambda preview,people:calls.append(preview.shape) or b'preview')
    assert worker.observation()==worker.observation()==b'preview' and calls==[(540,960,3)]
    worker._accept({'state':'ready','people':[]},None,frame,2,0,1,10)
    def encode_after_reset(preview,people):
        worker.reset();return b'old-source'
    monkeypatch.setattr(video_people,'annotate_people',encode_after_reset)
    assert worker.observation() is None, 'an old source must not overwrite the new source cache'


def test_pose_dependency_failure_does_not_stop_trained_fall_inference(monkeypatch):
    from app import video_people
    class Predictor:
        def __init__(self, path):
            pass
        def predict(self, frames):
            return {'fall_score': .1, 'embedding': np.ones(2), 'attention_jpeg': b'cam', 'attention_class': 'adl', 'device': 'cpu'}
    def missing():
        raise FileNotFoundError('pose weight missing')
    pose_worker = VideoPoseWorker(missing)
    worker = VideoModelWorker(Path('unused'), Predictor)
    worker.start()
    pose_worker.start()
    try:
        worker.submit([np.zeros((16, 16, 3), np.uint8)], 0, 2, 0, {}, 1)
        pose_worker.submit(np.zeros((16, 16, 3), np.uint8), 2, 0, 1)
        deadline = time.monotonic()+2
        while worker.snapshot()['windows_analyzed'] == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        result = worker.snapshot()
        assert result['model_state'] == 'ready' and result['predicted'] == 'adl'
        pose_worker._thread.join(1)
        assert pose_worker.snapshot()['state'] == 'error'
    finally:
        worker.stop()
        pose_worker.stop()


def test_point_cloud_does_not_masquerade_as_hardware_and_filters_invalid_points(monkeypatch):
    provider = RadarPhaseProvider.__new__(RadarPhaseProvider)
    provider._latest_points = np.array([[1, 2, 3], [float('nan'), 0, 0]])
    provider._latest_points_ms = int(time.time()*1000)
    monkeypatch.setattr(data_source._real, 'provider', provider)
    monkeypatch.setattr(data_source, 'get_source', lambda: 'mock')
    assert data_source.point_cloud_snapshot()['source'] == 'unavailable'
    monkeypatch.setattr(data_source, 'get_source', lambda: 'real')
    result = data_source.point_cloud_snapshot()
    assert result['source'] == 'real' and result['points'] == [[1., 2., 3.]] and not result['stale']
    provider._latest_points_ms -= 4000
    assert data_source.point_cloud_snapshot()['stale']
    provider._latest_points = np.zeros((3000, 3))
    assert len(data_source.point_cloud_snapshot()['points']) == 2048


def test_pose_queue_discards_old_frames_and_resets_video_timestamps():
    entered, release = threading.Event(), threading.Event()
    calls, resets = [], []
    class Predictor:
        def predict(self, frame, timestamp):
            calls.append(timestamp)
            if len(calls) == 1:
                entered.set()
                assert release.wait(2)
            return {'state':'ready','people':[]}, b'pose'
        def reset(self):
            resets.append(True)
        def close(self):
            pass
    worker = VideoPoseWorker(Predictor)
    worker.start()
    frame = np.zeros((16,16,3),np.uint8)
    try:
        worker.submit(frame, 1., 0, 1)
        assert entered.wait(2)
        worker.submit(frame, 2., 0, 2)
        worker.submit(frame, 3., 0, 3)
        worker.reset()
        worker.submit(frame, .1, 1, 4)
        worker.submit(frame, .2, 1, 5)
        release.set()
        deadline = time.monotonic()+2
        while worker.snapshot()['frames_analyzed'] == 0 and time.monotonic()<deadline:
            time.sleep(.01)
        result = worker.snapshot()
        assert calls == [1000, 200] and resets == [True]
        assert result['generation'] == 1 and result['frame_s'] == .2 and result['frames_analyzed'] == 1
        worker.submit(frame, .2, 1, 6)
        worker.submit(frame, .1, 1, 7)
        time.sleep(.03)
        assert calls == [1000, 200]
    finally:
        release.set()
        worker.stop()
