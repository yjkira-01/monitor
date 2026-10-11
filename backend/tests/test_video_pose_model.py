from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from app.video import VideoManager
from app.video_model import VideoModelWorker
from app.video_pose_model import PersonPoseModel
from app.llm import local_summary


def person(key, x):
    return {'track_id': key, 'landmarks': [{'x': x, 'y': .3, 'visibility': 1.} for _ in range(33)],
            'observation': {'video_usable': True}}


def predictor():
    result = PersonPoseModel.__new__(PersonPoseModel)
    result.torch, result.features = torch, lambda x: x
    result.model = lambda x: torch.tensor([[0., 4. if x[0, 0].mean() > .5 else -4.]])
    result.revision = 'fixture'
    result.reset()
    return result


def test_pose_history_is_person_specific_bounded_and_cleared_on_reset_or_gap():
    model = predictor()
    for i in range(75):
        people = [person(1, .2), person(2, .8)]
        model.update(people, i*.1, 1.)
    assert people[0]['pose_analysis']['fall_score'] < .5
    assert people[1]['pose_analysis']['fall_score'] > .5
    result = people[1]['pose_analysis']
    assert result['window_end_s']-result['window_start_s'] <= 5.001
    assert 7.4-result['window_end_s'] < .5
    model.update([person(1, .2)], 9., 1.)
    assert set(model.buffers) == {1} and not model.results
    model.reset()
    assert not model.buffers and not model.results


def test_pose_classifier_keeps_missing_joint_masks_and_does_not_require_complete_legs():
    model = predictor()
    visible = person(1, .8)
    for p in visible['landmarks'][25:]:
        p['visibility'] = 0
    for i in range(15):
        model.update([visible], i*.1, 1.)
    assert visible['pose_analysis']['fall_score'] > .5, 'visible torso must survive missing legs'
    hidden = deepcopy(visible)
    for p in hidden['landmarks']:
        p['visibility'] = 0
    for i in range(15, 80):
        model.update([hidden], i*.1, 1.)
    assert hidden['pose_analysis']['state'] == 'unobservable'
    assert hidden['pose_analysis']['fall_score'] is None


def test_pose_disagreement_changes_decision_and_events_without_rewriting_rgb_score():
    worker = VideoModelWorker(Path('unused'))
    pose = {'track_id': 1, 'state': 'fall_candidate', 'fall_score': .9}
    result = {'fall_score': .0001, 'embedding': np.ones(2), 'attention_jpeg': b'cam',
              'attention_class': 'adl', 'device': 'cpu'}
    assert worker._accept(result, 0., 2., 0, {'usable': True, 'pose_fall_people': [pose]},
                          int(time.time()*1000), 10)
    snapshot = worker.snapshot()
    assert snapshot['state'] == 'model_disagreement' and snapshot['watch'] and not snapshot['is_fall']
    assert snapshot['fall_score'] == .0001 and snapshot['predicted'] == 'adl'
    assert snapshot['events'][-1]['state'] == 'model_disagreement'


def test_current_partial_pose_can_raise_review_and_stale_pose_cannot():
    manager = VideoManager('unused')
    now = int(time.time()*1000)
    manager._update(connected=True)
    manager.worker._result.update(model_state='ready', windows_analyzed=1, fall_score=.0001,
                                  state='adl', captured_at_ms=now)
    p = person(1, .8)
    p['pose_analysis'] = {'state': 'fall_candidate', 'fall_score': .9}
    manager.pose_worker._result.update(state='ready', captured_at_ms=now, people=[p])
    assert manager.analysis()['state'] == 'model_disagreement'
    assert manager.pose()['people'][0]['analysis']['state'] == 'model_disagreement'
    manager.pose_worker._result['captured_at_ms'] = now-10000
    assert not manager.analysis()['watch']
    assert manager.pose()['people'][0]['analysis']['state'] == 'unobservable'


def test_local_report_describes_disagreement_instead_of_safety():
    context = {'hours': 24, 'alerts': 0, 'unhandled_alerts': 0, 'vitals': 0, 'actions': 0,
               'video': {'state': 'model_disagreement', 'watch': True, 'fall_score': .0001}}
    text = local_summary(context)
    assert '姿态时序模型' in text and 'RGB' in text and '分歧' in text
