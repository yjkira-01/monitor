from copy import deepcopy
from pathlib import Path

import numpy as np

from app.video_person_state import PersonTracks, PersonClips, observation, enhance_clip, scene_observation
from app.video_model import VideoModelWorker


def person(x=.25):
    image = [{'x': x, 'y': .2, 'z': 0., 'visibility': 1.} for _ in range(33)]
    world = [{'x': 0., 'y': 0., 'z': 0., 'visibility': 1.} for _ in range(33)]
    for left, right, y in ((11,12,.25), (23,24,.5), (25,26,.7), (27,28,.9)):
        for i, dx in ((left,-.04),(right,.04)):
            image[i].update(x=x+dx,y=y)
            world[i].update(x=dx*3,y=(y-.5)*1.8)
    return {'bbox':[x-.12,.08,x+.12,.94], 'landmarks':image, 'world_landmarks':world, 'visibility':1.}


def test_tracks_survive_detector_reordering_and_keep_independent_observations():
    frame = np.full((360,640,3), 100, np.uint8)
    tracks = PersonTracks()
    first = tracks.update([person(.25),person(.75)], frame, 0)
    second = tracks.update([person(.75),person(.25)], frame, .1)
    assert [(p['track_id'], p['bbox']) for p in first] == [(p['track_id'],p['bbox']) for p in second]
    assert all(p['observation']['usable'] for p in first)
    assert not scene_observation(first)['scene_usable']
    new = tracks.update([person(.25)],frame,2)
    assert new[0]['track_id'] not in [p['track_id'] for p in first]


def test_pose_free_ghost_needs_two_real_detector_refreshes_not_repeated_flow():
    frame=np.full((360,640,3),100,np.uint8);tracks=PersonTracks()
    real=person(.25)
    candidate={'bbox':[.6,.1,.9,.9],'landmarks':[],'world_landmarks':[],'visibility':0.,'detector_score':.6,'box_source':'ssdlite_coco_person'}
    first=tracks.update([real,candidate],frame,0)
    assert len(first)==1
    candidate['box_source']='optical_flow'
    assert len(tracks.update([real,candidate],frame,.1))==1
    candidate['box_source']='ssdlite_coco_person'
    confirmed=tracks.update([real,candidate],frame,.5)
    assert len(confirmed)==2 and confirmed[0]['track_id']==first[0]['track_id']
    assert len(tracks.update([real],frame,.6))==1


def test_one_frame_pose_hallucination_does_not_create_a_second_person():
    frame=np.full((360,640,3),100,np.uint8);tracks=PersonTracks()
    real=person(.25)
    ghost={**person(.75),'box_source':'pose_extent'}
    first=tracks.update([real],frame,0)
    measured=tracks.update([real,ghost],frame,.15)
    assert [p['track_id'] for p in measured]==[first[0]['track_id']]
    assert measured[0]['observation']['usable']
    assert len(tracks.update([real],frame,.3))==1
    assert len(tracks.update([real,ghost],frame,.6))==1, 'separated ghost frames are not continuous evidence'


def test_persistent_pose_only_person_is_confirmed_without_hiding_real_multipeople():
    frame=np.full((360,640,3),100,np.uint8);tracks=PersonTracks()
    real=person(.25)
    candidate={**person(.75),'box_source':'pose_extent'}
    assert len(tracks.update([real,candidate],frame,0))==1
    assert len(tracks.update([real,candidate],frame,.05))==1
    confirmed=tracks.update([candidate,real],frame,.15)
    assert len(confirmed)==2 and {p['track_id'] for p in confirmed}=={1,2}
    assert len(tracks.update([real,candidate],frame,.2))==2


def test_partial_and_overlapping_bodies_do_not_produce_confident_structures():
    frame = np.full((360,640,3), 100, np.uint8)
    p = person()
    assert observation(p,frame)['spatial_valid']
    p['landmarks'][23]['visibility'] = .1
    assert not observation(p,frame)['usable']
    assert observation(p,frame)['video_usable'], 'partial RGB evidence must still reach classification'
    p = person(); p['bbox'][1] = 0
    assert observation(p,frame)['clipped'] and not observation(p,frame)['spatial_valid']
    assert observation(p,frame)['video_usable']
    assert not observation(person(),frame,True)['usable']
    p = person(); p['world_landmarks'][11]['x'] = float('nan')
    assert not observation(p,frame)['spatial_valid']


def test_dark_enhancement_preserves_frames_and_uses_consistent_temporal_transform():
    dark = np.full((40,60,3), 12, np.uint8)
    corrected = enhance_clip([dark, dark.copy()])
    assert corrected[0].mean() > dark.mean()*3 and np.array_equal(corrected[0],corrected[1])
    assert dark.mean() == 12
    bright = np.full_like(dark,100)
    assert enhance_clip([bright])[0] is bright
    assert enhance_clip([dark],trained=False)[0] is dark, 'legacy enhancement regressed noisy-night recall'
    assert not enhance_clip([np.zeros_like(dark)])[0].any()
    assert observation(person(),np.full((360,640,3),20,np.uint8))['usable'], 'recoverable darkness must still be analyzed'


def test_person_buffers_are_bounded_and_partial_observations_clear_only_one_person():
    frame = np.full((360,640,3),100,np.uint8)
    people = PersonTracks().update([person(.25),person(.75)],frame,0)
    buffers = PersonClips()
    for i in range(70): buffers.sample(frame, people, i*.15)
    assert len(buffers.ready()) == 2 and all(len(p['frames']) <= 35 for p in buffers.ready())
    people[0]['observation']['spatial_valid'] = False
    buffers.sample(frame,people,10.5)
    assert len(buffers.ready()) == 2, 'invalid 3D must not suppress available RGB'
    people[0]['observation']['video_usable'] = False
    buffers.sample(frame,people,10.5)
    assert [p['track_id'] for p in buffers.ready()] == [people[1]['track_id']]
    buffers.sample(frame,[],10.6)
    assert buffers.ready() == []


def test_scores_are_person_specific_and_generation_reset_clears_them():
    worker = VideoModelWorker(Path('unused'))
    people = [{'track_id':i, 'start_s':0., 'end_s':2., 'quality':{'usable':True}} for i in (1,2)]
    worker._accept_people(people,people,[{'fall_score':.1},{'fall_score':.9}],0)
    results = worker.person_predictions()
    assert results[1]['state'] == 'adl' and results[2]['state'] == 'fall_candidate'
    assert results[1]['fall_score'] != results[2]['fall_score'] and not results[2]['validated']
    later = deepcopy(people); later[1]['end_s'] = 3
    worker._accept_people(later,[later[1]],[{'fall_score':.95}],0)
    assert worker.person_predictions()[2]['state'] == 'fall_sustained'
    later[0]['quality']['spatial_valid'] = False
    later[0]['end_s'] = 3
    worker._accept_people(later,[later[0]],[{'fall_score':.85}],0)
    assert worker.person_predictions()[1]['state'] == 'partial_fall_candidate'
    worker.reset()
    worker._accept_people(people,people,[{'fall_score':.9},{'fall_score':.9}],0)
    assert worker.person_predictions() == {}


def test_global_high_score_is_not_a_fall_when_person_observation_is_invalid():
    worker = VideoModelWorker(Path('unused'))
    prediction = {'fall_score':.99,'embedding':np.ones(2),'attention_jpeg':b'cam','attention_class':'fall','device':'cpu'}
    for end in (2,3):
        assert worker._accept(prediction,0,end,0,{'usable':True,'scene_usable':False,'observation_note':'body clipped'},1,10)
    result = worker.snapshot()
    assert result['state'] == 'observation_limited' and not result['is_fall'] and result['action'] == 'unknown'


def test_partial_training_metadata_survives_source_reset_and_accepts_visible_rgb():
    worker = VideoModelWorker(Path('unused'))
    worker._result['partial_body_trained'] = True
    generation = worker.reset()
    assert worker.snapshot()['partial_body_trained']
    prediction = {'fall_score':.1,'embedding':np.ones(2),'attention_jpeg':b'cam','attention_class':'adl','device':'cpu'}
    assert worker._accept(prediction,0,2,generation,{'usable':True,'scene_usable':False,'scene_video_usable':True},1,10)
    assert worker.snapshot()['state'] == 'adl'
