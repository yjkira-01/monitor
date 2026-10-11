import argparse
import json
from pathlib import Path

import pytest
import torch

from scripts import index_multifall, train_video
from yanglao_monitor.video import ACTIVITY_CLASSES


def test_multistate_metrics_distinguish_lying_from_fallen_and_report_all_classes():
    classes = sorted(ACTIVITY_CLASSES)
    probabilities = torch.eye(10).tolist()
    metrics = train_video.activity_metrics(probabilities,list(range(10)),classes)
    assert metrics['val_acc'] == metrics['val_macro_f1'] == metrics['fall_recall'] == 1
    assert len(metrics['confusion_matrix']) == 10 and metrics['per_class']['lying']['support'] == 1
    labels = [classes.index('lying'),classes.index('fallen')]
    scores = [probabilities[classes.index('lying')]]*2
    wrong = train_video.activity_metrics(scores,labels,classes)
    assert wrong['risk_confusion_matrix'] == [[1,0],[1,0]]
    with pytest.raises(ValueError):train_video.activity_metrics([[float('nan')]*10],[0],classes)


def test_author_fallen_label_is_retained_only_in_multistate_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr(index_multifall,'ROOT',tmp_path)
    directory = tmp_path/'data/raw/OmniFall/UP_Fall/UP_Fall'
    path = directory/'clips/Subject.12/Subject12Activity1Trial1Camera2_2_1.0_5.0.avi'
    path.parent.mkdir(parents=True);path.touch()
    line = 'Subject.12/'+path.name+' 2'
    assert index_multifall.annotation_row(line,'UP-Fall',directory) is None
    row = index_multifall.annotation_row(line,'UP-Fall',directory,True)
    assert row['label'] == 'fallen' and row['split'] == 'val'
    assert 'Camera' not in row['parent_id']


def test_published_gmd_state_intervals_preserve_label_boundaries(tmp_path,monkeypatch):
    import cv2
    import numpy as np
    monkeypatch.setattr(index_multifall,'ROOT',tmp_path)
    path=tmp_path/'data/raw/GMDCSA24/Subject 4/ADL/01.mp4';path.parent.mkdir(parents=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(16,16))
    assert writer.isOpened()
    for _ in range(30):writer.write(np.zeros((16,16,3),np.uint8))
    writer.release()
    labels=tmp_path/'labels.csv';labels.write_text('path,label,start,end,subject,cam,dataset\nSubject_4/ADL/01,5,0.2,1.4,4,1,GMDCSA24\nSubject_4/ADL/01,6,1.4,3.2,4,1,GMDCSA24\nSubject_4/ADL/01,4,0,0.1,4,1,GMDCSA24\n')
    rows,excluded=index_multifall.gmd_activity_rows(labels)
    assert [r['label'] for r in rows]==['lie_down','lying']
    assert rows[0]['start_s']==.2 and rows[0]['end_s']==1.4
    assert rows[1]['start_s']==1.4 and rows[1]['end_s']==2.9
    assert all(r['split']=='val' for r in rows) and len(excluded)==1


def test_real_ten_class_training_exports_compatible_scores(tmp_path,monkeypatch):
    class Dataset:
        def __init__(self,root,*args,**kwargs):
            self.root = Path(root);self.classes = sorted(ACTIVITY_CLASSES)
            self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
            self.samples = [(self.root/f'{i}.avi',i) for i in range(10)]
        def __len__(self):return len(self.samples)
        def __getitem__(self,index):return torch.full((3,4,16,16),index/10),index
    monkeypatch.setattr(train_video,'VideoFolderDataset',Dataset)
    args = argparse.Namespace(data_root=str(tmp_path),manifest=None,model='tiny',multi_state=True,clip_len=4,size=16,
                              epochs=1,batch_size=10,workers=0,lr=.001,weight_decay=.001,amp=False,cache_clips=False,
                              resume=None,output=str(tmp_path/'states.pt'),cpu=True)
    threads = torch.get_num_threads();torch.set_num_threads(1)
    try:train_video.train(args)
    finally:torch.set_num_threads(threads)
    state = torch.load(tmp_path/'states_inference.pt',weights_only=True)
    assert state['classes'] == sorted(ACTIVITY_CLASSES)
    report = json.loads((tmp_path/'states_evaluation.json').read_text())
    assert len(report['metrics']['per_class']) == 10
    for p in report['predictions']:
        assert p['predicted'] in ACTIVITY_CLASSES and len(p['activity_scores']) == 10
        assert p['fall_probability'] == p['activity_scores']['fall']+p['activity_scores']['fallen']


def test_live_multistate_batch_keeps_independent_actions_and_combines_risk_classes():
    import numpy as np
    from app.video_model import TrainedVideoModel
    live=TrainedVideoModel.__new__(TrainedVideoModel);live.torch=torch;live.device='cpu';live.classes=sorted(ACTIVITY_CLASSES)
    live.prepare_clip=lambda frames:torch.ones(1,1,1,1,1)
    scores=torch.full((2,10),.01);scores[0,live.classes.index('lying')]=.91;scores[1,live.classes.index('fallen')]=.81;scores[1,live.classes.index('fall')]=.11
    class Predictor(torch.nn.Module):
        def __init__(self):super().__init__();self.fc=torch.nn.Linear(1,10)
        def forward(self,clips):live.features=torch.ones(len(clips),1,2,2,2);return scores.log()
    live.model=Predictor()
    image=np.full((32,32,3),100,np.uint8)
    results=live.predict_many([[image],[image]])
    assert [p['activity']['label'] for p in results]==['lying','fallen']
    assert results[0]['fall_score']==pytest.approx(.02)
    assert results[1]['fall_score']==pytest.approx(.92)
    assert results[0]['attention_class']=='lying'
