import pytest
import torch

from scripts import index_multifall,train_video
from yanglao_monitor.video import ACTIVITY_CLASSES


def test_tempered_sampling_equalizes_domains_without_repeating_rare_clip_unboundedly():
    rows=[{'dataset':'large','label':'walk','split':'train'} for _ in range(900)]
    rows += [{'dataset':'large','label':'sit_down','split':'train'} for _ in range(2)]
    rows += [{'dataset':'small','label':'walk','split':'train'} for _ in range(20)]
    weights=train_video.sampling_weights(rows,True)
    assert sum(weights[:902]) == pytest.approx(1)
    assert sum(weights[902:]) == pytest.approx(1)
    assert weights[900] > weights[0]
    assert max(weights[:902]) <= 5/902
    legacy=train_video.sampling_weights(rows)
    assert sum(legacy[:900]) == pytest.approx(sum(legacy[900:902]))
    with pytest.raises(ValueError):train_video.sampling_weights([{'split':'test'}],True)


def test_risk_supervision_and_teacher_do_not_teach_the_wrong_label():
    classes=sorted(ACTIVITY_CLASSES)
    logits=torch.zeros(2,10,requires_grad=True)
    labels=torch.tensor([classes.index('lying'),classes.index('fall')])
    original=train_video.activity_loss(logits,labels,classes,.3)
    wrong_teacher=torch.tensor([[-8.,8.],[8.,-8.]],requires_grad=True)
    skipped=train_video.activity_loss(logits,labels,classes,.3,wrong_teacher,.2)
    assert skipped.item() == original.item()
    right_teacher=-wrong_teacher
    preserved=train_video.activity_loss(logits,labels,classes,.3,right_teacher,.2)
    assert preserved.item() > original.item()
    preserved.backward()
    assert torch.isfinite(logits.grad).all() and wrong_teacher.grad is None
    assert logits.grad[1,classes.index('fall')] < 0
    assert logits.grad[0,classes.index('fallen')] > 0
    grouped=train_video.risk_logits(logits,classes).softmax(1)[:,1]
    assert torch.allclose(grouped,logits.softmax(1)[:,[classes.index('fall'),classes.index('fallen')]].sum(1))


def test_low_quality_augmentation_preserves_time_range_and_source():
    clips=torch.rand(3,3,1,24,32).expand(-1,-1,8,-1,-1).clone()
    original=clips.clone()
    degraded=train_video.quality_view(clips)
    assert degraded.shape == clips.shape and torch.equal(degraded[:,:,0],degraded[:,:,-1])
    assert degraded.min() >= 0 and degraded.max() <= 1
    assert torch.equal(original,clips) and not torch.equal(degraded,clips)
    torch.manual_seed(0)
    result=train_video.prepare_batch(clips,augment=True,quality=True)
    assert torch.equal(result[:,:,0],result[:,:,-1])
    assert torch.equal(clips,original)
    with pytest.raises(ValueError):train_video.quality_view(clips,0)


def test_domain_selection_rejects_hidden_regression_and_ignores_absent_classes():
    classes=sorted(ACTIVITY_CLASSES)
    labels=[classes.index('fall'),classes.index('lying'),classes.index('fall')]
    scores=torch.eye(10)[labels].tolist()
    rows=[{'dataset':d,'path':str(i)} for i,d in enumerate(['A','A','B'])]
    by_domain=train_video.domain_metrics(scores,labels,rows,classes)
    assert by_domain['B']['supported_macro_f1']==1
    assert by_domain['B']['val_macro_f1']==pytest.approx(.1)
    baseline={'val_macro_f1':.8,'fall_recall':.9,'val_acc':.9,
              'by_dataset':{'A':{'supported_macro_f1':.8},'B':{'supported_macro_f1':.8}}}
    candidate={**baseline,'val_macro_f1':.9,'by_dataset':{'A':{'supported_macro_f1':.95},'B':{'supported_macro_f1':.3}}}
    bad_rank=train_video.selection_rank(candidate,baseline)
    assert bad_rank[0]==0 and not candidate['regression_gate_passed']
    acceptable={**baseline,'by_dataset':{'A':{'supported_macro_f1':.81},'B':{'supported_macro_f1':.79}}}
    assert train_video.selection_rank(acceptable,baseline)>bad_rank
    assert acceptable['worst_dataset_macro_f1']==.79
    with pytest.raises(ValueError):train_video.domain_metrics(scores,labels,rows[:1],classes)


def test_extended_author_metadata_uses_subjects_and_groups_cameras_and_internet_sources(tmp_path,monkeypatch):
    monkeypatch.setattr(index_multifall,'ROOT',tmp_path)
    directory=tmp_path/'data/raw/OmniFall/EDF/EDF'
    path=directory/'clips/jianjun/jianjun_jianjun_view1_rgb_1_10.000_12.000_1_1_edf.avi'
    path.parent.mkdir(parents=True);path.touch()
    first=index_multifall.annotation_row('jianjun/'+path.name+' 1','EDF',directory,True)
    second_path=path.with_name(path.name.replace('view1','view2').replace('_1_1_edf','_1_2_edf'));second_path.touch()
    second=index_multifall.annotation_row('jianjun/'+second_path.name+' 1','EDF',directory,True)
    assert first['split']==second['split']=='train' and first['parent_id']==second['parent_id']
    with pytest.raises(ValueError):index_multifall.annotation_row('jianjun/'+path.name+' 4','EDF',directory,True)
    wild=tmp_path/'wild'
    for source,action in [('Compilation42',15),('Compilation59',1)]:
        path=wild/f'clips/falls/{source}_{action}_0.000_2.000_-1_-1_OOPS.avi'
        path.parent.mkdir(parents=True,exist_ok=True);path.touch()
        row=index_multifall.annotation_row('falls/'+path.name+f' {action}','OOPS',wild,True)
        if action==15:
            assert row['label']=='other' and row['action_id']==15
            reference=row
        else:assert row['parent_id']==reference['parent_id'] and row['split']==reference['split']
    train_video.validate_splits([first,second,reference,row])


def test_training_smoke_retains_classes_and_never_reads_test_split():
    from scripts.run_multiscene_training import smoke_rows
    rows=[{'dataset':'A','split':split,'label':label} for split in ('train','val','test') for label in ACTIVITY_CLASSES for _ in range(4)]
    selected=smoke_rows(rows)
    assert len(selected)==40 and all(r['split']!='test' for r in selected)


def test_updated_oops_labels_correct_old_fall_and_reject_unmapped_or_mixed_intervals(tmp_path):
    from pathlib import Path
    mapping=tmp_path/'mapping.csv'
    mapping.write_text('oops_path,itw_path\noops_video/train/Example Clip42.mp4,falls/ExampleClip42.mp4\n')
    labels=tmp_path/'labels.csv'
    labels.write_text('path,label,start,end\nfalls/ExampleClip42,8,0,2\nfalls/ExampleClip42,1,2,2.5\nfalls/ExampleClip42,2,2.5,5\n')
    tables=index_multifall.oops_author_tables(labels,mapping)
    corrected=index_multifall.resolve_oops_clip(Path('42_Example_Clip42_1_0.000_1.500_-1_-1_OOPS.avi'),*tables)
    assert corrected['action_id']==8 and corrected['author_coverage']==1
    assert index_multifall.resolve_oops_clip(Path('42_Example_Clip42_1_1.800_2.700_-1_-1_OOPS.avi'),*tables) is None
    assert index_multifall.resolve_oops_clip(Path('99_Unknown99_1_0.000_1.000_-1_-1_OOPS.avi'),*tables) is None
    fallen=index_multifall.resolve_oops_clip(Path('42_Example_Clip42_9_3.000_4.000_-1_-1_OOPS.avi'),*tables)
    assert fallen['action_id']==2


def test_index_excludes_window_that_first_frame_check_misses(tmp_path,monkeypatch):
    import csv
    import json
    import sys
    import cv2
    import numpy as np
    monkeypatch.setattr(index_multifall,'ROOT',tmp_path)
    directory=tmp_path/'data/raw/OmniFall/EDF/EDF'
    (directory/'clips').mkdir(parents=True)
    (tmp_path/'artifacts').mkdir()
    lines=[]
    for subject,name in [(1,'good'),(2,'bad')]:
        filename=f'{name}_view1_rgb_1_0.000_1.000_{subject}_1_edf.avi'
        writer=cv2.VideoWriter(str(directory/'clips'/filename),cv2.VideoWriter_fourcc(*'MJPG'),5,(16,16))
        assert writer.isOpened()
        for _ in range(5):writer.write(np.full((16,16,3),subject*60,np.uint8))
        writer.release();lines.append(filename+' 1')
    (directory/'mmaction2_annotations_edf.txt').write_text('\n'.join(lines))
    def decode(path,*args):
        if path.name.startswith('bad'):raise RuntimeError('No decodable sampled frame')
        return torch.zeros(3,16,16,16)
    monkeypatch.setattr(index_multifall,'_read_frames',decode)
    monkeypatch.setattr(sys,'argv',['index_multifall.py','--project-root',str(tmp_path),'--multi-state','--datasets','EDF','--output','data/manifests/preflight.csv'])
    index_multifall.main()
    with (tmp_path/'data/manifests/preflight.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==1 and 'good' in rows[0]['path']
    report=json.loads((tmp_path/'artifacts/preflight_index.json').read_text())
    assert report['excluded'][0]['reason']=='training sampler cannot decode this window'


def test_index_quarantines_prefix_padding_and_uses_mcfd_source_timebase(tmp_path,monkeypatch):
    import csv,json,sys,cv2
    import numpy as np
    monkeypatch.setattr(index_multifall,'ROOT',tmp_path)
    (tmp_path/'artifacts').mkdir()
    for dataset,filename,fps,count in [
        ('EDF','padded_view1_rgb_1_10.000_10.500_1_1_edf.avi',30,90),
        ('MCFD','chute01_cam1_1_10.000_16.000_1_1_mcfd.avi',120,180)]:
        directory=tmp_path/f'data/raw/OmniFall/{dataset}/{dataset}'
        (directory/'clips').mkdir(parents=True)
        writer=cv2.VideoWriter(str(directory/'clips'/filename),cv2.VideoWriter_fourcc(*'MJPG'),fps,(16,16))
        assert writer.isOpened()
        for i in range(count):writer.write(np.full((16,16,3),i%240,np.uint8))
        writer.release()
        (directory/f'mmaction2_annotations_{dataset.lower()}.txt').write_text(filename+' 1')
    monkeypatch.setattr(sys,'argv',['index_multifall.py','--project-root',str(tmp_path),'--multi-state','--datasets','EDF','MCFD','--output','data/manifests/timebase.csv'])
    index_multifall.main()
    rows=list(csv.DictReader((tmp_path/'data/manifests/timebase.csv').open(encoding='utf-8-sig')))
    assert len(rows)==2 and all(r['dataset']=='MCFD' for r in rows)
    assert float(rows[0]['end_s'])==pytest.approx(1.25)
    assert float(rows[-1]['end_s'])==pytest.approx(179/120,abs=1e-6)
    assert float(rows[-1]['start_s'])==pytest.approx(29/120,abs=1e-6)
    report=json.loads((tmp_path/'artifacts/timebase_index.json').read_text())
    assert report['excluded'][0]['reason']=='decoded duration disagrees with author interval; alignment unverified'
    assert report['timebase_audit'][0]['source_fps']==30


def test_known_fall_audit_uses_late_fixed_windows_without_saving_frames(tmp_path,monkeypatch):
    import cv2
    import numpy as np
    from scripts import audit_video_regression
    path=tmp_path/'regression.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(16,16))
    assert writer.isOpened()
    for _ in range(36):writer.write(np.full((16,16,3),80,np.uint8))
    writer.release()
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'mock predictor')
    class Predictor:
        clip_len=16
        def __init__(self,*args):pass
        def predict(self,frames):
            assert len(frames)==16 and all(f.shape==(16,16,3) for f in frames)
            return {'fall_score':.8,'activity':None}
    monkeypatch.setattr(audit_video_regression,'TrainedVideoModel',Predictor)
    report=audit_video_regression.audit(path,checkpoint,tmp_path/'report.json')
    assert report['late_window_detected']
    assert all(p['end_s']-p['start_s']==pytest.approx(5) for p in report['predictions'])
    assert {p.name for p in tmp_path.iterdir()}=={'regression.avi','checkpoint.pt','report.json'}


def test_real_domain_selected_ten_class_training_exports_metrics_and_gates(tmp_path,monkeypatch):
    import argparse
    import csv
    import json
    from pathlib import Path
    classes=sorted(ACTIVITY_CLASSES)
    manifest=tmp_path/'data/manifests/states.csv';manifest.parent.mkdir(parents=True)
    rows=[{'path':f'{split}/{domain}/{i}.avi','split':split,'label':label,'dataset':domain,
           'subject':f'{domain}:{split}','parent_id':f'{split}:{i}'}
          for split in ('train','val') for domain in ('A','B') for i,label in enumerate(classes)]
    with manifest.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)
    class Dataset:
        def __init__(self,manifest,split,*args,**kwargs):
            self.root=tmp_path;self.classes=classes;self.class_to_idx={c:i for i,c in enumerate(classes)}
            self.samples=[(tmp_path/r['path'],self.class_to_idx[r['label']]) for r in rows if r['split']==split]
        def __len__(self):return len(self.samples)
        def __getitem__(self,i):return torch.full((3,4,16,16),self.samples[i][1]/10),self.samples[i][1]
    monkeypatch.setattr(train_video,'VideoManifestDataset',Dataset)
    args=argparse.Namespace(manifest=str(manifest),model='tiny',multi_state=True,clip_len=4,size=16,epochs=2,batch_size=10,
                           workers=0,lr=.001,weight_decay=.001,amp=False,cache_clips=True,resume=None,
                           output=str(tmp_path/'states.pt'),cpu=True,domain_balanced_sampling=True,domain_selection=True,
                           quality_augment=True,risk_loss_weight=.3)
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:train_video.train(args)
    finally:torch.set_num_threads(threads)
    report=json.loads((tmp_path/'states_evaluation.json').read_text())
    assert set(report['metrics']['by_dataset'])=={'A','B'}
    assert isinstance(report['metrics']['regression_gate_passed'],bool)
    assert set(report['metrics']['robustness'])=={'low_quality','night_low_quality'}
    state=torch.load(tmp_path/'states_inference.pt',weights_only=True)
    assert len(state['training_manifest_sha256'])==64 and state['classes']==classes
