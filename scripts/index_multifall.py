"""Fixed subject splits for GMDCSA24 and the verified OmniFall video clips."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from yanglao_monitor.video import ACTIVITY_CLASSES,_read_frames
SPLITS = {"CAUCAFall": ({8}, {9, 10}), "UP-Fall": ({12, 13, 14}, {15, 16, 17}), 'GMDCSA24':({4},set()), 'EDF':({4},{5}),
          'LE2I':({5},{2,7}),'OCCU':({4},{5}),'MCFD':(set(),set()),'OOPS':(set(),set())}
ARCHIVES = {'CAUCAFall':'Cauca_fall','UP-Fall':'UP_Fall','GMDCSA24':'GMDCSA24','EDF':'EDF','LE2I':'LE2I','MCFD':'MCFD','OCCU':'OCCU','OOPS':'OOPS_1k'}
SUBJECT_LIMITS = {'CAUCAFall':10,'UP-Fall':17,'GMDCSA24':4,'EDF':5,'LE2I':8,'MCFD':1,'OCCU':5}
FIELDS = ["path", "split", "label", "subject", "start_s", "end_s", "dataset", "parent_id", "action_id", "annotation_source",'archive_action_id','author_path','author_coverage']


def annotation_row(line, dataset, directory, multi_state=False):
    name, label = line.strip().rsplit(maxsplit=1)
    action = int(label)
    if action not in range(16 if dataset == 'OOPS' else 10):
        raise ValueError(f"Invalid action: {action}")
    if action == 2 and not multi_state:
        return None  # Fallen is neither a falling transition nor a safe normal activity.
    path = directory / "clips" / name.replace("\\", "/")
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Annotation path escaped dataset directory")
    if dataset in {'EDF','LE2I','MCFD','OCCU','OOPS'}:
        return extended_annotation_row(path,action,dataset,directory,multi_state)
    match = re.fullmatch(r"Subject\.(\d+)", path.parent.name)
    if not match or not path.is_file():
        raise ValueError(f"Missing video or subject: {name}")
    subject = int(match.group(1))
    if not 1 <= subject <= SUBJECT_LIMITS[dataset]:
        raise ValueError("Unexpected subject ID")
    val, test = SPLITS[dataset]
    split = "test" if subject in test else "val" if subject in val else "train"
    # First filename segment identifies the parent action, including camera for UP-Fall.
    parent = path.stem.split("_")[0]
    if dataset == "UP-Fall":
        parent = re.sub(r"Camera\d+$", "", parent)
    label_name = ACTIVITY_CLASSES[action] if multi_state else 'fall' if action == 1 else 'adl'
    return {"path": path.relative_to(ROOT).as_posix(), "split": split, "label": label_name,
            "subject": f"{dataset}:{subject}", "dataset": dataset, "parent_id": f"{subject}:{parent}",
            "action_id": action, "annotation_source": "zenodo17170592_author_temporal_clip"}


def extended_annotation_row(path,action,dataset,directory,multi_state):
    match = re.fullmatch(r'(?P<source>.+)_(?P<action>\d+)_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)_(?P<subject>-?\d+)_(?P<cam>-?\d+)_(?P<dataset>[A-Za-z0-9_-]+)',path.stem)
    if not match or int(match['action']) != action or match['dataset'].lower() != dataset.lower() or float(match['start']) >= float(match['end']):
        raise ValueError(f'Invalid author clip metadata: {path.name}')
    if not path.is_file():raise FileNotFoundError(path)
    subject = int(match['subject'])
    source = match['source']
    if dataset == 'OOPS':
        if subject != -1 or int(match['cam']) != -1:raise ValueError('Unexpected OOPS identity')
        # shortcut: compilation grouping cannot certify actor isolation; use consented field data for that audit.
        parent = re.sub(r'[_\d]+$','',re.sub(r'^\d+_','',source))
        if not parent:raise ValueError('Missing internet source group')
        bucket = int(hashlib.sha256(parent.encode()).hexdigest()[:8],16)%10
        split = 'test' if bucket >= 8 else 'val' if bucket == 7 else 'train'
        subject_name = 'unknown'
    else:
        if not (0 if dataset == 'LE2I' else 1) <= subject <= SUBJECT_LIMITS[dataset]:
            raise ValueError('Unexpected subject ID')
        val,test = SPLITS[dataset]
        split = 'test' if subject in test else 'val' if subject in val else 'train'
        subject_name = f'{dataset}:{subject}'
        parent = re.sub(r'(?:_view\d+_rgb|_?cam\d+)$','',source,flags=re.IGNORECASE)
    label = ACTIVITY_CLASSES[min(action,9)] if multi_state else 'fall' if action == 1 else 'adl'
    return {'path':path.relative_to(ROOT).as_posix(),'split':split,'label':label,'subject':subject_name,
            'dataset':dataset,'parent_id':f'{subject_name}:{parent}','action_id':action,
            'annotation_source':'zenodo17170592_author_temporal_clip_extended_to_other' if action>=10 else 'zenodo17170592_author_temporal_clip'}


def oops_author_tables(labels_path,mapping_path):
    from collections import defaultdict
    normalize = lambda value:re.sub(r'[^a-z0-9]','',value.lower())
    mapping,annotations={},defaultdict(list)
    with Path(mapping_path).open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            key=normalize(Path(row['oops_path']).stem)
            value=Path(row['itw_path']).with_suffix('').as_posix()
            if key in mapping and mapping[key]!=value:raise ValueError('Ambiguous author video mapping')
            mapping[key]=value
    with Path(labels_path).open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            start,end,action=float(row['start']),float(row['end']),int(row['label'])
            if not np.isfinite([start,end]).all() or not 0<=start<end or action not in range(16):
                raise ValueError('Invalid author OOPS interval')
            annotations[row['path']].append((start,end,action))
    return mapping,annotations


def resolve_oops_clip(path,mapping,annotations):
    match=re.fullmatch(r'(.+)_(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_-1_-1_OOPS',path.stem)
    if not match:raise ValueError('Unexpected OOPS clip filename')
    source=re.sub(r'^\d+_','',match[1])
    key=mapping.get(re.sub(r'[^a-z0-9]','',source.lower()))
    if key is None:return None
    start,end=float(match[3]),float(match[4])
    coverage=Counter()
    for a,b,label in annotations.get(key,[]):coverage[label]+=max(0,min(end,b)-max(start,a))
    if not coverage:return None
    action,overlap=coverage.most_common(1)[0]
    if overlap/(end-start)<.8:return None
    return {'action_id':action,'author_path':key,'author_coverage':round(overlap/(end-start),6)}


def gmd_activity_rows(labels):
    rows,excluded = [],[]
    with Path(labels).open(encoding='utf-8-sig') as handle:
        annotations = list(csv.DictReader(handle))
    for source in annotations:
        match = re.fullmatch(r'Subject_(\d+)/(ADL|Fall)/(\d+)',source['path'])
        if not match or int(match[1]) not in range(1,5) or int(source['label']) not in range(10):
            raise ValueError('Unexpected GMD author annotation')
        subject = int(match[1]);path = ROOT/f'data/raw/GMDCSA24/Subject {subject}'/match[2]/(match[3]+'.mp4')
        if not path.is_file():raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path));count,fps=cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS);cap.release()
        start,end=float(source['start']),float(source['end'])
        if not np.isfinite([count,fps,start,end]).all() or count<2 or fps<=0 or start<0 or start>=end:
            raise ValueError('Invalid GMD annotation interval or video')
        end = min(end,(count-1)/fps)
        if end-start<.4:
            excluded.append({'path':source['path'],'reason':'less than 0.4s valid labeled evidence','start_s':start,'end_s':end});continue
        ends=np.linspace(min(start+5,end),end,min(3,int(np.ceil((end-start)/5))))
        for window_end in ends:
            rows.append({'path':path.relative_to(ROOT).as_posix(),'split':'val' if subject==4 else 'train',
                         'label':ACTIVITY_CLASSES[int(source['label'])],'subject':f'GMDCSA24:{subject}',
                         'start_s':round(max(start,window_end-5),6),'end_s':round(float(window_end),6),'dataset':'GMDCSA24',
                         'parent_id':path.relative_to(ROOT).as_posix(),'action_id':int(source['label']),
                         'annotation_source':'OmniFall_author_GMDCSA24_temporal_csv'})
    return rows,excluded


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(SPLITS), default=['CAUCAFall','UP-Fall'])
    parser.add_argument("--output", default="data/manifests/multifall_v2.csv")
    parser.add_argument('--multi-state',action='store_true',help='保留作者十类动作标签，包括跌倒后倒地；不推测GMD粗标签')
    parser.add_argument('--project-root',default=str(ROOT),help='隔离源码快照时的数据所在项目目录')
    parser.add_argument('--gmd-labels',help='OmniFall原始GMD时间标签，复用已有RGB而非重复下载视频')
    parser.add_argument('--base-manifest',help='复用已校验的十类清单，仅索引新增来源')
    parser.add_argument('--oops-labels',help='作者更新后的OOPS时间标签；不直接采用旧1k归档二分类标签')
    parser.add_argument('--oops-mapping',help='作者原视频名与更新OOPS标签路径的精确映射')
    args = parser.parse_args()
    ROOT = Path(args.project_root).resolve()
    rows = []
    if args.base_manifest:
        if not args.multi_state:parser.error('Base manifest reuse requires ten-class labels')
        with Path(args.base_manifest).open(encoding='utf-8-sig') as handle:
            rows=[{field:row.get(field,'') for field in FIELDS} for row in csv.DictReader(handle)]
        if any(row['label'] not in ACTIVITY_CLASSES for row in rows):raise ValueError('Base manifest contains incompatible labels')
        if any(row['dataset'] in args.datasets for row in rows):raise ValueError('Base and new datasets overlap')
    if not args.multi_state:
        with (ROOT/"data/manifests/gmdcsa24_windows.csv").open(encoding="utf-8-sig") as handle:
            rows = [{key: r.get(key, "") for key in FIELDS} for r in csv.DictReader(handle)]
    if not args.multi_state:
        for row in rows:
            row["subject"] = f"GMDCSA24:{row['subject']}"
            row["parent_id"] = row["path"]
    excluded, hashes, timebase_audit = [], set(), []
    oops_tables=None
    if 'OOPS' in args.datasets:
        if not args.oops_labels or not args.oops_mapping:parser.error('OOPS requires updated author labels and mapping')
        oops_tables=oops_author_tables(args.oops_labels,args.oops_mapping)
    for dataset in args.datasets:
        if dataset == 'GMDCSA24' and args.multi_state and args.gmd_labels:
            labeled,ignored = gmd_activity_rows(args.gmd_labels)
            rows.extend(labeled);excluded.extend(ignored);continue
        archive = ARCHIVES[dataset]
        archive_root = ROOT/'data/raw/OmniFall'/archive
        candidates = list(archive_root.rglob('mmaction2_annotations*.txt'))
        if len(candidates) != 1:raise ValueError(f'Expected one author annotation list for {dataset}, got {len(candidates)}')
        annotations = candidates[0]
        directory = annotations.parent
        for line in annotations.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:row = annotation_row(line,dataset,directory,args.multi_state)
            except FileNotFoundError as exc:
                excluded.append({'path':str(exc),'reason':'author annotation references a missing release clip'});continue
            if row is None:
                continue
            path = ROOT/row["path"]
            if dataset=='OOPS':
                verified=resolve_oops_clip(path,*oops_tables)
                if verified is None:
                    excluded.append({'path':row['path'],'reason':'unmapped or less than 80% coverage by one updated author class'});continue
                if not args.multi_state and verified['action_id']==2:continue
                row.update(archive_action_id=row['action_id'],**verified,
                           label=ACTIVITY_CLASSES[min(verified['action_id'],9)] if args.multi_state else 'fall' if verified['action_id']==1 else 'adl',
                           annotation_source='OmniFall_updated_author_OOPS_temporal_overlap_80pct')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                excluded.append({"path": row["path"], "reason": "byte-identical duplicate"})
                continue
            hashes.add(digest)
            cap = cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]) if hasattr(cv2,'CAP_PROP_N_THREADS') else cv2.VideoCapture(str(path))
            if not cap.isOpened():cap.release();cap=cv2.VideoCapture(str(path))
            count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
            ok, first = cap.read()
            decoded_count = 1 if ok else 0
            extended = dataset in {'EDF','MCFD','LE2I','OCCU','OOPS'}
            if extended:
                while cap.grab():decoded_count += 1
            cap.release()
            if not ok or not np.isfinite([count, fps]).all() or count < 2 or fps <= 0:
                excluded.append({"path": row["path"], "reason": "unreadable/too short"})
                continue
            source_fps = fps
            if extended:
                match = re.fullmatch(r'.+_\d+_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_-?\d+_-?\d+_[A-Za-z0-9_-]+',path.stem)
                author_duration = float(match[2])-float(match[1])
                # MCFD clips contain 30 frames per annotated second; their AVI headers report 120 FPS.
                source_fps = 30. if dataset == 'MCFD' else fps
                audit = {'path':row['path'],'header_frames':count,'decoded_frames':decoded_count,'header_fps':fps,
                         'source_fps':source_fps,'author_duration_s':author_duration,'decoded_duration_s':decoded_count/source_fps}
                if decoded_count < 2 or abs(decoded_count/source_fps-author_duration) > 2/source_fps+.002:
                    excluded.append({**audit,'reason':'decoded duration disagrees with author interval; alignment unverified'});continue
                timebase_audit.append(audit)
                count = decoded_count
            source_duration = (count-1)/source_fps
            # A long ADL segment contributes at most three bounded windows, not hundreds of repeated frames.
            ends = np.linspace(min(source_duration, 5.), source_duration, min(3, int(np.ceil(source_duration/5))))
            for end in ends:
                window={**row,'start_s':round(max(0.,end-5)*source_fps/fps,6),'end_s':round(float(end)*source_fps/fps,6)}
                if extended:
                    try:_read_frames(path,16,(16,16),window['start_s'],window['end_s'])
                    except (RuntimeError,ValueError) as exc:
                        excluded.append({'path':row['path'],'start_s':window['start_s'],'end_s':window['end_s'],
                                         'reason':'training sampler cannot decode this window','error':str(exc)});continue
                rows.append(window)
    sys.path.insert(0, str(ROOT))
    from scripts.train_video import validate_splits
    validate_splits(rows)
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {"manifest": args.output, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "windows": len(rows),
              "counts": dict(Counter(f"{r['dataset']}/{r['split']}/{r['label']}" for r in rows)),
              "subjects": {split: sorted({r["subject"] for r in rows if r["split"] == split}) for split in ("train", "val", "test")},
              "video_files": len({r["path"] for r in rows}),
              "parent_actions": len({(r["dataset"], r["parent_id"]) for r in rows}), "excluded": excluded,
              "semantics": "Author's ten action classes retained; no inferred GMD activity labels." if args.multi_state else "Fall transition=1; fallen=2 excluded; other annotated actions are ADL. Test subjects are never read by training or model selection."}
    report['split_limitations'] = {'MCFD':'One annotated actor: training only; cameras are not new subjects.',
                                 'OOPS':'Unknown actors; compilation source groups are hashed into train/val/test. Not a cross-subject guarantee.',
                                 'LE2I':'Published author cross-subject IDs: validation 5, test 2/7.'}
    report['timebase_audit'] = timebase_audit
    report['timebase_semantics'] = 'New-source decoded duration must match author interval within two frames; MCFD source 30 FPS maps to AVI header time. Unverified padded clips excluded, not tail-cropped by guess.'
    if args.base_manifest:report['base_manifest_sha256']=hashlib.sha256(Path(args.base_manifest).read_bytes()).hexdigest()
    (ROOT/"artifacts"/(output.stem+"_index.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"excluded","timebase_audit"}}, indent=2))


if __name__ == "__main__":
    main()
