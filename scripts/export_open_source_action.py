"""Export the official 1.39M-parameter STGCN++ NTU60-2D checkpoint, without training."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
from project_archive import ensure_dependency
sys.path[:0] = [str(ensure_dependency('action_export_dependencies')), str(ROOT / '.deps')]
import numpy as np
import torch
import onnx
import onnxruntime as ort
import mmaction

# The published wheel contains optional localizers with unrelated dependencies.
# Import only the unmodified official skeleton backbone and utilities.
for name in ('mmaction.models', 'mmaction.models.backbones'):
    module = types.ModuleType(name)
    module.__path__ = [str(Path(mmaction.__file__).parent.joinpath(*name.split('.')[1:]))]
    sys.modules[name] = module
STGCN = importlib.import_module('mmaction.models.backbones.stgcn').STGCN
from prepare_open_source_models import fetch

URL = ('https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/'
       'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/'
       'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth')
out = ROOT / 'artifacts/open_source_research_20261009'
weights = out / 'stgcnpp_ntu60_2d.pth'
if not weights.exists():
    weights.write_bytes(fetch(URL))


class ExportModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = STGCN(gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn',
                              graph_cfg=dict(layout='coco', mode='spatial'))
        self.cls_head = torch.nn.Module()
        self.cls_head.fc = torch.nn.Linear(256, 60)

    def forward(self, x):
        features = self.backbone(x)
        return torch.softmax(self.cls_head.fc(features.mean(dim=(1, 3, 4))), dim=-1)


torch.set_num_threads(3)
model = ExportModel().eval()
# This is a known official checkpoint. Only its state_dict is used.
checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['state_dict'], strict=True)
assets = ROOT / 'desktop/CareMonitor/Assets/Realtime'
path = out / 'action_candidate.onnx'
torch.manual_seed(112)
fixture = torch.rand(1, 2, 100, 17, 3)
fixture[:, 1] = 0
with torch.inference_mode():
    expected = model(fixture).numpy()
torch.onnx.export(model, (fixture,), str(path), input_names=['poses'], output_names=['probabilities'],
                  opset_version=17, dynamo=False)
onnx.checker.check_model(str(path), full_check=True)
session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
actual = session.run(None, {'poses': fixture.numpy()})[0]
error = float(np.max(np.abs(actual - expected)))
if error > 1e-5:
    raise RuntimeError(f'Export parity failed: {error}')
names = (out / 'ntu_labels.txt').read_text(encoding='utf-8').splitlines()
if len(names) != 60 or names[42] != 'falling':
    raise ValueError('Action label ordering is incompatible')
metadata = {
    'name': 'STGCN++ NTU60-2D', 'source': URL, 'license': 'Apache-2.0',
    'package': 'mmaction2==1.2.0', 'parameters': sum(p.numel() for p in model.parameters()),
    'checkpoint_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'max_export_error': error,
    'input': [1, 2, 100, 17, 3], 'classes': names,
    'status': 'research_candidate',
    'notes': 'NTU60 action pretraining is not validation on elderly falls. Static sitting/lying use separate observations.',
}
(out / 'action_candidate.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
(out / 'action_fixture.json').write_text(json.dumps({'input': fixture.numpy().ravel().tolist(),
    'scores': actual.ravel().tolist(), 'max_error': error}), encoding='utf-8')
print(json.dumps({k: v for k, v in metadata.items() if k != 'classes'}), flush=True)
