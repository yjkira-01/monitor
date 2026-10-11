"""Export existing accepted runtime weights; training candidates require separate acceptance."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from yanglao_monitor.pose_temporal import PoseTemporalNet, pose_features


def export(model, sample, path, outputs, tolerance=2e-4):
    model.eval()
    with torch.inference_mode():
        expected = model(sample)
    torch.onnx.export(model, sample, str(path), input_names=['input'], output_names=outputs,
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'input': sample.numpy()})
    expected = expected if isinstance(expected, tuple) else (expected,)
    errors = []
    for a, e in zip(actual, expected):
        errors.append(float(np.max(np.abs(a-e.numpy()))) if a.size else 0.)
        np.testing.assert_allclose(a, e.numpy(), atol=tolerance, rtol=1e-4)
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
            'max_errors': errors, 'input': list(sample.shape), 'outputs': outputs}


class VideoExport(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.net = r3d_18(weights=None)
        self.net.fc = nn.Linear(512, len(state['classes']))
        self.net.load_state_dict(state['model'])
        self.preprocess = R3D_18_Weights.DEFAULT.transforms()

    def forward(self, frames):
        # Floating input avoids platform-specific uint8 interpolation and rounding.
        clip = frames.permute(0, 1, 4, 2, 3) / 255.
        clip = torch.nn.functional.interpolate(clip.flatten(0,1),size=(128,171),mode='bilinear',align_corners=False)
        clip = clip[:,:,8:120,30:142]
        mean = clip.new_tensor([.43216,.394666,.37645]).view(1,3,1,1)
        std = clip.new_tensor([.22803,.22145,.216989]).view(1,3,1,1)
        clip = ((clip-mean)/std).view(1,16,3,112,112).permute(0,2,1,3,4)
        features=self.net.layer4(self.net.layer3(self.net.layer2(self.net.layer1(self.net.stem(clip)))))
        embedding=self.net.avgpool(features).flatten(1)
        probabilities=self.net.fc(embedding).softmax(1)
        weights=self.net.fc.weight[probabilities.argmax(1)]
        cam=torch.relu((features*weights[:,:,None,None,None]).sum(1).mean(1))
        low=cam.flatten(1).amin(1)[:,None,None];high=cam.flatten(1).amax(1)[:,None,None]
        cam=(cam-low)/(high-low).clamp_min(1e-6)
        return probabilities,embedding,cam


class DetectorExport(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.net.load_state_dict(torch.load(ROOT/'artifacts/ssdlite320_mobilenet_v3_large_coco-a79551df.pth',
                                           weights_only=True, map_location='cpu'))
    def forward(self, image):
        result = self.net([image[0]])[0]
        return result['boxes'], result['scores'], result['labels']


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261009)
    assets = ROOT/'desktop/CareMonitor/Assets'
    assets.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT/'artifacts/care_video_active_inference.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['model_name'] == 'r3d18' and state['clip_len'] == 16
    sample = torch.rand(1, 16, state['read_size'][1], state['read_size'][0], 3)*255
    video = export(VideoExport(state), sample, assets/'video.onnx', ['probabilities','embedding','attention'])
    video.update(classes=state['classes'], revision=checkpoint.stem, read_size=state['read_size'],
                 low_light_trained=state.get('low_light_trained', False),
                 partial_body_trained=state.get('partial_body_trained', False),
                 acceptance='existing_runtime_baseline', clinical_validated=False,
                 source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    print('video exported and parity checked', flush=True)
    pose_state = torch.load(ROOT/'artifacts/multifall_pose_tcn_stage_b_inference.pt', map_location='cpu', weights_only=True)
    assert pose_state['clip_len'] == 48 and pose_state['classes'] == ['adl','fall']
    net = PoseTemporalNet();net.load_state_dict(pose_state['model'])
    features = pose_features(torch.rand(1, 3, 48, 33))
    pose = export(net, features, assets/'pose_tcn.onnx', ['logits'])
    pose.update(classes=['adl','fall'], clip_len=48, preprocessing=pose_state['preprocessing'])
    print('pose TCN exported and parity checked', flush=True)
    detector = export(DetectorExport(), torch.rand(1, 3, 320, 320), assets/'people.onnx', ['boxes','scores','labels'], .02)
    print('detector exported and parity checked', flush=True)
    landmark = assets/'pose_landmarks_detector_full.onnx'
    assert landmark.is_file()
    landmark_info = {'sha256':hashlib.sha256(landmark.read_bytes()).hexdigest(),
                     'source':'https://huggingface.co/unity/inference-engine-blaze-pose', 'license':'Apache-2.0',
                     'joints':33, 'outputs_include_auxiliary_joints':True,
                     'preprocessing':'square_body_roi_rgb_float_0_1'}
    report = {'video':video, 'pose_tcn':pose, 'people':detector, 'landmarks':landmark_info,
              'note':'Export parity is numerical equivalence, not real-scene accuracy validation.'}
    (assets/'models.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT/'artifacts/desktop_model_export_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
