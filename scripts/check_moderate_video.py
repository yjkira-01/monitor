"""Real video forward/backward and synchronized measurements before the long run."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from yanglao_monitor.foundation_video import build_internvideo2, normalize_video, layer_parameters, PRETRAINED_SHA256
from yanglao_monitor.video import VideoManifestDataset
from train_video import activity_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(9)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16 CUDA GPU is required for this training experiment')
    dataset = VideoManifestDataset(args.manifest,'train',8,(224,224),preserve_aspect=True)
    indices = [next(i for i,(_,label) in enumerate(dataset.samples) if dataset.classes[label]==name) for name in ('fall','lying')]
    examples = [dataset[i] for i in indices]
    clips = normalize_video(torch.stack([x for x,_ in examples]).transpose(1,2).cuda())
    labels = torch.tensor([y for _,y in examples],device='cuda')
    assert torch.isfinite(clips).all() and clips.std() > 0
    model = build_internvideo2(10,args.weights).cuda()
    count = sum(p.numel() for p in model.parameters())
    assert 80_000_000 < count < 100_000_000
    model.eval()
    inference_times = []
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        model(clips)
        for _ in range(3):
            torch.cuda.synchronize(); start = time.perf_counter()
            prediction = model(clips)
            torch.cuda.synchronize(); inference_times.append((time.perf_counter()-start)*1000)
    assert prediction.shape == (2,10) and torch.isfinite(prediction).all()
    model.train()
    optimizer = torch.optim.AdamW(layer_parameters(model,2e-5),weight_decay=.05)
    before = model.blocks[-1].attn.qkv.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss = activity_loss(model(clips),labels,dataset.classes,.25)
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(); train_ms = (time.perf_counter()-start)*1000
    assert (before-model.blocks[-1].attn.qkv.weight).abs().max()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    report = {'purpose':'functional and resource validation, not accuracy evaluation',
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__, 'parameters':count,
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'weight_sha256':PRETRAINED_SHA256,'classes':dataset.classes,
              'input':[2,3,8,224,224],'dtype':'BF16 autocast; FP32 parameters/optimizer',
              'inference_batch_ms':inference_times,'forward_backward_step_ms':train_ms,
              'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
              'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'loss':float(loss),
              'all_parameter_gradients_finite':True,'passed':True}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
