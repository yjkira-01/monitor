# 跌倒二分类权重选择与复测

云端训练已按用户要求停止，本轮只使用既有权重和官方预训练模型做导出、比较与应用接入。

## 保存的权重

| 权重 | 来源与 SHA256 | 当前用途 |
|---|---|---|
| GMD R3D-18 | `artifacts/gmdcsa24_r3d18_100ep_inference.pt`；`8226e722fb9245af0976868e730ea0d00156f7c4472b554c4f915cd7a01349fd` | 设置中的早期对照 |
| Stage C R3D-18 | `artifacts/care_video_active_inference.pt`；`7b53215ebb060fa65cf99352cf6324e1b93c6369576ba8057e384421cf917266` | 默认跌倒二分类 |

早期 GMD 的 35/37 是整片均匀采样的开发验证结果。它不能代表实时滑窗、GUI 告警、多人、老人或暗光场景的准确率。

## 同协议滑窗比较

GMD Subject 4 的全部 37 段，C# / DirectML，约 8 Hz 采样、最多 32 帧缓冲。整段至少两次分数 ≥ 0.5 判阳性，两个窗不要求连续。这是约四秒窗口的整段开发指标；实际桌面应用按时间维护五秒窗口，并通过人员跟踪与异步刷新输出事件，两个协议不同。

| 权重 | TP | FN | FP | TN |
|---|---:|---:|---:|---:|
| GMD | 13 | 4 | 0 | 20 |
| Stage C | 17 | 0 | 3 | 17 |

原始记录：`artifacts/open_source_research_20261009/binary_gmd_37_native.json`、`binary_stagec_37_native.json`。默认保留 Stage C，设置提供 GMD 对照。对已授权用户视频做过数值回归，GMD 阳性而 Stage C 只有一次高分窗；与历史五秒窗口、不同下采样下的绝对分值不可直接比较。

## 拒绝接入的候选

- `melihuzunoglu/human-fall-detection` 的 YOLO11，2.59M 参数，三类。固定 12 段开发片段，仅 1 TP、5 FN、0 FP、6 TN，未进入应用。
- `yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect`，86.2M 参数，实际七类，归并 FallDown 后 5 TP、1 FN、1 FP、5 TN。本机 CPU 推理约 1.8 秒，未接入默认模型。CC-BY-NC-4.0，作者未明确训练数据。
- 通用 NTU60 STGCN++，对该组跌倒片段没有足够表现，默认禁用且发布排除。17 点 RTMPose/RTMO 不会被伪映射给旧 33 点 TCN。

## 应用中的输入与证据

稳定单人窗口保留整景：窗口内无其他已知人员 ID，至少六帧且同一 ID 占至少 80%。多人分别使用人体裁剪，场景分值不会分配给指定人员；裁剪输入仍存在与整景训练不同的局限。

二维姿态与动作时序观测单独显示，不能称为训练得到的多类别模型概率。公开视频实时回放的后续证据位于 `artifacts/action_upgrade_20261010`，该回放使用实际 `VideoService`，不可与本页四秒整段协议混用。
