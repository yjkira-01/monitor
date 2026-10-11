# 云端训练监控

`scripts/monitor_training.py` 使用 Python 标准库读取独立 Stage F 实验状态，不加载模型或占用 CUDA 显存，不启动、停止或恢复训练。已上传到云端项目的 `scripts/` 目录。

先从本机登录：

```powershell
ssh -p 22 root@your-cloud-host
```

也可以在 Windows 上双击 `scripts/watch_cloud_training.cmd`，输入 SSH 密码后直接进入面板。密码不保存在脚本中；入口复用项目已核验的 `artifacts/cloud_known_hosts`。从本机终端直接进入同一面板：

```powershell
ssh -t -p 22 root@your-cloud-host "/root/miniconda3/bin/python /root/autodl-fs/yanglao-monitor/scripts/monitor_training.py --interval 1"
```

登录后在云端终端运行，每 1 秒刷新：

```bash
/root/miniconda3/bin/python /root/autodl-fs/yanglao-monitor/scripts/monitor_training.py
```

查看一次后退出：

```bash
/root/miniconda3/bin/python /root/autodl-fs/yanglao-monitor/scripts/monitor_training.py --once
```

指定实验目录或刷新间隔：

```bash
/root/miniconda3/bin/python /root/autodl-fs/yanglao-monitor/scripts/monitor_training.py \
  --run /root/autodl-fs/yanglao-monitor/artifacts/activity_stage_f_internvideo2_base_20261009 \
  --interval 10
```

`--run` 接受同样文件命名和状态格式的 InternVideo2 Stage F 实验目录，不适用于旧 Stage E 的文件格式。

需要机器可读结果时使用 `--json`，该选项只查询一次：

```bash
/root/miniconda3/bin/python /root/autodl-fs/yanglao-monitor/scripts/monitor_training.py --json
```

实时面板适应终端宽高，进度条保持在顶部。窗口较小时会缩减下方明细；使用 `--once` 查看全部数据。每秒刷新表示重新读取状态，训练日志和验证数据仍按其实际写入频率变化。

监控显示：

- 管理进程和实际训练进程。通过 `/proc` 的命令行、实验路径和进程状态核验，剔除 DataLoader 子进程和僵尸进程；不单凭保存的 PID 判断任务存活。
- 已完成轮数、轮数上限、早停耐心、训练/验证/测试窗数量，以及日志最近记录的批次和 loss。
- 完整轮次进度条和当前轮次的最近批次进度条。已经保存的轮次会将旧批次条标为历史；提前结束时明确标出任务已经结束，轮数上限不当作必训轮数。
- 最新和按训练选模排序取得的最佳验证指标，包括 macro-F1、风险召回、风险精确率、正常卧姿召回和开发验收门槛。
- 最近完整一轮的平均训练 loss 和训练准确率。
- 十类动作各自的验证召回率及验证窗数，以及七种困难条件的 macro-F1。
- last / best 权重的文件大小及距写入时间。只读取文件属性，不解析 `.pt`。
- 整张 GPU 的使用率、显存和温度；其他同时运行的任务也计入这些数字。

当前加速配置每 20 个批次记录进度和实际吞吐，旧配置每 100 批记录。验证与断点保存期间可能没有新批次记录。最近批次百分比只表示该轮训练批次的已记录位置，不包含后续验证，不是整个 120 轮任务的完成百分比。第一个完整 epoch 尚未落盘时，监控会明确显示没有可用验证结果和断点。

日志中的早期失败记录不直接决定当前状态；当前状态同时参考 `status.json` 和实际进程。若用 last 权重手动恢复而状态文件仍为 `failed`，会显示训练进程在运行，同时保留状态文件报告的错误供核查。所有准确率均为开发验证指标，最终 `*_evaluation.json` 也是验证报告，不能当作独立测试集成绩。

按 **Ctrl+C** 只退出监控。已有后台训练继续；云实例关机则训练会停止。监控不会自动恢复任务。首次 epoch 中途停止仍可能丢失尚未保存的该轮训练进度。

直接跟踪原始日志或单独查看 GPU：

```bash
tail -n 30 -f /root/autodl-fs/yanglao-monitor/artifacts/activity_stage_f_internvideo2_base_20261009/training.log
```

```bash
watch -n 2 nvidia-smi
```

本地离线解析测试：

```powershell
python tests/test_monitor_training.py
```

测试覆盖首轮无历史、旧错误后正常运行、最新与最佳轮次不同、陈旧 PID、其他实验、僵尸和 DataLoader 进程、手动恢复进程、失败状态、JSON 暂不可读及恢复、有界日志读取，以及无 `/proc` 时不误称云端训练停止。

2026-10-09 实时面板验证：12 项离线测试通过，包括进度条比例、完成轮次与批次分离、早停提示和小窗口宽高适配。实际云端 80×24 终端的进度条从第二轮 500 批次更新到 600 批次；Windows `.cmd` 入口也完成了登录、实时显示、Ctrl+C 和退出验证，随后确认训练仍运行。证据在 `artifacts/cloud_training_monitor_live_20261009.json`。
