# 论文研究交接 · 2026-10-16

**给接手人的入口：[Word交接说明](论文交接说明.docx) · [PDF说明](论文交接说明.pdf) · [Markdown全文](论文交接说明.md)**

下载完整材料：[科研资料发布页](https://github.com/yjkira-01/monitor/releases/tag/research-handoff-20261016)。

## 最新事实

12288帧主实验、几何OOF及同观测公平对照已完成，并通过独立核验。正常光照几何门控为59.2348mm；同观测MLP为60.1238mm。综合S几何门控81.5475mm、同观测MLP81.4391mm、9帧GRU79.7126mm。几何自由残差正常60.2021/S81.5609mm，未胜同观测MLP。

这是一个主体划分、三个初始化的开发结果。最终保留3/10/12/18未评价；目前没有“新算法全面胜强基线”或“创新性已证明”的结论。

## 下载内容

- paper-writing-kit：交接说明、相关论文与核读、BibTeX、LaTeX模板、全部主表/逐seed结果、PNG300dpi/PDF/SVG图、历史研究材料和源码。
- research-reproduction：主实验与消融的权重、预测、OOF载荷、全部实际使用开发缓存、固定RTMO和X-Fi核心源码。
- research-history：早期融合探索及native X-Fi训练结果、最终tokens_v3缓存和权重；适合追溯负结果和原始训练问题。
- dataset_release.zip：公开mRI原始雷达、标签、时间与标定文件，校验值与官方Dryad一致。
- 15GB原始视频通过DATA_SOURCES.json中的官方源恢复，实际开发缓存已包含；未重复镜像大视频。

单个分卷小于GitHub资产限制；见[合并说明](ASSEMBLE.md)、Release中的assemble_handoff.py和SHA256SUMS.txt。不同资料包解压到同一父目录。

## 快速使用

1. 只看论文：下载writing-kit，先读交接说明与[写作材料映射](WRITING_MAP.md)。
2. 复核主结果：再下载reproduction分卷及原始dataset_release.zip，运行包内tools/verify_results.py。
3. 从头审计或重训：按[复现说明](REPRODUCE.md)配置环境并保留原目录名称；原冻结源码在[research](../../research)。
4. 原始视频重审计：按官方源下载、校验并执行restore_raw_videos.py，仅提取16开发人color0。

数字来源：[全方法CSV](large_protocol_all_methods.csv)、[独立指标复算](independent_metric_verification.json)。文档经Word原生渲染检查，图表按单图PNG/PDF/SVG导出；实验相关授权与第三方许可分别保留。

原research笔记保留历史判断，遇到状态/结论变化请以本说明和Release的完整正式报告为准。

下载后可运行 [verify_downloads.py](verify_downloads.py) 校验已下载文件；加 `--all` 要求完整资料全部存在。该工具只读文件，不删除或修改数据。
