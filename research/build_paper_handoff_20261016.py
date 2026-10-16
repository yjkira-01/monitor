"""Build a traceable handoff of completed pose-fusion research, never train.

Preserve existing source bytes and experiment directory names. Large archives
use split ZIP streams so each GitHub asset stays below its size limit. Public
source datasets remain attributed; original videos have official recovery URLs.
"""
from pathlib import Path
import csv, hashlib, html, importlib.metadata, io, json, os, re, shutil, sys, zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables/paper-handoff-20261016'
WORK = OUT / 'paper_handoff_20261016'
DOCS = WORK / 'handoff'
TOP = WORK.name
TAG = 'research-handoff-20261016'
REPO = 'yjkira-01/monitor'
CORE = ['action_window_main_20261016','ray_oof_action_windows_20261016',
    'ray_raw_action_window_control_20261016','continuous_research_pipeline_20261016',
    'ray_raw_action_window_followthrough_20261016','mri_action_windows_resumed_20261016',
    'ray_oof_research_20261016','ray_oof_strict_controls_20261016',
    'ray_raw_observation_control_20261016','ray_raw_width32_control_fixed_20261016',
    'ray_raw_width32_control_20261016','fusion_method_research_20261013_expanded_centroid',
    'fusion_rgb_expanded_20261013','mri_label_geometry_20261016']
HISTORY_FULL = ['xfi_accuracy_formal_20261013','xfi_accuracy_formal_resumed_20261016',
                'xfi_stability_20261016','xfi_mri_native_tokens_v3_20261013']
STORED = {'.npz','.pt','.pth','.onnx','.zip','.pdf','.png','.jpg','.jpeg','.bin'}
MANIFEST = []
EXCLUSIONS = []


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def add(source, group, relative=None):
    if not source.is_file() or any(x in source.parts for x in ['.git','__pycache__']):
        return
    relative = Path(relative) if relative else source.relative_to(ROOT)
    target = WORK / relative
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    value = digest(target)
    assert source.stat().st_size == before.st_size and source.stat().st_mtime_ns == before.st_mtime_ns
    MANIFEST.append({'path': relative.as_posix(), 'bytes': before.st_size,
                     'sha256': value, 'bundle': group, 'provenance': 'existing research file, unchanged bytes'})


class SplitOutput:
    """Unseekable output accepted by zipfile; concatenation is a normal ZIP."""
    def __init__(self, stem, limit=1024**3):
        self.stem, self.limit, self.total, self.current_size = stem, limit, 0, 0
        self.parts=[]; self.stream=None
    def tell(self): return self.total
    def seek(self, *args): raise OSError('streaming split ZIP')
    def flush(self):
        if self.stream: self.stream.flush()
    def write(self, data):
        length=len(data); view=memoryview(data)
        while len(view):
            if self.stream is None or self.current_size==self.limit:
                if self.stream: self.stream.close()
                path=OUT/f'{self.stem}.zip.{len(self.parts)+1:03d}'
                self.parts.append(path); self.stream=path.open('wb'); self.current_size=0
            take=min(len(view),self.limit-self.current_size)
            self.stream.write(view[:take]); view=view[take:]
            self.total+=take; self.current_size+=take
        return length
    def close(self):
        if self.stream: self.stream.close(); self.stream=None


def metadata_files():
    for source in sorted((ROOT/'research').glob('*')):
        if source.is_file() and not any(k in source.name.lower() for k in ['worker_memory','review_worker_memory']):
            add(source,'writing')
    for source in [ROOT/'LICENSE',ROOT/'MODEL_NOTICES.md', ROOT/'references.md',
        ROOT/'docs/radar_video_fusion_research_20261012.md',ROOT/'docs/radar_video_fusion_references_20261012.bib']:
        add(source,'writing')
    # Exact primary experiments, cached observations and all source snapshots.
    for directory in sorted((ROOT/'artifacts').iterdir()):
        if not directory.is_dir():
            if any(k in directory.name.lower() for k in ['ray_','knap_','mri_','xfi_','fusion_','anchor_','past_anchor_','research_validation']):
                add(directory,'writing' if directory.suffix.lower() not in ['.npz','.pt'] else 'reproduction')
            continue
        name=directory.name
        if name in CORE or name.startswith('xfi_strong_inputs'):
            group='reproduction'
        elif name in HISTORY_FULL or name.startswith('fusion_method_research'):
            group='history'
        elif any(k in name for k in ['fusion_','mri_','xfi_','anchor_','past_anchor_','action_window_',
                                     'immfusion_','research_track_','paper_baseline_sources','research_execution']):
            group='metadata'
        else:
            continue
        for source in sorted(directory.rglob('*')):
            if not source.is_file() or '.git' in source.parts or '__pycache__' in source.parts:
                continue
            rel=source.relative_to(directory)
            if name=='paper_baseline_sources_20261013':
                if 'xfi_checkout' in rel.parts:
                    add(source,'reproduction')
                elif source.suffix.lower() in ['.pt','.zip'] and 'xfi_pretrained' in rel.parts:
                    add(source,'history')
                else:
                    add(source,'writing')
            elif name=='research_execution_20261012':
                if source.suffix.lower() in ['.mp4','.avi','.mov']:
                    continue # Same video bytes are in the attributed upstream ZIP.
                if any(k in source.name.lower() for k in ['cookie_session','browser_session','download_error','download_probe']):
                    continue
                add(source,'reproduction' if source.suffix.lower()=='.npz' else 'writing')
            elif group=='metadata':
                if source.suffix.lower() in ['.pt','.pth','.npz','.bin','.mp4','.zip','.npy']:
                    EXCLUSIONS.append({'path':source.relative_to(ROOT).as_posix(),'bytes':source.stat().st_size,
                        'reason':'Superseded or pilot binary; metadata/code retained. Primary and final native-baseline binaries included.'})
                else:
                    add(source,'writing')
            else:
                if source.suffix.lower() in ['.md','.csv','.json','.txt','.bib','.tex','.log','.html','.xml','.pdf','.png','.svg'] and source.stat().st_size<24*1024**2:
                    add(source,'writing')
                else:
                    add(source,group)
    # RTMO needed to repeat raw-pixel cache audits, with upstream notices.
    for source in [ROOT/'desktop/CareMonitor/Assets/Realtime/rtmo.onnx',ROOT/'desktop/CareMonitor/Assets/Realtime/rtmo.json']:
        add(source,'reproduction')
    notice=ROOT/'desktop/CareMonitor/Assets/Notices'
    if notice.exists():
        for source in notice.rglob('*'): add(source,'writing')
    add(ROOT/'artifacts/research_execution_20261012/mri_license.txt','writing')


def table_and_figures():
    main=read_json(ROOT/'artifacts/ray_oof_action_windows_20261016/reporting/results.json')
    raw=read_json(ROOT/'artifacts/ray_raw_action_window_control_20261016/reporting/results.json')
    methods=dict(main['aggregate']); methods['raw_observations']=raw['aggregate']['raw_observations']
    labels={'raw_observations':'相同观测神经网络（1843参数）','recentered_concat':'强拼接，髋重中心',
            'radar_ray_gate':'雷达深度几何门控','constant_ray_gate':'常深度几何门控',
            'free_radar_residual':'雷达几何自由残差','free_constant_residual':'常深度自由残差',
            'free_no_ray_residual':'无射线自由残差','gru':'GRU（9帧）','temporal_mlp':'时序MLP（9帧）',
            'awc_concat':'AWC论文训练规则适配'}
    result=[]
    for name,a in methods.items():
        result.append({'method':name,'description':labels.get(name,name),'clean_mean_mm':a['clean_mm']['mean'],
            'clean_initialization_sd_mm':a['clean_mm']['initialization_sd'],
            'S_mean_mm':a['S_mm']['mean'],'S_initialization_sd_mm':a['S_mm']['initialization_sd']})
    DOCS.mkdir(parents=True,exist_ok=True)
    with (DOCS/'large_protocol_all_methods.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    write(DOCS/'large_protocol_all_methods.json',json.dumps({'population':12288,'train_frames':9216,'validation_frames':3072,
        'folds':[0],'seeds':[7,17,27],'test_subjects_used':[],'records':result},ensure_ascii=False,indent=2))
    # Plot only archived numbers; never recompute model outputs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    names=['recentered_concat','awc_concat','gru','temporal_mlp','raw_observations','free_constant_residual','free_radar_residual','radar_ray_gate']
    captions=['Strong concatenation','AWC adaptation','GRU (9 frames)','Temporal MLP (9 frames)',
              'Same-observation MLP','Constant-depth residual','Radar-depth residual','Radar-depth gate']
    fd=DOCS/'figures';fd.mkdir(exist_ok=True)
    for metric,xlabel in [('clean_mm','Clean MPJPE (mm)'),('S_mm','Five-condition subject-macro error (mm)')]:
        fig,ax=plt.subplots(figsize=(8,5),dpi=300)
        values=[methods[n][metric]['mean'] for n in names];sd=[methods[n][metric]['initialization_sd'] for n in names]
        colors=['#8192a3','#8192a3','#c47c34','#c47c34','#49896d','#6783a1','#3777a0','#3777a0']
        ax.barh(np.arange(len(names)),values,xerr=sd,color=colors,capsize=3,height=.63)
        ax.set_yticks(np.arange(len(names)),captions);ax.invert_yaxis();ax.set_xlabel(xlabel)
        ax.set_xlim(0,max(v+s for v,s in zip(values,sd))*1.12);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for y,v in enumerate(values):ax.text(v+2,y,f'{v:.2f}',va='center',fontsize=9)
        for suffix in ['png','pdf','svg']:fig.savefig(fd/f'large_{metric}.{suffix}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    write(fd/'captions.md','''# 图注与统计口径

large_clean_mm：完整12288帧开发协议的正常光照根相对MPJPE。12训练主体、4开发验证主体，一个划分、三个初始化。误差线是初始化样本标准差，不是置信区间。图内所有方法均使用同一3072验证帧；GRU及MLP具有9帧历史，几何/同观测头是单帧。

large_S_mm：同人口下正常、暗噪、遮挡、视频缺失、雷达缺失五条件的受试者等权误差。图显示GRU综合误差更低；几何残差未优于同观测普通MLP。不得只展示正常条件宣称全面更优。

原始数值：handoff/large_protocol_all_methods.csv；正式证据：artifacts/ray_oof_action_windows_20261016/reporting/results.json、artifacts/ray_raw_action_window_control_20261016/reporting/results.json。
''')
    return methods,result


def make_documents(methods,rows):
    order=['recentered_concat','awc_concat','gru','temporal_mlp','raw_observations','free_constant_residual','free_radar_residual','radar_ray_gate','constant_ray_gate']
    lookup={r['method']:r for r in rows}
    lines=['| 方法 | 正常MPJPE / mm | 综合S / mm |','|---|---:|---:|']
    for name in order:
        r=lookup[name];lines.append(f"| {r['description']} | {r['clean_mean_mm']:.4f} ± {r['clean_initialization_sd_mm']:.4f} | {r['S_mean_mm']:.4f} ± {r['S_initialization_sd_mm']:.4f} |")
    table='\n'.join(lines)
    write(DOCS/'论文交接说明.md',f'''# 视频与毫米波人体姿态融合：论文交接说明

交接快照：{datetime.now().astimezone().isoformat(timespec='seconds')}。用途：交给其他研究者继续实验、核查和撰稿。源仓库：https://github.com/{REPO} 。

## 1. 接手人先读的结论

主实验、几何OOF实验和同观测公平对照都已完成，并有独立核验。最终保留主体3/10/12/18尚未评价。目前没有证据支持“创新算法全面优于强基线”，论文方向与最终方法仍需研究者确定。

小数据1536帧中几何自由残差59.2732mm，优于同观测同宽度MLP63.1860mm。但扩大到12288帧后，同观测MLP正常60.1238/S81.4391mm，几何自由残差60.2021/S81.5609mm，原优势没有保留。单帧几何门控正常59.2348mm更低，综合S81.5475mm仍高于同观测MLP；9帧GRU综合S79.7126mm最好。不得沿用旧记忆中的“正在跑”或仅用小数据正结果作为最终结论。

## 2. 研究问题与实际数据

研究问题：RGB关键点与毫米波点云在退化、遮挡、缺模态下如何有效融合，显式相机几何是否比同观测神经表示更有价值？这是mRI人体姿态估计研究，不是跌倒检测或临床有效性实验。

mRI数据源：An、Li、Ogras，数据DOI https://doi.org/10.5061/dryad.9ghx3ffpp 。公开元数据标记CC0，许可原文与源元数据均保留；原始发布数据、标定与时间戳没有重新命名或伪造。

完整开发人口16人×12动作×64帧，共12288。训练主体1/2/4/7/8/9/11/13/14/15/16/17，共9216帧；验证主体5/6/19/20，共3072帧。每动作在1/3及2/3位置各取连续32帧。三初始化7/17/27，均固定80轮末轮，没有按验证选择checkpoint。

全部有限发布标签为主人口。已在正式训练前发现少量169/252/296m级异常标签，保留并另报预设1m四肢骨长/3m根半径敏感性；所有主方法共用稳健MPJPE和训练坐标中位数模板，另保留MSE控制。发布标签源于双相机，不是独立动捕真值。一个主体划分和多个相邻帧不能当作多个独立受试者验证。

真实像素条件为clean、noisy_dark（0.12亮度＋Gaussian标准差8）、固定矩形occlusion；另有视频或雷达完全缺失。S是这五条件、四验证主体的等权误差。旧1536帧另有dark条件，但dark未计入S；两个开发人口不混算。

## 3. 已实现的算法与公式

强融合基础网络将冻结RTMO检测的51维姿态表示与196×5雷达点集编码融合，输出17关节根相对3D。先统一预测双髋重中心，再构造几何候选。

雷达14维统计包含XYZ三个分位数9维、Doppler/Intensity的均值与标准差4维、非零行比例1维。训练主体内用IRLS Huber（0.15m阈值、ridge1、25步）拟合相机根深度，输入median/MAD标准化，预测裁剪0.5–10m；缺雷达用训练深度中位数。GT根深度只用作训练监督或显式标注的oracle，不作为部署输入。

令基础姿态为 $b_j$，相机内参为 $K$、旋转为 $R$，检测位置为 $(u_j,v_j)$。归一化射线为

$$r_j=((u_j-c_x)/f_x,(v_j-c_y)/f_y).$$

按实现的标签轴符号变换转到相机坐标，保留基础相对深度 $z_j$，以估计根深度 $d$ 得到

$$a_j=(r_{{jx}}(z_j+d),r_{{jy}}(z_j+d),z_j+d).$$

减双髋中心后旋转回标签坐标即几何候选 $c_j$。关节与双髋检测confidence均至少0.3时，$I_j=1$；否则不校正。

自由残差头输入15维观测/几何特征和16维关节嵌入，经32维隐藏层输出：

$$\hat y_j=b_j+I_j\,0.1\tanh(f_\theta(x_j,e_j)).$$

单位为m，每轴修正限制0.1m，1395参数；AdamW lr0.001/wd0.0001、batch256、梯度裁剪1、80轮末轮。门控头则输出sigmoid权重，沿 $c_j-b_j$ 校正。精确坐标符号、实现及归一统计以源码和快照为准。

12训练人内三折各8人训练、4人留出，基础预测及距离拟合都没有用该留出主体拟合；校正头仅训练这些OOF载荷。部署评价读取冻结full12基础模型。OOF8到full12存在误差分布偏移。新OOF专家的batch64、full12主训练batch128，整流程训练计算量高于普通单阶段基线，不能声称完全同更新次数。

相同观测MLP看到基础3D及双髋、原始归一化2D及双髋、完整静态R、置信度、估计距离和雷达可用性。它具有hidden32/embedding16、1843参数，控制了几何头的隐藏宽度且参数更多。几何与raw头共享OOF数据、优化器、80轮、修正上限和mask。

## 4. 最新完整主结果

{table}

±为三个初始化的样本标准差，不是独立划分误差或置信区间。完整18方法CSV及逐seed/逐主体结果在本包内。

单帧几何门控相对强concat正常降低约4.5261mm、S降低约2.3547mm；但相对同观测MLP正常只低约0.8891mm、S高约0.1084mm。自由几何残差相对同观测MLP两项均略差。雷达深度相对常深度自由残差的S差约0.1323mm、正常反而略差0.0262mm。因此雷达深度独立贡献较小，不能把校准视觉几何的全部收益归于雷达。

## 5. 基线与相关论文

AWC为《Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation》的Shapley贡献与Fisher训练规则在本地两模态骨干上的重建，非MM-Fi官方完整模型。X-Fi保留官方X_Fusion核心，commit 9ba74cc98beb2d75a8eadbe74b1b151dc3d39614；强输入适配旧1536人口正常73.8105/S97.5552mm，不能与新人口直接算胜负。历史raw-CNN适配训练不稳与低学习率改善也全部保留，属于常规训练优化。

Knap等ICEIC2024（DOI10.1109/ICEIC61013.2024.10457094）完整4页已核读，原文Eq.(1)包含透视提升及雷达全局放置。Human3.6M上PA-MPJPE37.2/N-MPJPE61.7mm与本研究mRI根相对MPJPE不同。Amin等ICEET2023（DOI10.1109/ICEET60227.2023.10526102）前作摘要已核验、全文未取得。不能声称首次雷达解除单目尺度歧义。

文献阅读程度、DOI记录、原始摘要、已取得全文、作者代码及许可随包保存；未取得全文的文献不补造表格。理论工具OOF、残差、透视几何、稳健回归均有先例。

## 6. 论文作者可以使用的材料

可直接据实撰写：实验数据口径、算法实现、同人口基线比较、同观测公平控制、消融及负结果、标签异常和预处理审计。创新性、跨数据集效果、独立最终测试和实际现场泛化尚无足够证据，不应写成完成结论。

推荐写作顺序：先实验设置与可复现材料，再方法和全部消融，随后相关工作与讨论；摘要和最终贡献待方向确定后定稿。附有LaTeX工作模板、BibTeX、主表CSV、独立PNG300dpi/PDF/SVG结果图及图注。模板不预填虚假的最终贡献。

后续优先事项：固定比较对象与主指标；增加主体划分；解释几何优势为何随数据扩大消失；必要时检验训练预算与表示学习的关系；取得更直接同方向论文的完整可复现实现；方法与协议固定后才进行保留四人评价。当前没有新训练进程要接管，两条已登记流程状态均complete。

## 7. 包内目录和复现方式

写作包、主复现包、历史包解压到同一父目录，均恢复paper_handoff_20261016根目录。原始dataset_release.zip另下载放根目录。先运行tools/verify_results.py做不重训的CPU指标复核，再按REPRODUCE.md执行GPU完整重建审计。

15GB blurred_videos.zip未复制进交接压缩包，DATA_SOURCES.json给出官方Dryad下载地址、大小及SHA256。实际训练/验证使用的全部开发缓存、标定来源、点云与预测已包含，读表和复算指标不需要先下载大视频。像素端到端重审计时再下载、校验并按restore_raw_videos.py提取16开发人的color0；最终保留四人不解码。

Python环境按原版本记录，PyTorch2.11.0+cu128、NumPy2.3.4、ONNX Runtime GPU1.25.1、RTX4060 Laptop 8GB、IEEE FP32。原始源码保留哈希，不为迁移手改冻结脚本；若换GPU/依赖，先做数值重载检查。pip依赖目录与系统CUDA DLL不复制入包，用requirements重新安装。

## 8. 审计与交接完整性

MANIFEST.json记录每个交付文件的字节数、SHA256和所属包；SHA256SUMS.txt记录Release资产校验值。每个分卷ZIP通过合并视图检查CRC；Word文件经OOXML结构检查并渲染抽检。历史试运行和过时二进制的取舍明列EXCLUSIONS.json；正式主实验、几何、raw对照、强输入X-Fi及最终native X-Fi三组目标/低lr的权重和载荷均保留。

旧research笔记和报告属于当时状态，发生矛盾时以本交接快照和latest完整报告为准。根本地MEMORY可能包含与研究无关的设备/账户上下文，不公开；研究沿革已由实验目录与本说明承接。
''')
    write(WORK/'START_HERE.md',f'''# 论文研究交接入口

先打开 **handoff/论文交接说明.docx** 或 [Markdown交接说明](handoff/论文交接说明.md)。

**最新状态：12288帧全部实验和公平对照已完成。大数据上几何自由残差未优于同观测MLP，GRU综合误差更低；请保留负结果。最终4人未评价。**

- 写论文：交接说明、handoff/large_protocol_all_methods.csv、handoff/figures/、handoff/paper_template/。
- 读正式证据：artifacts/ray_oof_action_windows_20261016/reporting/results.md、artifacts/ray_raw_action_window_control_20261016/reporting/results.md。
- 复核数值：tools/verify_results.py。完整复现：REPRODUCE.md。
- 实现：research/；权重、逐帧预测及独立审计：artifacts/。
- 原始数据恢复入口：DATA_SOURCES.json；数据许可与第三方源码许可分别保留。
- 完整性：MANIFEST.json、SHA256SUMS.txt；历史二进制取舍：EXCLUSIONS.json。

发布地址：https://github.com/{REPO}/releases/tag/{TAG} 。多个包解压到同一个父目录。分卷.zip.001等按序拼接后再解压，见ASSEMBLE.md。
''')
    write(WORK/'REPRODUCE.md',r'''# 复现说明

## 0. 环境和目录

推荐Python3.13、NVIDIA CUDA环境。原实验Torch2.11.0+cu128、NumPy2.3.4、RTX4060 Laptop 8GB；精确已安装核心版本见ENVIRONMENT.json。创建虚拟环境后安装requirements-research.txt，PyTorch从https://download.pytorch.org/whl/cu128安装对应版本。Windows建议在可写目录解压并从paper_handoff_20261016根目录运行。

不要执行旧本机流水线launcher来“续跑”：本快照实验均已complete，launcher要求新目录。已有保存权重可直接重审计、复算报告。下面审计可能重写审计receipt/报告；如需保留交接快照，先复制解压目录。

## 1. 不重训的CPU指标核查

    python tools/verify_results.py

它从主预测NPZ独立计算欧氏误差，与交接主表逐方法逐seed核对。它不替代原始图像推理、权重前向重建或主体分离审计。

## 2. 原始数据与像素审计

从Release下载dataset_release.zip放根目录。若只读表与已缓存特征训练，不需要原视频。要重跑完整像素审计，从DATA_SOURCES.json官方源恢复blurred_videos.zip，确认SHA256；运行tools/restore_raw_videos.py只提取开发16人color0，再执行：

    python research/audit_mri_action_windows.py artifacts/mri_action_windows_resumed_20261016

RTMO固定ONNX已经包含在desktop/CareMonitor/Assets/Realtime/rtmo.onnx，来源与许可随包。不要用不同模型的检测结果替换缓存后继续引用旧数字。

## 3. 保存权重完整GPU重建

    python research/audit_action_window_research.py artifacts/action_window_main_20261016 --cache artifacts/mri_action_windows_resumed_20261016
    python research/report_action_window_research.py artifacts/action_window_main_20261016
    python research/audit_ray_oof_action_windows_20261016.py
    python research/report_ray_action_window_research.py
    python research/audit_report_ray_raw_action_window_control_20261016.py

几何审计会检查OOF训练/留出主体、训练内重新拟合距离、专家观测重建、CUDA及NumPy头部重建。上述脚本预设目录必须保持原名称。原始数据ZIP、所有research源码、主基线/几何/raw载荷均来自主复现包。

## 4. 从头训练（会耗时）

    python research/run_action_window_formal_20261016.py --cache artifacts/mri_action_windows_resumed_20261016 --output artifacts/action_window_retrained --folds 0 --seeds 7 17 27 --epochs 80 --temporal-epochs 80

新输出目录的训练可以独立完成；几何原冻结脚本默认读取原action_window_main_20261016且要求输出目录不存在。因此整流程重训应在新的解压副本中将旧结果目录移出、保持同名目录再按原流水线执行，或明确生成另一个runner并保存修改说明/新SHA。不要把新运行的数据填回旧结果冒充完全一致。

旧1536帧实验的入口是run_ray_oof_research.py、run_ray_oof_strict_controls.py、run_ray_raw_observation_control.py、run_ray_raw_width32_control_fixed_20261016.py；独立审计脚本同目录。X-Fi强输入核心：run_xfi_strong_inputs.py及resume源码，官方X-Fi固定代码在artifacts/paper_baseline_sources_20261013/xfi_checkout。历史native X-Fi复现需另外解压history包，包含tokens_v3及完整权重。

## 5. 最终保留集

3/10/12/18没有模型预测或评价。公开原始ZIP本来含20人，但所有主缓存和训练仅16开发人。交接包没有安排最终测试任务。方法和协议冻结后，由接手研究者决定是否运行最终测试，并单独记录。
''')
    write(WORK/'ASSEMBLE.md',f'''# 下载与合并分卷

用GitHub CLI一次下载：

    gh release download {TAG} --repo {REPO} --dir downloads

先运行下载目录中的verify_downloads.py（或校验SHA256SUMS.txt）。使用assemble_handoff.py把各.zip.001/.002按序拼接成普通ZIP；脚本默认保留原分卷，不自动删除文件。需要足够空间来同时保存分卷与合并ZIP。

只写论文：writing-kit包；复算主结果再加reproduction包和dataset_release.zip；历史native基线再加history包。不同包内目录前缀相同，解压到同一父目录即可。

15GB原视频：DATA_SOURCES.json给出官方Dryad源和已核验SHA。它不在Release重复镜像；全部开发缓存已经包含。
''')


def main():
    assert not OUT.exists(), 'Use a new handoff destination'
    for rel in ['artifacts/continuous_research_pipeline_20261016/status.json',
                'artifacts/ray_raw_action_window_followthrough_20261016/status.json']:
        assert read_json(ROOT/rel)['status']=='complete',rel
    WORK.mkdir(parents=True)
    metadata_files()
    methods,rows=table_and_figures(); make_documents(methods,rows)
    # Pin only packages actually used; no workstation credentials or full site-packages.
    for path in [ROOT/'.deps/research_cuda',ROOT/'.deps/research_ort_gpu',ROOT/'.deps/paper_baselines']:
        sys.path.insert(0,str(path))
    packages={}
    for name in ['torch','torchvision','numpy','onnxruntime-gpu','opencv-python','scipy','matplotlib','PyYAML','einops','python-docx']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    write(WORK/'ENVIRONMENT.json',json.dumps({'python':sys.version,'packages':packages,
        'hardware':'RTX4060 Laptop 8GB','floating_point':'IEEE FP32; deterministic settings in each source'},indent=2))
    write(WORK/'requirements-research.txt','\n'.join(f'{n}=={v}' for n,v in packages.items() if n not in ['torch','torchvision'])+'\n')
    source=read_json(ROOT/'artifacts/research_execution_20261012/dryad_files.json')
    files=[]
    for f in source['_embedded']['stash:files']:
        if f['path'] in ['dataset_release.zip','blurred_videos.zip']:
            files.append({'filename':f['path'],'bytes':f['size'],'sha256':f['digest'],
                'official_url':'https://datadryad.org'+f['_links']['stash:download']['href'],
                'included_in_release':f['path']=='dataset_release.zip'})
    write(WORK/'DATA_SOURCES.json',json.dumps({'dataset_doi':'10.5061/dryad.9ghx3ffpp','license':'CC0-1.0',
        'metadata_source':'artifacts/research_execution_20261012/zenodo_mri.json',
        'files':files,'raw_video_mirror':'Official source; experimental caches are included, not a second 15GB video mirror.'},indent=2))
    write(WORK/'EXCLUSIONS.json',json.dumps({'excluded':EXCLUSIONS,
        'policy':'Superseded/pilot binaries not needed for main claims omitted; all primary and final native-baseline binaries preserved. Raw videos via official source. No user uploads, app database, keys, cloud credentials, full local MEMORY, .git objects or pip/CUDA binaries.'},indent=2))
    write(OUT/'build_stage.json',json.dumps({'status':'staged_for_document_review','files':len(MANIFEST),
        'source_bytes':sum(r['bytes'] for r in MANIFEST),'workspace':str(WORK),
        'groups':{g:sum(r['bytes'] for r in MANIFEST if r['bundle']==g) for g in ['writing','reproduction','history']},
        'manifest_records':MANIFEST},indent=2))
    print(json.dumps({'status':'staged','files':len(MANIFEST),'groups':{g:sum(r['bytes'] for r in MANIFEST if r['bundle']==g) for g in ['writing','reproduction','history']}}),flush=True)


if __name__=='__main__':main()
