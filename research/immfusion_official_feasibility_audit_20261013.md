# ImmFusion官方代码：本机复现与mRI适配可行性核查

日期：2026-10-13。只读取公开源码、作者资料、文件元数据和本机环境；没有运行官方代码、安装依赖、读取预训练权重、申请数据或占用GPU。本报告支持后续资源判断，不是已复现结果。

## 1. 结论与明确边界

**官方源码可以核查，但当前不能原样完成论文原协议复现。** 已将[作者仓库](https://github.com/Chen3110/ImmFusion)固定到commit 4a958d837a8507f4968b28a6b937fcb48b1f9b11（2025-02-25），完整源码ZIP2,778,958字节、202文件，解压9,117,966字节，保存在[独立源码审计目录](../artifacts/immfusion_official_audit_20261013/)。源码体积不是阻碍；完整mmBody数据准入、body-model资产、旧预训练链接和依赖/编译环境尚未就绪。

已有mRI视频＋196×5点云可以支持**基于官方模块的姿态任务适配**，但必须改变数据加载、GT依赖的输入处理、关节语义/监督、预训练和训练预算。完成后只能写明“ImmFusion适配版”或更窄的“基于ImmFusion模块的joint-only适配”；不能把原论文数字移到mRI主表，不能声称原协议复现。

## 2. 原始数据、输入和划分

官方mmBodyDataset读取train/sequence目录或test/scene/sequence目录，需calib.txt、逐帧RGB PNG、radar NPY和mesh NPZ。mesh字段含22个SMPL-X身体关节和10,475 vertices。雷达原数据含XYZ及速度/幅值/能量等3个额外字段，6维量纲与mRI发布5字段不同。[作者数据说明](https://github.com/Chen3110/mmBody)已核读并归档。

源码默认输入image0＋radar0，图像裁至224×224并按ImageNet标准化，雷达固定重采样1024点。默认PointNet只使用XYZ；只有开启use_point_feat才输入额外3维。因此mRI可以明确选择XYZ版本，不能为了凑6维伪造未知量，或套用mmBody非XYZ缩放常数。196点重复/采样成1024必须记录，不会创造新的实际雷达信息。

关键协议差别在[官方dataset源码](https://github.com/Chen3110/ImmFusion/blob/4a958d837a8507f4968b28a6b937fcb48b1f9b11/src/datasets/fusion_dataset.py)：process_image根据GT joints投影取人体框；process_pcl按GT joints的三维包围框过滤雷达，再将XYZ减GT pelvis，训练和测试都经过此加载路径。mRI当前禁止GT框/GT root/GT选择人物进入输入，故直接复制加载器会违反本项目协议。后续适配应使用同一冻结检测器框和观测点集/训练内固定坐标规则，GT只用于目标根对齐和误差评价。

当前repo训练默认sequence_0..19；测试每scene的sequence_0..1。训练内部默认对帧random_split 90%/10%，另有eval_test_dataset选项。原论文描述选择10个lab场景序列训练、每场景2个测试序列；未给可直接核对的subject-disjoint清单。当前repo与ICRA2023文本的训练数量不能未经核验视为完全一致。本项目按主体划分，而不是复制作者帧级验证；改变划分必须披露。

## 3. 资产取得的实际状态

- mmBody README要求向作者邮件提供身份/机构/国家/用途，学生及博士后还需导师信息。数据只允许非商业学术研究/教学，并限制修改和再分发。本轮没有联系作者，未取得完整mmBody；原协议数据并非仓库源码ZIP中的直接公开下载。
- 官方README要求另从SMPL/SMPL-X站点获得body-model文件。源码快照和当前workspace没有SMPLX_NEUTRAL.pkl及basicModel_neutral文件；未登录或接受相关资产条款。mesh模板、采样器、顶点监督仍依赖这些资产。
- download_models.sh实际列出的是ImageNet预训练HRNet W40/W64及YAML；没有列出完整ImmFusion fusion checkpoint。不能称运行该脚本即得到论文最终模型。
- 2026-10-13对原Azure端点的公开HEAD均返回409；普通YAML GET明确返回PublicAccessNotPermitted。此结果已保存[权重链接receipt](../artifacts/immfusion_official_audit_20261013/pretrained_availability_receipt.json)和GET错误记录。没有绕过访问控制；尚未核验其他官方HRNet镜像的权重身份/SHA。
- GitHub API读取mmBody example树遭rate limit 403，因此没有据此判数据不存在。README可读取且已保存，但公开example不能被当成完整train/test已取得。

## 4. 预训练、默认模型与训练预算

源码默认加载ImageNet HRNet-W64。三组Graphormer输入维2051/512/128，hidden维1024/256/64，每组默认4层/4头，最后输出3维；SMPL-X模板含22joint＋655粗vertex共677 queries。局部图像token通常49个，当前repo默认雷达cluster49（论文方法文字为32），GIM为2048维、3层、8头。不能把这些全局/局部raw-image模块缩成现有51维输入而仍声称同一官方网络。

当前默认Adam、lr=1e-4、weight_decay=0、batch10、50epochs，学习率按半程因子衰减；seed88。实际损失是1000×3D joint MSE＋100×多尺度vertex L1（各vertex级权重约0.33），可选2D reprojection。代码按最佳验证PA-MPJPE以及每10轮保存，末轮亦保存。原论文写50epochs、Adam初始lr0.001和从头训练；当前repo预训练/学习率/cluster数与文字有差异，不替作者猜测实验配置。

本项目小网络是冻结RTMO骨架特征、约2.5万–11.1万参数、80epochs、batch64、最后一轮；官方raw-image/mesh网络的模型容量、预训练、训练目标和算力均不同。后续不能只控制epoch数就声称训练预算完全一致，应同时记录更新次数、训练样本、训练总时间、峰值显存和预训练数据来源。

## 5. 依赖与本机可运行性

requirements不是可直接使用的锁定环境：Pillow同时精确指定5.4.1/9.2.0/10.0.0，PyYAML、setuptools、torchvision也重复互斥版本；还混合torch1.8.0、torchvision0.9.0、旧NumPy/scipy、OpenDR/psbody、Azure与通知/绘图库。相互冲突的精确版本不能全部满足。官方引用的MeshGraphormer安装说明基于Ubuntu16.04/Python3.7/PyTorch1.4/CUDA10.1，也不等于此repo当前有效lock。

源码PointNet++包含自定义C++/CUDA扩展，import会要求pointnet2._ext，setup.py使用CUDAExtension；有旧式绝对import和硬编码cuda调用。旧body/render模块、BERT接口及可选云日志也需独立环境逐项核验。没有执行repo的import/setup/download命令；不会让科研训练环境承担盲目兼容试验。

只读本机快照：Python3.13.5；当前科研GPU环境记录torch2.11.0+cu128、CUDA12.8、RTX4060 Laptop。普通PATH未找到nvcc或MSVC cl，这不证明它们在所有安装位置都不存在，但不能据现有torch CUDA runtime推导自定义扩展已能编译。workspace当时可用约24.0GB；未取得mmBody完整发布清单，故不给未经核实的总数据大小。没有测官方batch10显存或运行时，不能承诺可在本机直接装下。

仓库根部未发现LICENSE，GitHub license字段为null；transformers有Apache许可证，PointNet文件头指向上游MIT许可证，但整个适配版的发布权限不能仅由仓库公开性推断。此次只下载公开源码作本地审查；后续分发修改版应核对各模块来源和原许可文本，不伪称已经获得整体再分发授权。

## 6. mRI适配需要改哪些内容

| 原始实现部分 | 当前mRI条件 | 若继续必须采取的改变 |
| --- | --- | --- |
| GT裁图、GT point box、GT pelvis输入中心 | 一路完整视频＋冻结检测器；禁止GT推理输入 | 用预测框/观测规则替代，并给所有raw-image基线相同裁图预算；公开坐标变换与失败回退 |
| mmBody22关节和完整mesh目标 | mRI17关节，无独立SMPL-X顶点标注 | 明确语义共同关节映射，不能截取前17；先采用共同关节指标或明确joint-only head，不从GT关节拟合伪mesh后冒充原监督 |
| 677模板queries＋mesh graph＋upsampling | 原body资产未就绪，输出只需姿态 | 保留原模板就需资产及无vertex监督政策；删去vertex模块就是结构适配，应按实际保留模块命名 |
| raw image HRNet local/global features | 当前缓存主要用于RTMO51维输入 | 从已有合规视频按同帧读取原图token；不能将51维骨架当作官方HRNet等价输入 |
| 1024×6雷达、GT过滤/中心 | 196×5公开预处理点集 | 明确XYZ-only选择/重采样/空点处理；五维扩展若实施则记录与原默认差别 |
| 作者序列/帧划分和最佳checkpoint | 本项目主体拆分、末轮固定 | 所有方法统一本项目协议；另做原协议时单独报告，不跨表混用 |
| 原图/mesh大网络、旧预训练依赖 | 现有新96帧GPU小网络实验正在运行 | 等受控实验完成后做隔离pilot；新增训练/迁移成本单列，不争用当前GPU，不覆盖旧权重 |

最经济而诚实的后续路线是先确定joint-only/共同关节适配是否仍保留ImmFusion局部/全局融合的核心行为，解决GT-free输入与PointNet扩展，再在开发主体上做极小前向/反向验证。只有实际通过兼容与预算核验后，才决定完整重训；所有结果应写“适配”，不能用修剪后的弱近似给官方方法背书或制造优势。

如果研究目标是原论文复现，则需要原mmBody授权数据、body资产、可核验官方预训练与作者实验配置，还需协调repo文本差异。当前只读审计并未达到这些条件。

## 7. 可追溯记录

完整fetch_receipt包含commit、ZIP SHA256、大小、源码文件数与GitHub license状态；source_manifest保存ZIP条目CRC，audited_source_sha256保存实际审读的11个关键文件SHA；相关README/安装说明、旧权重公开请求和本机只读库存另有receipt。主目录未改动，当前33套研究训练不受本次审计影响。

本报告沿用[近邻文献核查](fusion_literature_audit_20261013.md)的来源核验流程。报告中的源码、预算和许可状态来自实际公开资料读取；方法性能、新数据下载、编译成功和复现成功均未宣称。
