# Knap等2024：完整方法核读与可比性边界

核读日期：2026-10-16。原文4页全部提取并阅读，方法第2页实际渲染查看。开放全文保存为 `artifacts/Knap_ICEIC2024_open_access.pdf`，逐页文字 `artifacts/Knap_ICEIC2024_text.txt`。文献服务题录DOI：10.1109/ICEIC61013.2024.10457094；原PDF页首标注arXiv:2403.09437v1。已用Crossref独立核对标题、5作者、日期2024-01-28、会议及DOI一致，证据 `artifacts/Knap_ICEIC2024_crossref_verification.json`。PDF不是未取得的2023前作。

题名：Improving Real-Time Omnidirectional 3D Multi-Person Human Pose Estimation with People Matching and Unsupervised 2D-3D Lifting。作者Pawel Knap、Peter Hardy、Alberto Tamajo、Hwasup Lim、Hansung Kim，ICEIC 2024。

## 原文确实实现的步骤

1. Ricoh Theta V全向相机、三个TI AWR1843BOOST雷达、RTX3060；OpenPose产生15关键点。
2. 相机标定；雷达通过已知50cm间隔位置的数据及Levenberg–Marquardt拟合二维仿射变换，修正x/z定位。
3. 雷达people-counting程序定位多人，将雷达坐标变换到图像空间；以图像平均x匹配2D姿态和雷达个体。
4. LInKs在Human3.6M的OpenPose关键点上重训，无监督预测相对根的关键点深度偏移。第2页II-C的Eq.(1)：

   x_i = (u_i z_i, v_i z_i, z_i)，z_i = max(1, d_i + c)。

   原文用x_i同时表示2D/3D，以上改用u/v避免歧义。c是lifting假定的相机距离。之后减c并加雷达x/z坐标进入全局坐标；再减最低脚踝y以接触地面。

5. 原文文字将雷达用于全局放置，没有明确写出以逐帧雷达根深度替换Eq.(1)中的c。因此不能声称它已经逐项实现了我们当前的雷达根深度射线候选；同样不能声称我们首次结合雷达、相机标定和2D–3D提升。

## 原文数据：不能直接减我们的59.27mm

第3页Table II在Human3.6M评估lifting：原LInKs无遮挡PA-MPJPE33.8 / N-MPJPE61.6mm；其OpenPose重训版本37.2 / 61.7mm。遮挡结果按左臂、左腿、右臂、右腿、组合、双腿、躯干另列，非本研究图像矩形遮挡。

第4页Table III在实体雷达位置上报告x/z绝对MAE，单位cm：改进后三个雷达分别16.45/11.45、24.86/10.77、15.94/13.46。第3页Table I是匹配百分比误差；两者都不是根相对17关节MPJPE。

本研究的59.273mm来自mRI、RTMO17点、主体留出、根相对MPJPE；PCK、PA-MPJPE另报。Human3.6M上的PA/N指标经过对齐，与其直接算差没有意义。也不能把全球定位cm表转换成mm后当作论文姿态基线。

## 对当前算法主张的约束

- 校准相机射线逆投影是既有工具；雷达解除单目定位/尺度歧义的方向已有前作。
- 我们实际增加的是：冻结的RGB+雷达强先验、观测雷达统计拟合根深度、对可用2D/双髋执行几何候选、主体分离OOF训练的小残差头，以及三种真实像素退化/缺模态控制。这些组合与验证尚不自动构成方法新颖性。
- 当前自由雷达几何优于受限门控，不将后者列为贡献；常深度控制收益已接近雷达深度，雷达额外效应必须单独披露。
- 原文的雷达全局平移、脚踝地面平移会在按预测髋重中心后抵消。根相对指标不衡量这些全球定位贡献，因此不能以该指标断言原文的雷达放置步骤无效。
- 当前free_constant_residual是共享强先验上的常深度几何算子控制，可用于机制对比；它没有重训原LInKs，也不是Knap完整官方复现。

## 已完成的公平控制与接下来的判据

同观测raw网络1392参数：clean63.6406、S88.7453mm。相同hidden32/embedding16且更多参数1843的raw网络：clean63.1860、S88.4028mm。几何1395参数：clean59.2732、S86.2554mm。两组raw均有独立外部NumPy重建和受试者配对结果。

上述对照支持显式几何表示在本预算下有效，不能证明所有神经网络不及几何，不能证明首次提出。大数据12288帧正式比较正在进行，须完整audit后判断收益是否保留；最终四人未预测。

## 检索限制

语义检索的原始摘要/题录保存在 `artifacts/ray_geometry_undermind_search_20261016.json`。跨库搜索 `artifacts/ray_geometry_related_search_20261016.json` 中OpenAlex HTTP429，Crossref宽关键词返回多项无关文献，不能据此称系统检索完备。Undermind本次达到免费搜索限额，服务提示2026-10-16 11:30UTC（北京时间19:30）恢复；该限制不影响本地训练或已取得原文的核读。没有调用会创建内部代理的read_pdfs/deep_search，也没有联系作者。
