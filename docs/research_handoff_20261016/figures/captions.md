# 图注与统计口径

large_clean_mm：完整12288帧开发协议的正常光照根相对MPJPE。12训练主体、4开发验证主体，一个划分、三个初始化。误差线是初始化样本标准差，不是置信区间。图内所有方法均使用同一3072验证帧；GRU及MLP具有9帧历史，几何/同观测头是单帧。

large_S_mm：同人口下正常、暗噪、遮挡、视频缺失、雷达缺失五条件的受试者等权误差。图显示GRU综合误差更低；几何残差未优于同观测普通MLP。不得只展示正常条件宣称全面更优。

原始数值：handoff/large_protocol_all_methods.csv；正式证据：artifacts/ray_oof_action_windows_20261016/reporting/results.json、artifacts/ray_raw_action_window_control_20261016/reporting/results.json。
