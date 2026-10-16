# 下载与合并分卷

用GitHub CLI一次下载：

    gh release download research-handoff-20261016 --repo yjkira-01/monitor --dir downloads

先运行下载目录中的verify_downloads.py（或校验SHA256SUMS.txt）。使用assemble_handoff.py把各.zip.001/.002按序拼接成普通ZIP；脚本默认保留原分卷，不自动删除文件。需要足够空间来同时保存分卷与合并ZIP。

只写论文：writing-kit包；复算主结果再加reproduction包和dataset_release.zip；历史native基线再加history包。不同包内目录前缀相同，解压到同一父目录即可。

15GB原视频：DATA_SOURCES.json给出官方Dryad源和已核验SHA。它不在Release重复镜像；全部开发缓存已经包含。
