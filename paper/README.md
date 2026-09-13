# ICRA 2027 匿名论文初稿

标题：**Route B: An Asset-Informed Geometric Baseline for Language-Conditioned Manipulation**

- [论文 PDF](main.pdf)
- [LaTeX 源文件](main.tex)
- [参考文献](references.bib)
- [实验记录投影](data/episodes.jsonl)
- [统计结果](data/statistics.json)
- [数据来源与完整性](data/provenance.json)
- [逐项证据说明](EVIDENCE.md)
- [本地成稿检查](data/artifact_validation.json)

这是依据当前代码和实测结果撰写的英文匿名初稿，尚未提交 PaperPlaza。
论文按一个使用资产先验、显式几何和人工技能的系统基线定位；没有把它描述为
无先验零样本系统，也没有加入未执行的消融、实机实验或与 VLA 的排名比较。
作者及单位暂用 Anonymous Authors；PDF 内不包含本项目的 GitHub 账号或仓库链接。

## 构建

从本目录执行：

    make
    make check

所需工具：PDFLaTeX、BibTeX、latexmk、Poppler 工具；Ubuntu 可安装：

    sudo apt-get install texlive-latex-base texlive-latex-recommended \
      texlive-fonts-recommended texlive-pictures latexmk poppler-utils

官方 ieeeconf.cls 和 IEEEtran.bst 已随稿保存，不需要联网获取模板。
图表、统计表和数值宏也已提交，单纯编译 PDF 不需要仿真环境或 GPU。

## 从记录重新生成结果

冻结的 data/episodes.jsonl 足以重建数值、表格和逐任务图，所需 Python 包是
NumPy 与 Matplotlib：

    python3 -m pip install -r requirements-figures.txt
    python3 scripts/analyze_results.py
    make
    make check

如果要重新读取服务器的完整评测结果：

    python3 scripts/capture_results.py --source-root /path/to/route-b-v170-cloud
    python3 scripts/analyze_results.py
    make
    make check

capture_results.py 默认要求评测完成且校验通过，防止把未完成的结果当作最终分数。
仅用于撰写期间的 --allow-partial 选项会将数据标记为未完成，并在编译稿中明确提示；
这种稿件不能通过 make check。

录像图使用 ImageIO、imageio-ffmpeg、NumPy 与 Matplotlib，可在已部署的虚拟环境中运行：

    python3 scripts/make_rollout_figure.py --source-root /path/to/route-b-v170-cloud

图像来自真实仿真录像，只提取原始双视图中固定相机的一半。
选用的回合、帧号与原录像 SHA256 记录在 data/figure_provenance.json。
重建录像图需要原服务器录像；正常编译使用已经提交的 figures/rollouts.pdf。

## 数据与结论边界

400 次清单覆盖四套各十个任务，每任务官方初态 0–9，seed=7。
其中 Spatial task 00 的十条结果是按预先确定的初态清单复用的，其余 390 条为新跑结果。
时间统计排除了复用回合，失败回合完整保留。

论文使用外部 ever-success 判定：回合中曾满足官方谓词即为成功。
控制器继续执行至自己停止或达到预算。因此论文同时报告内部完成状态与外部成功的交叉表。
这一口径不能直接替换成终态成功，也不等价于 OpenVLA 的整套评测协议。

数据投影保存原始数值和每条原始 JSONL 行的 SHA256，省略完整几何追踪和视频。
原追踪、完整日志与视频仍保存在评测服务器。源文件校验值和字段来源见 provenance.json。
已有 2000 次历史摘要缺少当前包中的完整逐回合原记录，没有作为本论文的实验结果。

## 作者后续工作

投稿前需由实际作者核对全文和引用、明确原创贡献与开发数据使用范围，
并根据投稿策略补充证据。当前主要不足是：

1. 尚无同一输入条件和评测协议下的学习策略对照。
2. 尚无资产先验、第二相机、携带偏移补偿和恢复逻辑的消融。
3. 尚无多种子复测、未见物体/语言、校准噪声或实机验证。
4. 记录没有单独保存首次成功时刻和终态成功，需要后续实验才能量化两者差异。

这些是当前研究证据的限制，不是已经完成的实验。论文正文已如实说明。

## 投稿格式依据

查阅日期：2026-09-14。

[ICRA 2027 官方征稿说明](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
要求全文含参考文献最多八页、双栏和双匿名；生成式 AI 生成的内容需在致谢中披露。
稿件已经加入说明，包含实际使用的工具名称与用途。
按照同页 FAQ，PDF 中保留可读网址文本，不加入可点击链接注释。

本地检查覆盖页数、Letter 纸型、匿名元数据、字体嵌入、未定义引用、
排版溢出和 400 条记录覆盖；这些检查不等同于 PaperPlaza 官方合规检测或作者的科学审阅。

官方模板来源及原文件 SHA256：

- [ieeeconf.zip](https://ras.papercept.net/conferences/support/files/ieeeconf.zip) 中的 ieeeconf.cls：
  4befef671c2a996889d325f5170d3387bf42aac9a37dcaa93724ad49816e4ec2
- [IEEEtranBST.zip](https://ras.papercept.net/conferences/support/files/IEEEtranBST.zip) 中的 IEEEtran.bst：
  b11af8e5096681f1eccdce6c72c047dc056ddeef52dff340213104019bcf3409

模板文件保留原始版权与许可说明，未做修改。
