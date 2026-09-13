# ICRA 2027 匿名论文初稿

标题：**Route B: An Asset-Informed Geometric Baseline for Language-Conditioned Manipulation**

- [论文 PDF](main.pdf)
- [LaTeX 源文件](main.tex)
- [参考文献](references.bib)
- [实验记录投影](data/episodes.jsonl)
- [完整原始记录及几何追踪](data/raw_episodes.jsonl.gz)
- [统计结果](data/statistics.json)
- [公开对比数据及原论文来源](data/published_comparisons.json)
- [数据来源与完整性](data/provenance.json)
- [逐项证据说明](EVIDENCE.md)
- [本地成稿检查](data/artifact_validation.json)

这是依据当前代码和实测结果撰写的英文匿名初稿，尚未提交 PaperPlaza。
当前结果为同一冻结版本全新复测的 **360/400（90.0%）**。
论文按一个使用资产先验、显式几何和人工技能的系统基线定位。
对比表引用原论文公开结果，并标明输入、训练和评测条件；本项目一行来自实际评测记录。
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
公开对比表由 Python 标准库脚本生成；make 会在数据或脚本变化时自动更新它。

## 从记录重新生成结果

冻结的 data/episodes.jsonl 足以重建数值、表格和逐任务图，所需 Python 包是
NumPy 与 Matplotlib：

    python3 -m pip install -r requirements-figures.txt
    python3 scripts/analyze_results.py
    make
    make check

如果要重现原服务器的历史基线结果：

    python3 scripts/capture_results.py --source-root /path/to/route-b-v170-cloud
    python3 scripts/analyze_results.py
    make
    make check

capture_results.py 默认要求评测完成且校验通过，防止把未完成的结果当作最终分数。
仅用于撰写期间的 --allow-partial 选项会将数据标记为未完成，并在编译稿中明确提示；
这种稿件不能通过 make check。

改进实验的完整新复测使用批次目录导入：

    python3 scripts/capture_results.py --batch-dir ../runtime/route_b_90/full400_candidate_01
    python3 scripts/analyze_results.py
    python3 scripts/make_rollout_figure.py --batch-dir ../runtime/route_b_90/full400_candidate_01
    make check

这种导入还会生成 `data/raw_episodes.jsonl.gz`，保存全部原始逐回合记录与几何追踪。
校验脚本核对压缩文件、解压内容和每条原始记录的 SHA256，并逐条比对论文投影中的
成功标记、初态、步数和内部状态。开发用小批次不能作为最终论文数据导入。

录像图使用 ImageIO、imageio-ffmpeg、NumPy 与 Matplotlib，可在已部署的虚拟环境中运行：

    python3 scripts/make_rollout_figure.py --source-root /path/to/route-b-v170-cloud

图像来自真实仿真录像，只提取原始双视图中固定相机的一半。
选用的回合、帧号与原录像 SHA256 记录在 data/figure_provenance.json。
重建录像图需要原服务器录像；正常编译使用已经提交的 figures/rollouts.pdf。

## 公开结果对比

对比表列出四个 LIBERO 套件及平均成功率，外部结果来自以下原论文：

- Diffusion Policy、Octo、OpenVLA：[OpenVLA v3，表 12](https://arxiv.org/html/2406.09246v3)。
  这三项均由该论文作者评测；表中展示其报告均值，原标准误保存在数据文件中。
- SmolVLA 0.45B：[SmolVLA v1，表 2](https://arxiv.org/html/2506.01844v1)。
  采用 VLM 初始化、跨任务训练的仿真结果。
- OpenVLA-OFT：[OpenVLA-OFT v2，表 I](https://arxiv.org/html/2502.19645v2)。
  采用增加腕部相机和机器人状态、过滤训练示范后的完整模型结果。

data/published_comparisons.json 保存原论文版本、表号、PDF 页码、PDF SHA256、
数值和设置说明。来源 PDF 已用于复核，未在本仓库转载。
文献中的平均值按原值保留，不用经过舍入的套件分数重新计算。
这些行是文献报告值，本项目未重跑这些策略。

本项目对比行由同一份 400 条记录生成，保证它与正文统计一致：

    python3 scripts/build_comparison_table.py
    python3 scripts/build_comparison_table.py --check

## 数据与结论边界

400 次清单覆盖四套各十个任务，每任务官方初态 0–9，seed=7。
本次 400 条记录全部新跑，没有复用开发批次或历史回合；全部使用同一冻结控制器。
失败回合完整保留。Spatial / Object / Goal / Long 分别为 93 / 98 / 91 / 78 次成功。

论文使用外部 ever-success 判定：回合中曾满足官方谓词即为成功。
控制器继续执行至自己停止或达到预算。因此论文同时报告内部完成状态与外部成功的交叉表。
这一口径不能直接替换成终态成功，也不等价于 OpenVLA 的整套评测协议。

数据投影保存原始数值和每条原始 JSONL 行的 SHA256。
`data/raw_episodes.jsonl.gz` 包含全部 400 条原始记录及几何追踪；完整日志和录像仍保留在
评测服务器。源文件与压缩档的校验值、字段来源见 provenance.json。
已有 2000 次历史摘要缺少当前包中的完整逐回合原记录，没有作为本论文的实验结果。

## 作者审阅

投稿前需由实际作者核对全文和引用、明确原创贡献与开发数据使用范围，
并审定题名、摘要与结论。公开对比表保留各自实验设置，不能据此宣称在同一条件下
优于某种方法。当前结论适用于已评测的资产、语言形式和校准条件。
记录没有单独保存首次成功时刻和终态成功，正文说明了成功判定的具体含义。

## 投稿格式依据

查阅日期：2026-09-14。

[ICRA 2027 官方征稿说明](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
要求全文含参考文献最多八页、双栏和双匿名；生成式 AI 生成的内容需在致谢中披露。
稿件已经加入说明，包含实际使用的工具名称与用途。
按照同页 FAQ，PDF 中保留可读网址文本，不加入可点击链接注释。

本地检查覆盖页数、Letter 纸型、匿名元数据、字体嵌入、未定义引用、
排版溢出、400 条记录覆盖和对比表与来源数据的一致性；
这些检查不等同于 PaperPlaza 官方合规检测或作者的科学审阅。

官方模板来源及原文件 SHA256：

- [ieeeconf.zip](https://ras.papercept.net/conferences/support/files/ieeeconf.zip) 中的 ieeeconf.cls：
  4befef671c2a996889d325f5170d3387bf42aac9a37dcaa93724ad49816e4ec2
- [IEEEtranBST.zip](https://ras.papercept.net/conferences/support/files/IEEEtranBST.zip) 中的 IEEEtran.bst：
  b11af8e5096681f1eccdce6c72c047dc056ddeef52dff340213104019bcf3409

模板文件保留原始版权与许可说明，未做修改。
