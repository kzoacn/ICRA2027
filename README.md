# ANCHOR · ICRA 2027

**ANCHOR: Asset-Informed Geometric Skills for Language-Conditioned Manipulation**。
本项目基于原 Route B v170 实现，源码位于 [`route-b-v170-cloud/`](route-b-v170-cloud/)。包含控制器、感知与 LIBERO 评测代码、依赖版本、资源校验清单和部署脚本。

当前冻结版本在全新 400 回合中实测 **360/400（90.0%）**，基线为 347/400。
Spatial / Object / Goal / Long 分别为 **93 / 98 / 91 / 78** 次成功（各 100 回合）。
保持 40 个任务、官方初态 0–9、seed 7、原步数预算及外部 ever-success 判定；
13 个原失败回合变为成功，没有原成功回合退步。
完整验收与逐任务对照见 [accepted_result.json](experiments/route_b_90/accepted_result.json)。

## 获取与运行

```bash
git clone git@github.com:kzoacn/ICRA2027.git
cd ICRA2027/route-b-v170-cloud
sha256sum -c SHA256SUMS.current
```

系统依赖、Python 3.12 环境及 Docker 用法见 [中文部署说明](route-b-v170-cloud/README.zh-CN.md)。系统依赖安装完成后：

```bash
TORCH_BACKEND=cu128 ./setup.sh
./run.sh doctor --render --device cuda
./run.sh smoke --device cuda

# 四套共 40 个任务，每任务 10 个官方初态，共 400 次，顺序运行。
./run.sh campaign --episodes-per-task 10 --device cuda --run-name cloud_400_01
```

`setup.sh` 会安装依赖并下载 `resources.lock.json` 固定的模型和仿真资产。虚拟环境、下载资源、日志、评测输出与录像保留在运行机器上，不纳入 Git。

## 部署与评测记录

- [服务器部署与验证](DEPLOYMENT.md)
- [400 次并行评测说明](RUN_400_PARALLEL.md)
- [原 2000 次评测记录](FULL_TEST.md)
- [LIBERO 评测口径核查](LIBERO_EVALUATION_PROTOCOLS.md)
- [90% 改进目标、开发记录与完整复测](experiments/route_b_90/README.md)

以上记录中的绝对路径和进程信息对应原部署服务器。`route-b-v170-cloud/runtime/jobs/` 中保留了该服务器的调度脚本与固定配置；清单包含本机路径和旧评测结果的复用关系，不能直接在新机器上续跑。新评测可使用上面的标准入口。

本仓库保留原部署来源记录与改进后的当前源码。`SHA256SUMS` 和 `SHA256SUMS.deployed` 分别保存原上传包、原服务器部署的历史校验值；核对当前源码请使用 `SHA256SUMS.current`。来源与差异记录位于 [`provenance/`](route-b-v170-cloud/provenance/)，改进记录位于 [`experiments/route_b_90/`](experiments/route_b_90/)。

## ICRA 2027 论文

英文匿名初稿、LaTeX 源码、PDF、图表和可复核的实验统计位于 [paper/](paper/)。

- [阅读论文 PDF](paper/main.pdf)
- [LaTeX 源码](paper/main.tex)
- [构建方法与证据范围](paper/README.md)
