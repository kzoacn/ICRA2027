# 原始 400 次并行评测记录

本次评测已完成，结果为 347/400（86.75%）。下文保留原调度过程、时间估计和操作说明。后续改进及全新复测见 [Route B 改进记录](experiments/route_b_90/README.md)。

原评测：`smolvla_scale_400_parallel_20260913`

采用 [SmolVLA 论文 §4.1](https://arxiv.org/html/2506.01844v1) 的采样规模：Spatial、Object、Goal、Long 四套，每套 10 个任务，每任务官方初态 0–9，共 400 次。沿用已部署的 Route B 控制器、seed=7、256×256 双路 RGB-D、220/280/300/520 步上限与原来的外部成功判定方式。这是对齐采样规模；控制器、输入模态等实验条件仍属于 Route B。

后台任务于 2026-09-13 15:09:18 UTC 启动，先用 4 个独立进程运行，再扩到 8 个进程。每个进程负责一个任务的 10 个初态，完成后自动领取下一个任务。进度每 10 秒更新，任务已脱离当前终端运行。

2026-09-13 15:14 UTC 的实测：已完成 34/400 次（新跑 24 次、复用 10 次）。8 进程稳定阶段约完成 427 次/小时，直接外推剩余约 51 分钟；考虑尚未覆盖的复杂任务、失败回合和最后任务分配不均，建议预留 **1–2 小时**。已测资源峰值约为 16.5 GiB 显存、34.6 GiB 容器内存，CPU 平均约 18.2 核。时间估计依据保存在 `runtime/jobs/smolvla_scale_400_parallel_20260913/time-estimate.json`，实际进度以状态文件为准。

原 2000 次任务已停止，其 18 次完整结果保留。新任务复用其中 Spatial 任务 0 的初态 0–9，共 10 次，另外新跑 390 次。复用条件按初态清单预先确定，核对控制器指纹及影响评测的配置一致后保存了原记录快照与来源说明；没有按成功与否筛选。复用记录的录像仍位于原任务目录，请一并保留。

```bash
cd /root/route-b-upload/route-b-v170-cloud

# 查看进度、活跃进程、逐套成功率。
cat runtime/jobs/smolvla_scale_400_parallel_20260913/status.json

# 查看调度日志。
tail -f runtime/jobs/smolvla_scale_400_parallel_20260913/runner.log
```

主要文件：

- 总进度和逐任务汇总：`outputs/smolvla_scale_400_parallel_20260913/summary.json`
- 400 次固定任务清单和有效配置：`outputs/smolvla_scale_400_parallel_20260913/manifest.json`
- 新评测的记录及录像：`outputs/smolvla_scale_400_parallel_20260913/shards/<suite>_task<id>/`
- 复用记录及来源：`outputs/smolvla_scale_400_parallel_20260913/reused/spatial_task00/`
- 每个进程的日志：`runtime/jobs/smolvla_scale_400_parallel_20260913/logs/`
- 调度与吞吐历史：`runtime/jobs/smolvla_scale_400_parallel_20260913/progress.jsonl`

完成后自动生成合并的 `outputs/smolvla_scale_400_parallel_20260913/episodes.jsonl` 以及 `runtime/jobs/smolvla_scale_400_parallel_20260913/final-validation.json`，检查 400 个初态完整覆盖、无重复、配置一致、录像存在、汇总一致和部署源码校验。状态应为 `completed`；如出现 `completed_with_exceptions`、`failed` 或 `runner_error`，须结合日志和校验报告判断。失败回合保留在统计中，不自动重试以挑选更好的结果。

若任务被外部中断，确认旧管理进程和评测子进程均已停止，再续跑：

```bash
cd /root/route-b-upload/route-b-v170-cloud
nohup .venv/bin/python runtime/jobs/smolvla_scale_400_parallel_20260913/runner.py --resume \
  > runtime/jobs/smolvla_scale_400_parallel_20260913/runner-resume.log 2>&1 < /dev/null &
```

续跑保留已完成回合，继续未完成的初态。管理进程有文件锁，避免重复启动同一个调度器。状态文件中的简单剩余时间采用已完成新回合的平均吞吐外推；初期加载、并发度变更和任务难度变化都会影响它，不能作为保证。
