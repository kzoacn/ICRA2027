原全量评测任务：`full_2000_20260913`

已按用户要求于 2026-09-13 停止，保留 18 次完整结果。当前改跑 [400 次并行评测](./RUN_400_PARALLEL.md)，其中复用本任务符合官方初态 0–9 的 10 次结果。以下为原任务的历史运行说明。

已于 2026-09-13 14:45:45 UTC 在后台启动。实际进度以状态文件和进程为准。

执行 Spatial、Object、Goal、Long 四套任务，每套 10 个任务、每任务使用官方初态 0–49，共 2000 次。使用 RTX 5090、seed=7、256×256 双相机，保存录像。四套步数上限分别为 220、280、300、520，顺序运行，控制器源码与部署验证版本一致。

```bash
cd /root/route-b-upload/route-b-v170-cloud

# 查看运行状态，每 15 秒更新。
cat runtime/jobs/full_2000_20260913/status.json

# 持续查看每次评测的完成记录。
tail -f logs/full_2000_20260913.log

# 查看评测进程是否仍在运行。
ps -p "$(cat runtime/jobs/full_2000_20260913/campaign.pid)" -o pid,etime,pcpu,pmem,args
```

运行目录：

- `outputs/full_2000_20260913_spatial/`
- `outputs/full_2000_20260913_object/`
- `outputs/full_2000_20260913_goal/`
- `outputs/full_2000_20260913_long/`

每个目录持续保存 `episodes.jsonl`、`summary.json` 和 `videos/`。完成后生成 `outputs/full_2000_20260913_campaign.json`，后台管理进程还会生成 `runtime/jobs/full_2000_20260913/final-validation.json`，核对覆盖的初态、重复记录、配置、汇总计数、录像文件和源码校验值，并统计异常次数。

状态中的 `success_rate_so_far` 是已完成样本的阶段成功率。最终结果需等待状态变为 `completed` 并检查全量汇总；`failed` 或 `completed_with_exceptions` 需要查看日志和校验结果。

任务已脱离当前终端运行。若评测中断，可在确认原评测进程已停止后，用相同配置续跑：

```bash
cd /root/route-b-upload/route-b-v170-cloud
nohup .venv/bin/python runtime/jobs/full_2000_20260913/runner.py --resume \
  > runtime/jobs/full_2000_20260913/runner-resume.log 2>&1 < /dev/null &
```

续跑会保留已完成记录，并跳过已经评测的初态。后台管理进程使用文件锁防止重复启动；进程退出或机器重启后，原状态文件可能是退出前的记录，需要结合进程检查判断。
