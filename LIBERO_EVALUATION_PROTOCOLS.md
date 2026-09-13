**LIBERO 评测口径核查（2026-09-13）**

2000 次有主流论文依据，但不是 LIBERO 强制规定。这里的四套指 Spatial、Object、Goal、Long（代码中 Long 也叫 libero_10），各 10 个任务。评测次数、随机种子、初始状态、步数预算和输入模态需要分别对齐。

| 论文 | 每任务评测次数 | 重复与报告方式 |
| --- | ---: | --- |
| [LIBERO 原论文，附录 D](https://arxiv.org/html/2306.03310) | 20 | 每 5 个训练 epoch 评估一次，单次上限 600 步，选择最佳 checkpoint；整个持续学习实验用 100、200、300 三个种子重复。还评估知识迁移与遗忘，不能把其完整实验简单算成一次四套成功率测试。 |
| [OpenVLA，附录 E.1](https://arxiv.org/html/2406.09246v3) | 50 | 每套 500 次，三个随机种子，每套论文统计量对应 1500 次；四套单个种子合计 2000 次，三个种子合计 6000 次。 |
| [OpenVLA-OFT 官方评测说明](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) | 50 | 默认每套 500 次；官方说明明确论文结果对三个随机种子取平均。四套单个种子同样是 2000 次。 |
| [SmolVLA，§4.1](https://arxiv.org/html/2506.01844v1) | 10 | 四套共 40 个任务，每任务 10 次，合计 400 次，按任务是否完整完成计算二元成功率；这一段没有声明三个种子的重复设置。 |

论文中的 50 条训练示范与每任务 50 次评测不是同一概念。50 个官方初态也不等于 50 个随机种子。原论文的训练重复、评测脚本的全局种子和模拟器种子不能混写成一个参数。

[OpenVLA 官方评测脚本](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)采用官方初态列表，按 episode 索引逐个设置初态，先执行 10 步稳定动作，然后给 Spatial / Object / Goal / Long 分别 220 / 280 / 300 / 520 步动作预算。环境返回成功后结束该回合。模拟评测的核心统计是成功次数除以计划评测次数；长任务需要达到完整任务目标。建议同时报告每任务、每套和四套平均成功率，并列出实际样本数及重复种子的误差，而不是只给总分。

当前 Route B 使用四套 × 10 任务 × 50 初态、一个 seed=7，步数预算和 10 步稳定期与上述 OpenVLA 脚本相同。采样规模相近不意味着完整协议相同：

- **模拟器种子不同。** 当前适配器调用 env.seed(7)；[OpenVLA 环境辅助代码](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/libero_utils.py)显式调用 env.seed(0)，它与评测脚本默认的全局 seed=7 是两个设置。官方代码还提示固定初态下种子仍可能影响物体位置。
- **终止方式不同。** 当前评测器记录整个回合内曾经出现的官方成功信号，但不会据此提前停下控制器，而是等控制器主动停止或到步数上限。OpenVLA 的外部评测循环在成功时结束回合。因此当前录像和耗时包含的动作范围可能更长，尚未测量这部分时间占比。对应本地代码：libero_system/integration/evaluator.py 的 _ExternalStickyScore 和 _run_b。
- **输入信息不同。** 当前 Route B 使用双路 RGB-D 等输入；OpenVLA 原论文比较限制为第三人称 RGB，OFT 另有双相机及本体状态设置。对照论文分数时应明确这些条件。[OpenVLA 附录 E.1](https://arxiv.org/html/2406.09246v3)、[OFT 实验设置](https://arxiv.org/html/2502.19645v2)

串行不是评测要求。[LIBERO 官方配置](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/configs/eval/default.yaml)默认 n_eval=20、use_mp=true、num_procs=20；[其评测实现](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/metric.py)使用多个子进程环境。当前 Route B 的 campaign 则依次运行各套任务。这个证据支持尝试并行，但不能据此承诺当前模型开 20 个进程或获得 20 倍加速。

针对当前速度，建议下一轮采用以下安排：

1. **开发验证先覆盖全部 40 个任务，每任务 10 次，共 400 次。** 这是借鉴 SmolVLA 的采样规模，不宣称复现其全部协议。提前固定初态 ID，例如各任务官方列表 0–9；保留失败样本和逐回合记录。每任务 10 次时，一次成败改变该任务成功率 10 个百分点；50 次时为 2 个百分点，不能把两者视作相同统计精度。
2. **需要最终对照时再跑每任务 50 次。** 对照 OpenVLA 系列论文还需说明三个随机种子如何重复，以及种子实际控制哪些随机性；固定策略和固定场景的完全相同重跑不能冒充独立样本。
3. **先测 2 个、4 个进程的吞吐与一致性，再确定并发度。** 保持策略、步数预算、初态和逐回合随机状态一致。并行模拟是计算调度变化；更改动作分块、终止条件或步数预算则需要另立实验配置。

单纯把 2000 次减至 400 次，评测工作量约为原来的五分之一。按此前单个 Spatial 任务约 59 秒/回合的实测速度，400 次粗算约 6.6 小时；这不是完成时间承诺，因为其他任务尤其 Long 尚未取样，真实均值可能更高。并行能节省多少时间需要本机实测。

本次仅核查论文与代码并记录建议，运行中的 full_2000_20260913 沿用原配置。
