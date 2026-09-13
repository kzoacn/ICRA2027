Route B v170 已部署到当前服务器的数据盘，可通过以下路径使用：

```bash
cd /root/route-b-upload/route-b-v170-cloud
./run.sh doctor --render --device cuda
./run.sh smoke --device cuda
```

实际目录为 `/root/autodl-tmp/route-b-v170-cloud`；上传目录下的 `route-b-v170-cloud` 是指向它的符号链接。Python 环境、资源及评测输出均在数据盘，部署后约占 9.3 GiB。通过 `run.sh` 启动即可，无需手动激活虚拟环境。

本次验证结果（2026-09-13 UTC）：

- Ubuntu 22.04，Python 3.12.3，PyTorch 2.11.0+cu128，RTX 5090；CUDA 实际运算通过。
- 27 个项目固定版本依赖匹配，585 个仿真资产和 8 个模型文件全部通过 SHA256 校验。
- 四套共 40 条任务指令编译通过；固定相机和腕部相机均为 256×256 RGB。
- `deployment_smoke_20260913`：LIBERO Object task 0 / init 0，成功 1/1，127 步，39.571 秒。
- 双相机视频：512×256，20 fps，共 64 帧，全部可解码。
- 部署阶段仅执行了单任务验证；后续全量任务见 [FULL_TEST.md](FULL_TEST.md)。

结果和日志（以下路径相对于实际部署目录）：

- `outputs/deployment_smoke_20260913/summary.json`：评测汇总。
- `outputs/deployment_smoke_20260913/episodes.jsonl`：单次评测记录。
- `outputs/deployment_smoke_20260913/videos/route-b_libero_object_task-00_ep-0000.mp4`：双相机录像。
- `logs/doctor-render.json`、`logs/cuda-check.json`、`logs/video-check.json`：环境与录像检查。
- `logs/environment-freeze.txt`：已安装 Python 包的完整版本记录。
- `provenance/server-deployment.json`：本机部署记录。

本机兼容调整：

- 安装系统 OSMesa 库及相关运行依赖。
- 在 `.venv/lib/libstdc++.so.6` 建立指向 `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` 的链接，并让 `run.sh` 优先预加载该环境的 C++ 库，解决基础 Conda Python 的旧库与 OSMesa/LLVM 的 `GLIBCXX_3.4.30` 冲突。
- 在项目虚拟环境中补充 `httpx[socks]==0.28.1` 所需的 `socksio==1.0.0`，支持服务器已有的代理设置。
- `.venv/pip.conf` 使用 `https://pypi.org/simple`，PyTorch 按安装脚本使用官方 cu128 源；全局 Python 和 pip 配置未修改。

控制器源码指纹仍为 `67c93be1df7da7572ecbb8faff53d01a47aea42ec36affb07ac54ba36e59edba`，与上传包一致。上传包中的 104 个文件仅 `run.sh` 有上述环境兼容调整，原文件保存在 `provenance/run.sh.uploaded`，差异见 `provenance/SERVER_RUNTIME.patch`。原始 `SHA256SUMS` 保留，可用 `sha256sum -c SHA256SUMS.deployed` 校验部署后的文件。

后续运行示例：

```bash
# Spatial：10 个任务，每个任务 5 个初态，共 50 次。
./run.sh run --suite spatial --episodes-per-task 5 --device cuda

# 四套完整评测：共 2000 次，运行名须为新名称。
./run.sh campaign --episodes-per-task 50 --device cuda --run-name cloud_2000_01

# 中断后，用相同参数和原运行名续跑。
./run.sh campaign --episodes-per-task 50 --device cuda --run-name cloud_2000_01 --resume
```

资源准备完毕后，评测入口使用本地模型和资产。新实验会写入 `outputs/<运行名>/`；省略运行名会自动生成唯一名称。
