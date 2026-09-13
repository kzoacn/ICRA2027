# Route B v170 云端精简部署包

用于在 Linux 云服务器上运行 Route B 的 LIBERO 仿真评测。控制器来自已完成四套、2000 次评测的 v170 源码快照。支持固定相机与腕部相机 RGB-D、几何感知、抓放及接触技能。

压缩包只包含运行代码、安装/启动脚本、资源版本与校验值。Grounding DINO 模型约 658 MiB，LIBERO 仿真资产约 403 MiB，在服务器首次部署时下载；不下载演示数据集、SmolVLA 或 SmolVLM。

## 推荐：Docker

准备 Linux x86_64 服务器、Docker；使用 GPU 时，服务器还需已安装 NVIDIA 驱动和 NVIDIA Container Toolkit。容器内安装 Python 3.12，使用 OSMesa 无窗口渲染。

```bash
tar -xzf route-b-v170-cloud.tar.gz
cd route-b-v170-cloud
sha256sum -c SHA256SUMS

# NVIDIA GPU。根据服务器驱动选择 cu126、cu128 或 cu130。
docker build --build-arg TORCH_BACKEND=cu128 -t route-b:v170 .

mkdir -p resources outputs
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/resources:/opt/route-b/resources" \
  route-b:v170 prepare

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/resources:/opt/route-b/resources:ro" \
  route-b:v170 doctor --render --device cuda

# 完整执行 Object task 0 / init 0，保存双相机视频和评测结果。
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/resources:/opt/route-b/resources:ro" \
  -v "$PWD/outputs:/opt/route-b/outputs" \
  route-b:v170 smoke --device cuda
```

纯 CPU：构建时使用 `TORCH_BACKEND=cpu`，运行时去掉 `--gpus all` 并使用 `--device cpu`。准备资源不需要 GPU。GPU wheel 版本组合使用 [PyTorch 官方安装列表](https://pytorch.org/get-started/previous-versions/)中的 torch 2.11.0 / torchvision 0.26.0；原本地环境使用 cu130，本包默认提供 cu128 安装方式。

示例将容器用户映射为当前用户，下载资源和评测结果都归当前用户所有。容器的临时配置与 Hugging Face 缓存写入 `/tmp`。

## 已有 Ubuntu 24.04 云服务器

系统依赖安装一次；后续 Python 包安装在本目录的 `.venv` 中。

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv ca-certificates \
  libosmesa6 libgl1 libglib2.0-0

TORCH_BACKEND=cu128 ./setup.sh
./run.sh doctor --render --device cuda
./run.sh smoke --device cuda
```

`setup.sh` 包含依赖安装、固定版本资源下载和检查。CPU 安装使用 `TORCH_BACKEND=cpu ./setup.sh`。如果已在独立 Python 3.12 环境中，可用 `ROUTE_B_PYTHON=/path/to/python TORCH_BACKEND=cu128 bash scripts/install.sh`，再用同一个 `ROUTE_B_PYTHON` 执行 `run.sh prepare` 和后续命令。

## 跑任务

下面以直接安装为例；Docker 下把相同参数接在 `route-b:v170` 后即可。

```bash
# Spatial 10 个任务，每任务 5 个初态，共 50 次。
./run.sh run --suite spatial --episodes-per-task 5 --device cuda

# 指定任务和初态。suite 可选 spatial、object、goal、long。
./run.sh run --suite goal --task-ids 0,3 --init-start 5 \
  --episodes-per-task 2 --device cuda --run-name goal_probe_01

# 四套各 10 任务、每任务 50 个官方初态，共 2000 次，顺序执行。
./run.sh campaign --episodes-per-task 50 --device cuda --run-name cloud_2000_01

# 查看实际调度参数，不执行仿真。
./run.sh run --suite spatial --episodes-per-task 5 --dry-run

# 不需要视频时节省输出空间。
./run.sh run --suite object --episodes-per-task 5 --no-video

# 中断后使用相同参数、原运行名续跑；每次新实验使用新名称。
./run.sh campaign --episodes-per-task 50 --device cuda \
  --run-name cloud_2000_01 --resume
```

默认 seed=7、双相机 256×256；Spatial/Object/Goal/Long 的步数上限分别为 220/280/300/520。单套结果在 `outputs/<运行名>/summary.json`、`episodes.jsonl` 和 `videos/`；四套运行另有 `<运行名>_campaign.json`。不指定运行名时自动生成唯一名称。

此包提供云端运行和统计入口。85.9% 是原环境 v170 的历史正式结果；打包后的环境检查或单任务 smoke 不替代新的 2000 次完整评测。使用不同的 PyTorch CUDA wheel、OSMesa 版本或硬件，应保存新运行自己的分数。

## 资源与路径

| 内容 | 固定来源 |
|---|---|
| 感知模型 | `IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| 仿真资产 | `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，dataset 仓库 |
| 任务定义和初态 | `hf-libero==0.1.4` 安装包内置 |
| 模拟器 | `robosuite==1.4.0`、`mujoco==3.3.2` |

`resources.lock.json` 固定每个模型/资产文件的大小与 SHA256；`prepare` 和 `doctor` 会完整校验。运行评测时使用本地资源并关闭 Hugging Face 在线查询。

可通过环境变量指定已有资源，避免重复下载：

```bash
export LIBERO_ASSET_ROOT=/data/libero-assets
export ROUTE_B_MODEL_PATH=/data/grounding-dino-tiny
export LIBERO_CONFIG_PATH=/data/route-b-runtime/libero
./run.sh doctor --render
```

`run.sh` 负责设置依赖和资源路径，请通过该入口启动。LIBERO 的 YAML 配置会按当前安装位置生成，原机器的用户目录和 Conda 路径不需要存在。若下载站点需要代理，使用云服务器自己的网络/代理设置；包内不包含账号、Token 或代理配置。

## 包内保留与裁剪

- `libero_system/`：v170 Python 运行代码。保留 Route C 公共类型和几何模块，因为现有感知/集成模块会导入它们；启动入口固定执行 Route B。
- `cloud.py`、`run.sh`、`setup.sh`、`Dockerfile`：新增部署入口与安装方式。
- `provenance/`：原版本成绩摘要、源码差异记录、打包验证摘要。
- 不包含历史 candidates/development、录像、传感器转储、研究日志、旧 README、测试集、`.git`、`.local`、虚拟环境或模型缓存。

模拟器的三个上游包使用 `--no-deps` 安装，再由 `requirements.txt` 显式安装所需运行依赖，避免 hf-libero 的训练、实验跟踪和 notebook 依赖。`doctor` 检查实际运行导入和全部 40 条任务编译；`smoke` 用真实 MuJoCo 验证感知与控制链路。

录像使用 `imageio-ffmpeg` wheel 自带的编码器，无需另装系统 FFmpeg 及其音视频依赖。

控制器、感知和接触技能函数保留 v170 实现；源码中的两项模型/资产默认路径改为环境变量和项目相对路径。具体差异见 `provenance/PORTABILITY.patch` 与 `provenance/source.json`。新的部署入口只处理资源定位、参数和结果汇总。
