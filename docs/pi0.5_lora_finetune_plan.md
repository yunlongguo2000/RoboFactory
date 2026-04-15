# pi0.5 LoRA 微调方案 — RoboFactory LiftBarrier 任务

## Context

RoboFactory 的 Diffusion Policy 训练完成，但多 agent 成功率仅约 7%。我们希望尝试用 pi0.5（Physical Intelligence 的升级版 VLA 模型）通过 LoRA 微调来提升效果。

pi0.5 相比 pi0 在泛化能力上更强（open-world generalization）。

由于没有 VLA 原生支持多 agent，**先从单 agent（Agent0）开始**，验证整个 pipeline 可行后再扩展。

## 数据概况

- **数据源**: `data/zarr_data/LiftBarrier-rf_Agent0_150.zarr`
- **Episodes**: 150，共 15123 帧
- **FPS**: 10
- **相机**: `head_camera_agent0`（320x240 RGB，resize 到 224x224）
  - 非全局相机，但 Agent0 的相机能同时看到自己和对方（两台机器人面对面）
- **State/Action**: 均为 8 维（7 关节角度 + 1 夹爪），`pd_joint_pos` 控制模式
  - **注意**: state 和 action 数值完全相同（数据采集阶段的固有特点，pkl→zarr 转换时都来自 `joint_action`）
- **Action 空间**: 不同于 pi0.5 预训练的 32 维（DROID），我们用 8 维，模型内部自动 pad 到 32 维
- **数据流水线**: h5 → pkl → zarr → LeRobot v2.1

## 环境规划

| 环境 | 职责 |
|------|------|
| **RoboFactory 环境 (conda, Python 3.9)** | 仿真环境（ManiSkill）、数据生成、评估推理 |
| **openpi/.venv (uv, Python 3.11)** | 数据格式转换、norm_stats 计算、pi0.5 微调训练 |

openpi 的 `.venv` 中已包含 lerobot 库（v2.1，openpi fork 版本 0.1.0），不需要单独安装。

**openpi 公共代码不直接修改**，仅新增文件（`robofactory_policy.py`）和在 `config.py` 中添加配置（参考官方 Libero/ALOHA 的做法）。

## 总体流程

```
方案 B (openpi/JAX) — 主力：
1. ✅ 安装 openpi (uv sync)
2. ✅ 数据格式转换 (zarr → LeRobot v2.1)
3. ✅ 计算 norm_stats
4. ⬜ pi0.5 LoRA 微调训练
5. ⬜ 评估推理

方案 A (LeRobot/PyTorch) — 可选对比：
1. 在 openpi/.venv 中 pip install lerobot[pi05]
2. 用同一份 LeRobot 数据
3. pi0.5 expert-only fine-tuning
4. 评估推理
```

两套方案共享同一份 LeRobot 格式数据，通过 vepfs 交换。

## 目录结构

```
/root/projects/
├── RoboFactory/                  # 原项目
│   ├── data/                     # 软链接到 vepfs
│   ├── checkpoints/              # 软链接到 vepfs
│   └── scripts/
│       ├── convert_zarr_to_lerobot.py  # 数据转换脚本
│       └── compute_norm_stats.py       # norm_stats 计算（独立脚本，不依赖 JAX pipeline）
│
├── openpi/                       # openpi 项目（方案 B 主力）
│   ├── .venv/                    # uv 虚拟环境
│   ├── assets/
│   │   └── pi05_liftbarrier_lora/
│   │       └── robofactory_liftbarrier_agent0/
│   │           └── norm_stats.json     # ✅ 已生成
│   ├── scripts/
│   │   ├── train.py
│   │   └── compute_norm_stats.py
│   ├── src/openpi/
│   │   ├── policies/
│   │   │   └── robofactory_policy.py  # 新增：RoboFactory transforms + 自定义 WeightLoader
│   │   └── training/
│   │       └── config.py              # 新增：pi05_liftbarrier_lora 配置和 LeRobotRoboFactoryDataConfig
│   └── ...
│
└── /vepfs-mlp2/c20250510/250404002/
    ├── robofactory_data/         # 已有（含 pkl_data/Agent0, Agent1, global）
    ├── robofactory_checkpoints/  # 已有（openpi checkpoints 软链接到此）
    ├── robofactory_lerobot/      # ✅ LeRobot v2.1 格式数据
    │   ├── meta/                 # info.json, episodes.jsonl, tasks.jsonl
    │   ├── data/chunk-000/       # 150 个 parquet 文件（图片 bytes 内嵌）
    │   └── videos/chunk-000/     # mp4 视频（可选，用于查看）
    └── pi05_models/              # ✅ pi05_base 预训练权重（本地 TOS 下载）
        └── pi05_models/
            └── pi05_base/
                ├── assets/       # 各平台 norm_stats
                └── params/       # 模型权重（12.5 GiB）
```

## 步骤详情

### Step 1: 环境搭建 ✅

```bash
cd /root/projects/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

openpi 的 `.venv` 中已包含 lerobot 库（v2.1）。数据集需要软链接到 lerobot 默认目录：

```bash
mkdir -p /root/.cache/huggingface/lerobot
ln -s /vepfs-mlp2/c20250510/250404002/robofactory_lerobot /root/.cache/huggingface/lerobot/robofactory_liftbarrier_agent0
```

### Step 2: 数据格式转换 ✅

```bash
cd /root/projects/openpi
uv run python /root/projects/RoboFactory/scripts/convert_zarr_to_lerobot.py \
    --zarr_path /root/projects/RoboFactory/data/zarr_data/LiftBarrier-rf_Agent0_150.zarr \
    --output_dir /vepfs-mlp2/c20250510/250404002/robofactory_lerobot
```

LeRobot v2.1 格式：图片以 bytes 形式内嵌在 parquet 文件中（不是独立 png），openpi 的数据加载器能正确读取。

数据 key 映射（兼容两方案）：
- 图像: `observation.images.agent0`（openpi repack 映射为 `observation/image`）
- 状态: `observation.state`
- 动作: `actions`（8 维: 7 joint + 1 gripper）

### Step 3: 计算 norm_stats ✅

使用独立脚本（不走 openpi 的 JAX pipeline，避免 batch_size/设备数的 sharding 问题）：

```bash
cd /root/projects/openpi
uv run python /root/projects/RoboFactory/scripts/compute_norm_stats.py
```

输出: `assets/pi05_liftbarrier_lora/robofactory_liftbarrier_agent0/norm_stats.json`

norm_stats 中 state 和 actions 的 mean/std 完全相同，这是数据的固有特点（见"数据概况"）。

**注意**: `compute_norm_stats.py` 输出的 json 需要包含 `norm_stats` 外层 key，openpi 的 `_NormStatsDict` (pydantic) 要求此格式：
```json
{
  "norm_stats": {
    "state": {"mean": [...], "std": [...]},
    "actions": {"mean": [...], "std": [...]}
  }
}
```

### Step 4: 方案 B — openpi pi0.5 LoRA 训练 ⬜

训练配置 `pi05_liftbarrier_lora`：

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | pi05_base | 本地路径 `/vepfs-mlp2/c20250510/250404002/pi05_models/pi05_models/pi05_base` |
| LoRA | gemma_2b_lora + gemma_300m_lora | 冻结 VLM backbone，只训练 LoRA |
| action_dim | 8 | 7 joint + 1 gripper（模型内部 pad 到 32） |
| action_horizon | 10 | 每次预测 10 步 action chunk |
| batch_size | 4 | 4 卡数据并行，每卡 1 个样本 |
| fsdp_devices | 1 | LoRA 参数量小，无需 FSDP 分片 |
| EMA | 关闭 | LoRA 微调不需要 |
| 训练步数 | 30000 | |
| use_quantile_norm | False | 使用 z-score 归一化，不依赖 q01/q99 |

```bash
cd /root/projects/openpi
WANDB_API_KEY=<your_key> CUDA_VISIBLE_DEVICES=0,1,2,3 \
    uv run scripts/train.py pi05_liftbarrier_lora --exp-name=liftbarrier_agent0 --overwrite
```

### Step 5: 方案 A — LeRobot pi0.5 训练（可选）

```bash
cd /root/projects/openpi
uv run lerobot-train \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.train_expert_only=true \
  --dataset.repo_id=robofactory_liftbarrier_agent0 \
  --dataset.root=/vepfs-mlp2/c20250510/250404002/robofactory_lerobot \
  --output_dir=/vepfs-mlp2/c20250510/250404002/lerobot_outputs \
  --batch_size=1 \
  --steps=30000
```

### Step 6: 评估 ✅

采用 **Server-Client 模式**：openpi `serve_policy.py` 启动推理服务，评估脚本在 RoboFactory conda 环境中通过 websocket 请求 action。环境依赖完全隔离。

**终端 1**（openpi .venv）：启动推理服务
```bash
cd /root/projects/openpi
uv run scripts/serve_policy.py --port 8777 policy:checkpoint \
    --policy.config=pi05_liftbarrier_lora \
    --policy.dir=/vepfs-mlp2/c20250510/250404002/robofactory_checkpoints/pi05_liftbarrier_lora/liftbarrier_agent0/29999
```

**终端 2**（RoboFactory conda）：运行评估
```bash
conda activate RoboFactory
cd /root/projects/RoboFactory
python scripts/eval_pi05.py --num_episodes 10 --seed 10000
```

需要在 RoboFactory conda 环境中安装 `openpi_client`：
```bash
conda run -n RoboFactory pip install -e /root/projects/openpi/packages/openpi-client
```

pi0.5 直接输出关节角度（action chunk），不需要 TOPP 运动规划（DP 需要）。通过 `ActionChunkBroker` 管理 action chunk，每 10 步请求一次推理。

**`--port` 参数必须放在 `policy:checkpoint` 子命令之前**，否则会被当作子命令参数而报错。

## 关键文件

| 文件 | 说明 |
|------|------|
| `data/zarr_data/LiftBarrier-rf_Agent0_150.zarr` | 数据源（150 episodes, 15123 帧） |
| `scripts/convert_zarr_to_lerobot.py` | 数据转换脚本 |
| `scripts/compute_norm_stats.py` | 独立 norm_stats 计算脚本（输出含 norm_stats 外层 key） |
| `scripts/eval_pi05.py` | 评估脚本（Server-Client 模式，含 Panda1Controller） |
| `openpi/src/openpi/policies/robofactory_policy.py` | RoboFactory 输入输出 transforms + `RoboFactoryCheckpointWeightLoader`（action 维度截断） |
| `openpi/src/openpi/training/config.py` | `pi05_liftbarrier_lora` 训练配置 + `LeRobotRoboFactoryDataConfig` |
| `openpi/assets/pi05_liftbarrier_lora/robofactory_liftbarrier_agent0/norm_stats.json` | 归一化统计量（z-score 格式） |
| `/vepfs-mlp2/c20250510/250404002/pi05_models/pi05_models/pi05_base/` | 本地 pi05_base 预训练权重（assets + params） |
| `/vepfs-mlp2/c20250510/250404002/robofactory_lerobot/` | LeRobot v2.1 格式数据 |

## 归一化方式

openpi 支持两种归一化方式：

### Z-score（当前使用，use_quantile_norm=False）

```python
normalized = (x - mean) / std
```

- 用均值和标准差归一化
- 输出范围无界
- 适合小数据集、分布集中的数据（如关节角度）

### Quantile（use_quantile_norm=True）

```python
normalized = (x - q01) / (q99 - q01) * 2 - 1
```

- 用第 1 百分位（q01）和第 99 百分位（q99）做 min-max 归一化
- 输出范围固定在 [-1, 1]
- 对离群值更鲁棒，但需要额外统计 q01/q99
- 适合大规模数据集（如 DROID），150 episodes 不适合用 quantile

## 相机配置

pi0.5 模型期望 **1 个 base + 2 个 wrist**，共 3 个相机输入：

| 输入 key | 含义 | image_mask |
|---------|------|-----------|
| `base_0_rgb` | 第三视角（全局/环境） | 必须为 True |
| `left_wrist_0_rgb` | 左腕相机（操作细节） | 必须为 True |
| `right_wrist_0_rgb` | 右腕相机 | pi0.5 下为 False（不使用） |

当前 RoboFactory 只有 1 个相机（`head_camera_agent0`），两个 wrist 用零填充：
- `base_0_rgb` → `head_camera_agent0`（真实图像）
- `left_wrist_0_rgb` → 零填充（mask=False）
- `right_wrist_0_rgb` → 零填充（mask=False）

**后续优化**：加上 global 相机作为 base（第三视角），head_camera_agent0 作为 left_wrist（操作视角），需要重新采集数据。

## 多卡训练配置

| 方案 | fsdp_devices | batch_size | mesh shape | 说明 |
|------|-------------|------------|------------|------|
| **当前使用** | 1 | 4 | (4, 1) | 4 卡数据并行，LoRA 参数量小无需 FSDP |
| 备选 | 4 | 1 | (1, 4) | 4 卡 FSDP 分片（batch_size 必须能被总设备数整除） |

openpi 的 batch_size 检查：`batch_size % jax.device_count() != 0` 则报错。
FSDP 分片通过 `fsdp_devices` 控制，mesh 形状为 `(总卡数/fsdp_devices, fsdp_devices)`。

## WeightLoader：action 维度截断

pi0.5 预训练权重的 action 维度是 32（DROID），而我们用 8 维。`train.py` 中 `_load_weights_and_validate` 会严格校验 shape，导致加载失败。

**解决方案**：在 `robofactory_policy.py` 中新增 `RoboFactoryCheckpointWeightLoader`，加载权重时自动将 action 相关的权重从 32 维截断到 8 维。不修改 openpi 公共代码（`weight_loaders.py` 和 `train.py`）。

`config.py` 中使用自定义 WeightLoader：
```python
weight_loader=robofactory_policy.RoboFactoryCheckpointWeightLoader("/vepfs-mlp2/c20250510/250404002/pi05_models/pi05_models/pi05_base/params"),
```

## Checkpoint 存储

openpi 默认将 checkpoint 保存在 `./checkpoints/`（项目根目录下）。为避免本地磁盘空间不足，通过软链接将 checkpoint 目录指向 vepfs：

```bash
rm -rf /root/projects/openpi/checkpoints
ln -s /vepfs-mlp2/c20250510/250404002/robofactory_checkpoints /root/projects/openpi/checkpoints
```

## wandb 配置

- 需要设置 `WANDB_API_KEY` 环境变量（86 位新版 API key）
- `WANDB_API_KEY` 写入 `.bashrc` 时 `=` 后面不能有空格
- 临时禁用：`WANDB_MODE=disabled`
- 训练记录：loss、grad_norm、param_norm、camera_views（每 batch 前 5 个样本的 3 视角拼接图）

## 验证方式

1. ✅ 数据转换后 LeRobot 数据集可正常加载（150 episodes, 15123 帧, v2.1 格式）
2. ✅ norm_stats.json 生成成功（含 norm_stats 外层 key）
3. ✅ `uv run scripts/train.py pi05_liftbarrier_lora` 正常启动，loss 下降
4. ⬜ 评估脚本能加载模型并推理出 action

## 已解决的问题

1. **磁盘空间不足**: 清理 conda/pip/uv 缓存，删除 conda openpi 环境，改用 uv；checkpoint 软链接到 vepfs
2. **lerobot 版本**: openpi 内置 lerobot 0.1.0（v2.1 格式），无需单独安装
3. **RepackTransform key 映射**: LeRobot key 用 `.` 分隔（`observation.images.agent0`），不是 `/`，需用点号
4. **norm_stats sharding 错误**: openpi 的 `compute_norm_stats.py` 使用 JAX sharding，batch_size=1 不能被 4 设备整除 → 用独立脚本替代
5. **action_dim 差异**: pi0.5 预训练用 32 维，我们用 8 维 → `PadStatesAndActions` 自动 pad
6. **norm_stats 格式错误**: 独立脚本缺少 `norm_stats` 外层 key → pydantic `_NormStatsDict` 校验失败 → 修正脚本输出格式
7. **quantile norm 报错**: `use_quantile_norm=True` 但 norm_stats 中没有 q01/q99 → 在 `LeRobotRoboFactoryDataConfig` 中设置 `use_quantile_norm=False`
8. **WeightLoader action 维度不匹配**: 预训练权重 action 维度 32 vs 我们的 8 → 在 `robofactory_policy.py` 中新增 `RoboFactoryCheckpointWeightLoader` 做截断加载，不修改公共代码
9. **batch_size 与设备数不匹配**: `batch_size=1` 不能被 4 整除 → 改为 `batch_size=4, fsdp_devices=1`（4 卡数据并行）
10. **wandb API key 格式**: 新版 86 位 key，`.bashrc` 中 `=` 后面不能有空格
11. **wandb 升级**: 从 0.19.11 升到 0.26.0（`uv pip install -U wandb`）
12. **pi05_base 权重来源**: 开发机 TOS 桶 `tos://c20250510/yunlong/pi05_models/` 下载到 vepfs，通过本地路径加载（不走 GCS）

## 评估发现

### LiftBarrier 任务分析

LiftBarrier 是一个**双臂协作任务**：panda-0 和 panda-1 从 barrier 两侧同时抓取并抬升。

**训练数据来源**：通过 `robofactory/planner/solutions/lift_barrier.py` 中的 motion planner 生成：
```python
planner.move_to_pose_with_screw(pose=[pose1, pose2], move_id=[0, 1])  # 两个臂同时移动到抓取位
planner.close_gripper(close_id=[0, 1])  # 两个臂同时闭合夹爪
pose1[2] += 0.02; pose2[2] += 0.02
planner.move_to_pose_with_screw(pose=[pose1, pose2], move_id=[0, 1])  # 两个臂同时抬升
```

h5 原始数据中包含完整的双臂动作（`actions/panda-0` 和 `actions/panda-1`），但转换到 zarr 时只保留了 panda-0 的动作（`parse_pkl_to_zarr_dp.py` 中 `--agent_id=0`）。pi0.5 模型只学习了 panda-0 的动作。

**核心矛盾**：pi0.5 模型只控制 panda-0，但训练数据中 barrier 是被两个臂从两侧托起的。单臂无法独自完成任务。

### 初始评估结果（panda-1 固定不动）

| 版本 | 成功率 | panda-1 行为 |
|------|--------|-------------|
| v1（初始） | 0-10% | 固定在初始位置 |
| v2（夹爪闭合） | 0% | 固定位置 + 夹爪闭合（导致物理碰撞不稳定） |
| v3（保持位置+张开） | ~10% | 固定初始位置 + 夹爪张开 |

### Panda1Controller：motion planner 驱动 panda-1

在每个 episode 开始时（`env.reset()` 后），用 `PandaArmMotionPlanningSolver` 为 panda-1 规划完整的抓取+抬升轨迹，然后在评估过程中逐 step 回放。

**4 个阶段**：
1. **Pre-grasp**：移动到 barrier 抓取位上方 5cm（gripper 张开）
2. **Descend**：下降到抓取位（gripper 张开）
3. **Close gripper**：20 步渐进闭合夹爪（1.0 → -1.0）
4. **Lift**：抬升 15cm（gripper 闭合）

**关键实现**：
- `plan_screw` 生成单步轨迹（motion planning），从 barrier 当前位姿实时计算抓取点
- 抓取点使用 `model_data.json` 中的 `contact_points_pose[id=2]`（panda-1 侧的标注接触点）
- barrier 位置随机化（`randp_scale: [0.3, 0.05, 0.]`）不影响 motion planner，因为 planner 在每次 `env.reset()` 后从 barrier 的实际位姿重新规划
- 如果 `plan_screw` 失败，fallback 到 IK 直接求解
- trajectory 结束后（~80-87 步）进入 hold-last-pose 模式

**评估结果**（10 episodes, seed=10000）：

| 成功率 | 平均步数 |
|--------|---------|
| **60%**（6/10） | 成功 episode 约 80 步 |

相比无 panda-1 配合的 ~10% 有显著提升。

### 失败模式分析

1. **panda-0 抓空**（3 次失败）：panda-1 成功抓取并抬升了 barrier，但 pi0.5 模型没有学到"配合另一端抓取"的策略，barrier 一端被抬起但倾斜
2. **panda-1 "手松"**（1 次失败）：两个臂都抓起来了，但后续 panda-1 的夹爪在 trajectory 结束后可能因 physics 不稳定导致维持不住

### 评估数据输出

每次运行自动创建时间戳子目录：
```
eval_debug/<YYYYMMDD_HHMMSS>/
    ep{N}_data.npz          # panda-0 和 panda-1 的 action/state
    ep{N}_step0.png         # 每 episode 首帧图像

eval_video/pi05_liftbarrier/<YYYYMMDD_HHMMSS>/
    0.mp4 ~ N.mp4           # 评估视频（RecordEpisodeMA 自动生成）
```

npz 文件包含：
- `actions`: panda-0 的 action (N, 8)
- `states`: panda-0 的 state (N, 8)
- `panda1_actions`: panda-1 的 action (N, 8)
- `panda1_states`: panda-1 的实际 state (N, 8)

### 调试注意事项

1. **Websocket 代理问题**：mihomo 代理（端口 7890）会拦截 websocket 握手，需设置 `NO_PROXY=127.0.0.1,localhost`
2. **端口冲突**：Nginx 反向代理拦截 8000 端口，改用 8777
3. **配置路径**：`${CONFIG_DIR}` 等占位符需用 `robofactory.DIR_MAP` 替换；默认使用 `table/lift_barrier.yaml`（不需要 robocasa 数据集）
4. **Gripper 映射**：ManiSkill 用宽度 0~0.04，训练数据用 [-1, 1]（open=1, close=-1），公式：`gripper = width / 0.04 * 2 - 1`
5. **panda-1 的 qpos 维度**：9 维（7 关节 + 2 夹爪宽度），action 为 8 维（7 关节 + 1 抽象夹爪指令）
