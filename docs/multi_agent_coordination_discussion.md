# 多机械臂协同协作讨论

## 背景

LiftBarrier 是一个双臂协作任务：panda-0 和 panda-1 从 barrier 两侧同时抓取并抬升。当前方案采用混合控制——panda-0 由 pi0.5 模型推理，panda-1 由 motion planner 驱动。两种不同的"智能体"在同一个物理任务中配合，本质上是一个 **AI + 规划器的混合协作系统**。

当前评估结果：成功率 50-60%（10 episodes），主要失败模式是 panda-0 "抓空"——pi0.5 没有学到配合另一端抓取的策略。

---

## 1. Panda-0（pi0.5 模型）的配合问题

### 时机协调

- 训练数据中两臂是**同步**闭合、**同步**抬升的（motion planner 同时控制）
- 评估中 panda-1 的 trajectory 是预规划好的，panda-0 完全不知道对方什么时候会闭合夹爪
- panda-0 太早闭合 → barrier 可能还没被 panda-1 夹住，一端翘起
- panda-0 太晚闭合 → panda-1 已经在抬了，barrier 倾斜，panda-0 可能抓不到正确的点

**核心问题**：pi0.5 能不能从视觉上"学习"到——看到 panda-1 的手接近 barrier 了，我就该准备抓了？它有 head_camera，理论上能看到对面。

### 自适应 / 重试

- panda-0 抓空后能不能**重新对准**？当前模型是一次性推理 10 步 action chunk，没有反馈循环
- 人类面对这种情况会怎么做？——发现没抓住，退回来，重新看一眼，再试一次
- 这涉及到 **closed-loop adaptation**，pi0.5 的 action chunk 模式天然不太擅长这个

### 从倾倒中恢复

- 如果 barrier 被抬起来但倾斜了，panda-0 能否感知到倾斜方向并调整夹爪角度？
- 训练数据中 barrier 几乎是水平抬升的，倾斜恢复可能不在分布内

---

## 2. Panda-1（Motion Planner）的改进空间

### 从被动回放到主动感知

当前 panda-1 是 **open-loop trajectory replay**——plan 一次，然后无脑执行完。但实际物理仿真中有很多不确定性：

- **Barrier 被 panda-0 碰偏了**：panda-1 按计划抓，但 barrier 不在预期位置了
- **抓取后 barrier 滑落**：panda-1 的夹爪可能没完全对准，物理摩擦力不够

**改进思路**：给 panda-1 加一个 **re-planning 机制**——每 N 步检查一下 barrier 位置是否偏离太多，如果偏离就重新规划。把 motion planner 从 open-loop 变成 closed-loop。

### 等待信号

- 当前 panda-1 不管 panda-0 在干嘛，自顾自执行
- 更好的方案：panda-1 观察 panda-0 的状态（比如已经接近 barrier 了），再开始自己的动作
- 这需要一个 **协调信号**——什么样的信息可以作为"两臂都准备好了"的判据？

---

## 3. 系统层面的有趣设计

### 方案 A：角色互换的对称训练

- 训练数据中采集**两种配置**：panda-0 抓左端 / panda-0 抓右端
- 这样 pi0.5 学到的策略对两端是对称的，不会过度依赖特定一侧
- 评估时可以随机分配哪个臂用模型、哪个用 planner

### 方案 B：让 pi0.5 输出"意图"而非直接动作

- pi0.5 不直接输出关节角度，而是输出 **期望的抓取点坐标 + 夹爪开合时机**
- 然后用 IK + motion planner 将意图转换为具体关节轨迹
- 这样就变成了 "pi0.5 做高层决策，motion planner 做底层执行"
- 有点像 **hierarchical policy** 的思路

### 方案 C：Panda-0 也用 Motion Planner，但由 pi0.5 决定"何时触发"

- pi0.5 做一个 **时机预测器**：当前帧是否应该开始抓取？
- 具体动作仍然由 motion planner 生成
- 模型只负责高层决策（when），planner 负责低层执行（how）

### 方案 D：对抗 / 博弈视角

- 把两个臂的配合看成一个 **decentralized POMDP**
- 每个臂只观察自己的局部信息，但需要达成共同目标
- 这和多智能体强化学习（MARL）的研究方向很接近
- pi0.5 作为其中一个 agent 的 policy，另一个用 planner，研究他们的 **emergent coordination**

---

## 4. 更深层的思考

### "配合"到底需要什么信息？

当前 panda-0 只看到自己的 head_camera（能同时看到两个机器人，因为面对面），但：

- 它能不能从图像中**推断出 panda-1 的意图**？（比如看到对方的手在动，在接近 barrier）
- 它能不能**预测 panda-1 什么时候会闭合夹爪**？
- 这本质上是一个 **theory of mind** 问题——agent 需要对另一个 agent 的行为建模

### 数据瓶颈

- 150 episodes 的数据量对学习协调行为可能不够
- 协调行为是一种 **long-horizon + sparse reward** 的模式，需要大量数据才能学到
- 也许可以考虑 **data augmentation**：对已有轨迹做时序偏移（panda-0 提前/延后几帧），制造更多协调场景

### 评估指标的问题

- 当前只看最终成功率，但其实可以分解成更细的指标：
  - **panda-0 抓取精度**：夹爪中心离目标点的距离
  - **协调时机差**：两臂闭合夹爪的时间差
  - **barrier 倾斜度**：抬升过程中 barrier 与水平面的夹角
- 这些中间指标能帮我们更精准地定位问题

---

## 5. 优先探索方向

考虑到当前的资源和条件，以下两个方向性价比最高：

1. **给 panda-1 加 re-planning**：实现简单（在 step 中加检测 + 重新 plan），可能直接提升成功率，而且对理解整个系统的行为很有帮助
2. **分析 pi0.5 在协调时机上的表现**：通过对比成功/失败 episode 中 panda-0 的动作时序，看模型是否真的在"等待"panda-1，还是完全无视对方
