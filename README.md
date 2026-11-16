# T800 AMP 起身 / HoST task

独立项目位于 `/home/ye/Desktop/t800_amp_getup`。复用现有 `amp_mjlab` Conda 环境、AMP_mjlab 的管理器、观测、PPO/AMP、MuJoCo-Warp、平面地形和通用奖励；没有修改已有项目或共享 Python 环境。

## 快速使用

```bash
cd /home/ye/Desktop/t800_amp_getup
./run.sh check
./run.sh smoke --num-envs 8 --steps 400
./run.sh train --num-envs 512 --iterations 10000
# 从已完成的 iteration 继续，恢复网络、判别器、归一化器、优化器和课程 iteration：
./run.sh train --num-envs 512 --iterations 2000 --checkpoint logs/<run>/model_8000.pt
# 无外力评估（目前 play 是无窗口 rollout）：
./run.sh play --num-envs 64 --checkpoint logs/<run>/model_10000.pt --steps 350 --assist-iteration 8000
```

目前已在本机使用4096环境训练；历史短测与验证记录见 docs/validation.md。训练不会自动上传日志。TensorBoard：`/home/ye/miniconda3/envs/amp_mjlab/bin/python -m tensorboard.main --logdir logs`。

`run.sh` 默认使用 `/home/ye/miniconda3/envs/amp_mjlab/bin/python` 和 `/home/ye/BeyondMimicExp/AMP_mjlab`。可设置 `T800_PYTHON`、`AMP_MJLAB_ROOT`。本机另有 `/home/ye/AMP_mjlab`（含其他任务修改），本项目优先绑定 BeyondMimicExp 下的版本；共享 rsl_rl 的 editable 安装不会被覆盖。

## 任务结构

- `t800_amp_getup/robot.py`：25 关节 T800，沿用重定向时的 GMR 骨架、惯量和碰撞代理，添加已有外观 mesh。自碰撞开启。
- `base_task.py`：`T800BaseTask` / `base_cfg()`，公共资产、PD、观测、50Hz 控制和通用正则项。可作为以后其他 T800 任务的基类。
- `host_task.py`：`T800HoSTTask(T800BaseTask)` / `host_cfg()`，倒地重置、持续上拉、5+2 秒状态机、真实终止观测。
- `mdp.py`：有界高度/直立/站稳奖励、能耗、终止规则。
- `motion_loader.py`：向量化的上游 AMP 特征；全部 531 条等概率抽动作，再抽有效相邻帧，禁止跨动作配对。
- `runner.py` / `amp_ppo.py`：保留上游许可证的本地适配副本。保留 AMP 判别器原奖励公式；修复终止特征、iteration 恢复、原始特征归一化统计和逐项日志，不覆盖共享库。

## PD 来源与关节映射

来源是众擎官方 [engineai_rl_lab T800 AMP 配置](https://github.com/engineai-robotics/engineai_rl_lab/blob/c53f7ec614e34f1ea514cefb935894082f3d4c2a/source/engineai_rl_lab/engineai_rl_lab/tasks/locomotion/amp/robots/t800.py)。完整来源副本在 vendor，数值在 `assets/t800/pd.json`。

| 关节组 | Kp | Kd |
|---|---:|---:|
| 髋 pitch / 膝 pitch | 180 | 5 |
| 髋 roll / yaw | 100 | 3 |
| 踝 pitch / roll | 40 | 0.3 |
| 腰 yaw | 100 | 5 |
| 肩 pitch / 肘 pitch | 60 | 0.3 |
| 肩 roll / yaw、肘 yaw | 50 | 0.3 |
| 头 pitch / yaw | 100 | 1 |

按语义匹配，因为数据骨架右臂编号从 J20 开始、头部 J27/J28，官方 AMP 则右臂 J18 开始、头部 J23/J24。没有按数字直接复制。官方 AMP 动作表排除头部，当前数据有25关节，本任务控制全部25关节。官方 armature 和 effort 上限同步映射；使用 MuJoCo 原生 PD，并非 IsaacLab actuator 后端的完全等价复现；velocity 参数记录在 PD JSON，未伪装成已实现厂商电机转速曲线。

动作是相对默认站姿的关节目标偏移，普通关节每单位动作0.75rad，头部0.15rad，动作截断±3；初始标准差0.35。它保留 AMP 控制结构，不使用原 HoST 的“当前 q 增量”控制器。物理步长0.005s、decimation=4。

## 固定课程和每个环境的时序

课程只在 `host_task.py::assistance_at_iteration` 中定义，以实际 PPO iteration 为准：

| iteration（0基） | 上拉力 / 体重 |
|---|---:|
|0–499|80%|
|500–999|70%|
|1000–1499|60%|
|1500–1999|50%|
|2000–2499|40%|
|2500–2999|30%|
|3000–3499|20%|
|3500–3999|10%|
|≥4000|0%|

在 `LINK_TORSO_YAW` 质心施加世界坐标 +Z 力，零外加力矩，按整个机器人质量计算，每个物理子步写入。贯穿0–7秒，不随身体朝向或站立阶段关闭；如果 iteration 在 episode 内跨过边界，下一控制步立即采用新阶段。质量约84.99kg，初始拉力约666.99N。没有质量随机化，因此每个环境质量相同；后续加入质量随机化时需同步改为逐环境质量。

每次 reset：从四类落地姿态（俯卧、仰卧、左侧、右侧）的512姿态库采样，随机yaw，零根速度和关节速度。姿态库使用随机关节/身体方向与部分数据首帧关节组合，提前在CPU MuJoCo中无驱动落地0.5s并检查穿透，稳定过程不占训练episode时间，也不关闭 episode 内拉力。符合 HoST 的倒地/姿态随机化思路，但不宣称是原 HoST reset 分布的精确复制。

- 起身阶段高度检查：第100步（2秒）根部离地高度低于0.4m终止；第250步（5秒）低于0.8m终止。恰好等于阈值通过，仅在指定步检查，日志为 `Termination/height_at_2s` 和 `Termination/height_at_5s`。
- 5–7秒：保持站立；不再因站立判据失稳提前终止。最终成功仍要求整个后2秒都满足完整站立判据。保留7秒时限与 catastrophic 异常保护；高度检查通过不等于成功。
- 最长350控制步，严格7秒。禁止随机初始episode年龄。时间相位和拉力比例进入策略观测。
- 有限时域任务：7秒结束不跨episode bootstrap；非终止部分正常使用价值函数。

站立判据：根高度0.88–1.15m、直立投影>0.92、双脚有真实接触/接触力幅值门槛、脚部接近地面、根投影在双脚包围范围内（带0.12m容差）、根速度<0.5m/s、角速度<1rad/s。根投影约束不是严格质心支撑多边形/ZMP证明。

## 奖励比例与日志

沿用 AMP_mjlab 的判别器公式 `r_amp = 0.08 * max(0, 1 - (D-1)^2/4)`。

**训练总奖励 = 0.75 × task_reward + 0.25 × r_amp**。task_reward 中普通项按0.02s积分，终止惩罚和成功奖励是单次值。不要把75/25误读为实际观测回报占比固定：实际贡献需看日志。

| task项 | 权重（每秒，除注明项） | 约束 |
|---|---:|---|
|body_height|3|根高度目标1m × 直立，限0–5s|
|head_height|1|头相对脚高度 × 直立，限0–5s|
|upright|1|直立 × 根高度门控，限0–5s|
|stand_hold|5|全程0–7s，严格站立判据|
|stand_still|1|站立判据及低速度，5–7s|
|success|+5单次|完整后2秒站立，7s正常结束|
|joint_acc|−2.5e−7|原 mjlab 关节加速度平方项，上限对应−0.5/s|
|joint_limits|−10|原 mjlab 软关节越限项，上限对应−1/s|
|action_rate|−0.1|HoST覆盖：直接动作差平方和，无截断，与本地 tracking 一致|
|torques|−2.5e−6|原 mjlab 力矩平方项，上限−0.5/s|
|joint_velocity|−0.001|原 mjlab 速度平方项，上限−0.5/s|
|energy|−2.5e−5|逐关节绝对机械功率求和，上限−0.5/s|
|self_collision|−0.1|原 AMP_mjlab 自碰撞项，上限−0.4/s|
|termination|−(10+8×剩余秒数)单次|提前退出不能通过少交正则惩罚获利|

控制台和 TensorBoard 每 iteration 输出全部 `Reward/*`（各 task 项是混合前每控制步贡献）、`Reward/amp_weighted`、`Reward/task_weighted`、`Reward/total`、`AMP/discriminator`、`Curriculum/*`、`Task/*`、`Termination/*`（发生重置时）及判别器/PPO损失。

额外防利用措施：不按正向高度增量累加奖励；不奖励单纯静止/存活；高度奖励封顶且要求直立；站立要求双脚接触，不能靠吊起悬空得分；倒地本身不提前终止；根过低/过高、逃离重置点、非有限数值、关节/根超速才触发异常终止；重置前 AMP 特征不与新episode拼接。

这些规则减少明显漏洞，不能替代长程对抗评估。当前 smoke 权重不是已训练起身策略。

## 数据与验证

`data/T800_get_up_only_531.zip` 是原105MB包的完整副本（SHA256见 `docs/input_provenance.json`）；531段全部导入，无静默筛除。`assets/motions/` 为按当前T800模型FK重建的50Hz AMP格式，总计115686帧，保持原速。原数据待复查/淘汰标签和41条分割边界标记保留在 `data/motion_manifest.json`。这些片段尚非动力学可行的专家保证。

验证记录见 `docs/validation.md`。原数据非商业许可继续适用；见 `data/SOURCE_LICENSE.md`。训练/验证应按源片段分组，不应把同一原动作的关联片段分别放到两边。

## 固定80%拉力的 test80 task

`T800Test80Task`（`t800_amp_getup/test80task.py`）继承 `T800HoSTTask`，复用同一个 `host_cfg`，只覆盖拉力课程：所有 iteration 始终施加 `0.8 × 身体质量 × 9.81` 的世界 +Z 拉力。倒地重置、5秒起身+2秒站立、奖励、AMP数据和终止条件完全继承 HoST。训练、续训和播放均保持80%，不会在4k iteration撤去拉力。

```bash
./run.sh train --task test80 --num-envs 512 --iterations 10000
./run.sh smoke --task test80 --num-envs 8 --steps 400
```

省略 `--task` 时仍运行原 `host` 课程。`test80` 默认日志目录带 `test80_` 前缀；checkpoint记录固定拉力课程标识。播放时 `--assist-iteration` 仍记录iteration，但不改变 test80 的80%拉力。

续训可用 `--resume-iteration 3000` 在载入 checkpoint 后重设 learning iteration；保留模型、判别器、归一化和优化器状态，课程从新计数开始。`--iterations` 表示追加迭代数。新日志目录记录 `resume_provenance.json` 和重设计数后的初始checkpoint。

## HoST 慢速撤力续训任务

`--task host_slow` 使用 `T800HoSTSlowTask` 继承 HoST，仅覆盖拉力课程。3000之前复用原课程；3000–3499为20%，3500–3999为15%，4000–4499为10%，4500–4999为5%，5000起为0%。复用当前HoST全部奖励、PD、重置和终止配置，动作变化惩罚为−0.1直接平方和。使用 `--resume-iteration 3000` 从指定checkpoint重设计数续训，原 `host`/`test80` 课程不变。

## GitHub 仓库与本机运行文件

私有仓库：`Nagato-Yukii/host_t800`。工作目录保持 `/home/ye/Desktop/t800_amp_getup`；GitHub仓库名不改变Python包名或训练入口。

Git仅保存源码、配置、测试、资产XML/PD和数据来源说明。`.gitignore` 排除 `logs/`、模型checkpoint、TensorBoard日志、视频、`assets/motions/`、`assets/t800/meshes/`、`data/*.zip|npz|npy`、`vendor/`、虚拟环境和凭据。忽略文件仍保留本机，不影响正在运行的tmux训练。

新机器clone后不能直接训练：需按 `docs/dependencies.json` 恢复外部AMP_mjlab依赖，设置 `T800_PYTHON`/`AMP_MJLAB_ROOT`，并恢复T800 meshes、531段动作数据和 `data/fallen_resets.npz`。这些资产可从原机器复制，或在补齐源资产后使用 `scripts/prepare_assets.py`、`prepare_motions.py`、`prepare_resets.py` 重建。脚本中的本机默认路径需按新机器调整。数据与资产的许可见 `docs/NOTICE.md`、`data/SOURCE_LICENSE.md` 和 `assets/t800/GMR_LICENSE`。
