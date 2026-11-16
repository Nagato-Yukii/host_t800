# 来源及移植范围

- AMP_mjlab：`/home/ye/BeyondMimicExp/AMP_mjlab`，commit `6c7a2947fccc973e4af8e6d90e550400f1b6fcfc`，只读外部依赖。PPO及runner本地适配保留原 BSD-3-Clause 声明；原判别器与网络模块直接复用。
- HoST：`https://github.com/InternRobotics/HoST`，commit `70bb580949a336a920833700e4b5dc3bf7fe87ce`；原MIT许可在 `vendor/HoST/LICENSE`。参考 ground 起身任务的高度/朝向奖励、随机倒地姿态、上拉辅助、超速终止。本地另有 `/home/ye/Desktop/HoST-isaaclab3`，只读检查，没有修改。
- 众擎：`https://github.com/engineai-robotics/engineai_rl_lab`，commit `c53f7ec614e34f1ea514cefb935894082f3d4c2a`，许可在 `vendor/engineai_rl_lab/LICENSE.txt`。提取其 locomotion/amp/robots/t800.py 的PD/armature/effort，按语义映射到当前骨架。
- T800物理资产：`/home/ye/GMR/assets/engineai_t800/t800.xml`；外观STL复用前序重定向生成资产；GMR许可副本见 `assets/t800/GMR_LICENSE`。此处使用碰撞代理，不能替代精细碰撞和真实硬件标定。
- 起身动作：用户提供G1数据的T800重定向/分割结果，原105MB包逐字节复制，原数据许可在 `data/SOURCE_LICENSE.md`。

有意区别：固定iteration课程替代HoST基于成功率的课程；辅助力在完整episode内作用；严格5+2秒有限时域；保留AMP默认站姿偏移控制，而非HoST当前q增量控制；单critic AMP PPO，不复现HoST四critic算法；四类预稳定倒地库，不复现原HoST的每一种初始分布；不移植楼梯地形。新任务是基于AMP_mjlab的HoST风格起身实现，而非HoST完整等价复现。

源码哈希和软件版本见 `dependencies.json`、`input_provenance.json`。
