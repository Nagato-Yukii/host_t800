# 新机器安装与训练

本仓库为private，先使用有权限的GitHub账号登录。运行目标为Linux + NVIDIA GPU；本机验证Python 3.11、RTX 5070 Ti 16GB。已附运行资产，不需要GMR原目录或原始动作压缩包。

```bash
git clone https://github.com/Nagato-Yukii/host_t800.git
cd host_t800
conda create -n host_t800 python=3.11 pip -y
conda activate host_t800
```

先根据GPU和驱动安装兼容的CUDA版PyTorch。本机使用torch 2.13.0（CUDA 13.0）；下面为对应安装命令，其他驱动/CUDA组合需选择合适的PyTorch构建：

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
bash scripts/setup_env.sh
```

setup_env.sh安装锁定的主要运行依赖、当前项目，并把vendor内的观测历史补丁复制到**当前环境**的mjlab安装目录，首次修改前保留备份。请使用专用环境。通过run.sh执行时，vendor中的修改版rsl_rl优先于site-packages，不需要单独安装vendor根目录或rsl_rl；直接运行python脚本时必须同样配置PYTHONPATH。

```bash
./run.sh smoke --task host_slow --num-envs 8 --steps 400
# 从头训练标准HoST：每500 iterations减10个百分点拉力
./run.sh train --task host --num-envs 4096 --iterations 10000
# 慢撤力任务：3000开始20%，每500 iterations减5个百分点，5000起零拉力
./run.sh train --task host_slow --num-envs 4096 --iterations 10000
```

显存不足时减小num-envs。无图形界面默认使用EGL，需安装可用的NVIDIA驱动和EGL/OpenGL运行库。首次GPU编译需要等待。运行中的训练应置于tmux：`tmux new -s t800`。

续训需要自行从原机器复制checkpoint，仓库不含训练权重：

```bash
./run.sh train --task host_slow --num-envs 4096 --iterations 7000 \
  --checkpoint /path/to/model_4900.pt --resume-iteration 3000 \
  --log-dir logs/host_slow_resume3000
```

`--iterations`为追加次数。`--resume-iteration`只重设训练计数与课程，保留权重、优化器及归一化状态。

资产完整性清单：`docs/runtime_assets_manifest.json`。531段动作共约163MB、T800网格约17MB，数据直接存储在Git中，无需LFS。`scripts/prepare_*`是历史重建工具，其中源路径仍需按机器调整；正常训练无需运行它们。历史诊断脚本引用旧checkpoint路径，运行前也需自行修改。

验证范围：仓库内依赖在现有运行环境通过模型检查和小规模训练验证；尚未在另一台机器全新安装验证。安装失败时先检查Python、CUDA驱动与包版本，不要把其他环境的site-packages复制进来。
