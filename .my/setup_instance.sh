#!/usr/bin/env bash
# =============================================================
# SGLang 云 GPU 实例一键初始化脚本
# 适用: Ubuntu 20.04/22.04 + NVIDIA GPU（AutoDL 等云平台）
#
# ┌─────────────────────────────────────────────────────────┐
# │          每次换新机器 - 在本地 Mac 执行                    │
# └─────────────────────────────────────────────────────────┘
#
#   Step 1. 打包仓库（含所有分支和历史）
#     cd /Users/lichongmin/Documents/LocalRepo/sglang
#     git bundle create ~/sglang.bundle --all
#
#   Step 2. 上传文件到远端（torch_wheels 可选，如云服务器可访问 PyPI 则不需要）
#     scp -P <port> ~/sglang.bundle root@<host>:~/
#     scp -P <port> ~/.my/setup_instance.sh root@<host>:~/   # 本脚本
#     # scp -P <port> -r ~/torch_wheels root@<host>:~/       # 可选：离线 wheel
#
#   Step 3. 登录远端并执行
#     ssh -p <port> root@<host>
#     bash ~/setup_instance.sh
#
# ┌─────────────────────────────────────────────────────────┐
# │          启动 SGLang 服务                                 │
# └─────────────────────────────────────────────────────────┘
#   # 标准启动（flashinfer backend，需要 CUDA >= 12.9 / SM >= 75）
#   python -m sglang.launch_server \
#       --model-path /root/autodl-tmp/models/<model> \
#       --tp 1 --port 30000
#
#   # SM120 / CUDA 12.8 兼容启动（flashinfer 不支持时用此方式）
#   python -m sglang.launch_server \
#       --model-path /root/autodl-tmp/models/<model> \
#       --tp 1 --port 30000 \
#       --attention-backend triton \
#       --sampling-backend pytorch
#
# ┌─────────────────────────────────────────────────────────┐
# │          下载模型（modelscope，国内更快）                  │
# └─────────────────────────────────────────────────────────┘
#   export MODELSCOPE_CACHE=/root/autodl-tmp/.cache/modelscope
#   modelscope download --model Qwen/Qwen3-0.6B \
#       --local_dir /root/autodl-tmp/models/Qwen3-0.6B
#
# ┌─────────────────────────────────────────────────────────┐
# │          已知问题 & 解决方案                               │
# └─────────────────────────────────────────────────────────┘
#   Q: FlashInfer requires GPUs with sm75 or higher
#   A: GPU 是 SM120，CUDA 版本 < 12.9 导致 SM 检测失败
#      → 使用 --attention-backend triton --sampling-backend pytorch
#
#   Q: No space left on device (系统盘 30G 不够)
#   A: pip cache purge 清理缓存；数据盘已自动配置（见 Step 4）
#
#   Q: libnvrtc.so.13: cannot open shared object file
#   A: 已在 bashrc 中自动添加 nvidia/cu13/lib 到 LD_LIBRARY_PATH
#
# 【如果远端网络可以访问 GitHub，可跳过 bundle 直接 git clone】
#     CLONE_MODE=git GIT_PAT=<token> bash ~/setup_instance.sh
# =============================================================
set -euo pipefail

# ============ 配置区（按需修改）============
REPO_BRANCH="pr-fast-recovery"
REPO_DIR="$HOME/sglang"
BUNDLE_FILE="$HOME/sglang.bundle"
# clone 方式: "bundle"（本地上传） 或 "git"（远端拉取）
CLONE_MODE="${CLONE_MODE:-bundle}"
# git 模式下的 URL（PAT 可通过环境变量传入）
REPO_URL="https://github.com/tryryry/sglang.git"
if [ -n "${GIT_PAT:-}" ]; then
    REPO_URL="https://${GIT_PAT}@github.com/tryryry/sglang.git"
fi
# ============================================

echo "========================================="
echo "  SGLang Instance Setup"
echo "========================================="

# ------ 0. 基本工具 ------
echo "[0/7] 安装基础工具..."
apt update -qq
apt install -y -qq build-essential git curl wget vim htop tmux tree jq unzip \
    net-tools iotop sysstat > /dev/null 2>&1
echo "  ✅ 基础工具就绪"

# ------ 1. Clone 仓库 ------
echo "[1/7] 获取仓库代码..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  目录已存在: $REPO_DIR，跳过 clone"
else
    if [ "$CLONE_MODE" = "bundle" ]; then
        if [ -f "$BUNDLE_FILE" ]; then
            git clone "$BUNDLE_FILE" "$REPO_DIR"
            cd "$REPO_DIR"
            git remote set-url origin https://github.com/tryryry/sglang.git
            git checkout "$REPO_BRANCH" 2>/dev/null || git checkout -b "$REPO_BRANCH" "origin/$REPO_BRANCH"
            echo "  ✅ 从 bundle 恢复到 $REPO_DIR (分支: $REPO_BRANCH)"
        else
            echo "  ❌ 未找到 $BUNDLE_FILE，请先从本地上传:"
            echo "     本地执行: git bundle create ~/sglang.bundle --all"
            echo "     然后 scp: scp -P <port> ~/sglang.bundle user@host:~/"
            exit 1
        fi
    else
        git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
        echo "  ✅ 仓库已克隆到 $REPO_DIR (分支: $REPO_BRANCH)"
    fi
fi
cd "$REPO_DIR"

# ------ 2. 检查 GPU 驱动 ------
echo "[2/7] 检查 GPU 驱动..."
if command -v nvidia-smi &> /dev/null; then
    echo "  ✅ nvidia-smi 可用:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "  ⚠️  未检测到 nvidia-smi，请手动安装驱动"
    echo "  参考: apt install -y nvidia-driver-535"
fi

# ------ 2. 检查 CUDA ------
echo "[3/7] 检查 CUDA..."
if command -v nvcc &> /dev/null; then
    echo "  ✅ CUDA: $(nvcc --version | grep release | awk '{print $6}')"
else
    echo "  ⚠️  nvcc 不在 PATH 中（云镜像通常已预装，检查 /usr/local/cuda）"
    if [ -d /usr/local/cuda ]; then
        echo "  发现 /usr/local/cuda，添加到 PATH..."
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
    fi
fi

# ------ 3. Python 环境 ------
echo "[4/7] 配置 Python 环境..."

# AutoDL 数据盘路径（如不存在则忽略）
AUTODL_TMP="/root/autodl-tmp"
if [ -d "$AUTODL_TMP" ]; then
    export PIP_CACHE_DIR="$AUTODL_TMP/.pip_cache"
    export HF_HOME="$AUTODL_TMP/.cache/huggingface"
    export CONDA_ENVS_PATH="$AUTODL_TMP/conda_envs"
    export CONDA_PKGS_DIRS="$AUTODL_TMP/conda_pkgs"
    mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"
    echo "  ✅ 所有缓存已指向数据盘: $AUTODL_TMP"
fi

if command -v conda &>/dev/null; then
    echo "  ✅ 检测到 conda，使用当前 conda 环境（跳过 venv 创建）"
    pip install -U pip setuptools wheel -q
else
    VENV_DIR="$AUTODL_TMP/.venvs/sglang"
    if [ ! -d "$AUTODL_TMP" ]; then
        VENV_DIR="${HOME}/.venvs/sglang"  # 无数据盘时 fallback 到系统盘
    fi
    if [ -d "$VENV_DIR" ]; then
        echo "  虚拟环境已存在: $VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
        echo "  ✅ 创建虚拟环境: $VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    pip install -U pip setuptools wheel -q
fi

# ------ 4. 安装 PyTorch + SGLang 依赖 ------
echo "[5/7] 安装 PyTorch 与项目依赖..."

# 检测是否已安装 torch，已安装则跳过
EXISTING_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -n "$EXISTING_TORCH" ]; then
    echo "  ✅ PyTorch 已安装: $EXISTING_TORCH，跳过安装"
else
    # 自动检测 CUDA 版本选择 PyTorch wheel
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' || echo "")
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        CUDA_TAG="cu121"
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        CUDA_TAG="cu118"
    else
        CUDA_TAG="cu121"
        echo "  ⚠️  未检测到 CUDA 版本，默认使用 cu121"
    fi
    echo "  CUDA 版本: ${CUDA_VERSION:-未知}, tag: $CUDA_TAG"

    LOCAL_WHEELS="$HOME/torch_wheels"
    OFFICIAL_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

    if [ -d "$LOCAL_WHEELS" ] && ls "$LOCAL_WHEELS"/*.whl &>/dev/null; then
        echo "  发现本地 wheel 目录，离线安装..."
        pip install torch torchvision --no-index --find-links "$LOCAL_WHEELS" --no-deps -q
        pip install torch torchvision -q 2>/dev/null || true  # 补全依赖（如有网络）
    else
        echo "  无本地 wheel，使用官方源: $OFFICIAL_INDEX"
        pip install torch torchvision --index-url "$OFFICIAL_INDEX" -q
    fi
    echo "  ✅ PyTorch 安装完成: $(python3 -c 'import torch; print(torch.__version__)')"
fi

# 安装项目（editable mode）
PROJECT_DIR="$REPO_DIR"
if [ -f "$PROJECT_DIR/python/pyproject.toml" ]; then
    cd "$PROJECT_DIR/python"
    echo "  安装 sglang 项目依赖..."
    pip install -e "." --no-build-isolation \
        || echo "  ⚠️  pip install -e . 失败，请手动检查: cd $PROJECT_DIR/python && pip install -e . --no-build-isolation"
    cd "$PROJECT_DIR"
fi

# ------ 5. 常用 GPU 工具 ------
echo "[6/7] 安装 GPU 监控工具..."
pip install gpustat -q

# ------ 6. Shell 配置（追加到 .bashrc，幂等） ------
echo "[7/7] 配置 Shell 环境..."
# 先清理旧的 sglang-setup 块（防止重复或错误内容残留）
sed -i '/# >>> sglang-setup <<</,/# >>> sglang-setup-end <<</d' ~/.bashrc
MARKER="# >>> sglang-setup <<<"
if command -v conda &>/dev/null; then
    cat >> ~/.bashrc << 'BASHRC'

# >>> sglang-setup <<<
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# 数据盘缓存（系统盘空间有限，所有缓存放数据盘）
if [ -d /root/autodl-tmp ]; then
    export PIP_CACHE_DIR=/root/autodl-tmp/.pip_cache
    export HF_HOME=/root/autodl-tmp/.cache/huggingface
    export MODELSCOPE_CACHE=/root/autodl-tmp/.cache/modelscope
    export CONDA_ENVS_PATH=/root/autodl-tmp/conda_envs
    export CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs
fi
# nvidia cu13 库路径（SM120 GPU 需要）
NVIDIA_CU13_LIB=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/nvidia/cu13/lib
[ -d "$NVIDIA_CU13_LIB" ] && export LD_LIBRARY_PATH="$NVIDIA_CU13_LIB:${LD_LIBRARY_PATH:-}"
# NCCL 默认配置（按需调整）
export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=0
# 常用别名
alias gs='gpustat -cp'
alias ns='nvidia-smi'
alias gw='watch -n1 gpustat -cp'
# >>> sglang-setup-end <<<
BASHRC
else
    cat >> ~/.bashrc << 'BASHRC'

# >>> sglang-setup <<<
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# 数据盘缓存（系统盘空间有限，所有缓存放数据盘）
if [ -d /root/autodl-tmp ]; then
    export PIP_CACHE_DIR=/root/autodl-tmp/.pip_cache
    export HF_HOME=/root/autodl-tmp/.cache/huggingface
    export MODELSCOPE_CACHE=/root/autodl-tmp/.cache/modelscope
    source /root/autodl-tmp/.venvs/sglang/bin/activate
else
    source $HOME/.venvs/sglang/bin/activate
fi
# NCCL 默认配置（按需调整）
export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=0
# 常用别名
alias gs='gpustat -cp'
alias ns='nvidia-smi'
alias gw='watch -n1 gpustat -cp'
# >>> sglang-setup-end <<<
BASHRC
fi
echo "  ✅ Shell 配置已写入 ~/.bashrc"

echo ""
echo "========================================="
echo "  ✅ 初始化完成！"
echo "========================================="
echo ""
echo "快速验证:"
echo "  source ~/.bashrc"
echo "  python -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')\""
echo "  gs              # GPU 状态"
echo ""
echo "启动 SGLang 服务:"
echo "  python -m sglang.launch_server --model-path <model> --tp <num_gpus>"
