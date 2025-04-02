#!/bin/bash

# 参数检查（允许 0 或 1 个参数）
if [ $# -gt 1 ]; then
    echo "Usage: $0 [CUDA_EXECUTABLE]"
    echo "Example（使用默认）: $0"
    echo "Example（自定义）: $0 /path/to/your/cuda_program"
    echo "默认可执行文件路径: ./cpp/build/IVF_FLAT_FP16_EXAMPLE"
    exit 1
fi

# 设置可执行文件路径（支持命令行参数或默认路径）
CUDA_EXEC="${1:-./cpp/build/IVF_FLAT_FP16_EXAMPLE}"

# 自动创建导出目录
mkdir -p ~/ns-report

# 执行性能分析
/usr/local/cuda/bin/ncu \
    --config-file off \
    --export ~/ns-report \
    --force-overwrite \
    --set full \
    "$CUDA_EXEC"