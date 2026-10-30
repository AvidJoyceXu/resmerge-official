#!/bin/bash
# Modified for the ResMerge source release: portable examples and released configuration name.

################################################################################
# Single Task Evaluation Script (补测脚本)
# 
# 功能说明：
#   本脚本用于指定单一policy path和单一specific task id，进行补测。
#   适用于批量评估中某个测试失败后需要重新运行的情况。
#
# 使用方法：
#   bash eval_single_task.sh [config_name] [policy_path] [task_id]
#
# 参数说明：
#   config_name: Hydra配置文件名（不含.yaml后缀），默认为 libero_object_task6_lora_residual_sac_openvlaoft
#   policy_path: 要评估的policy路径（可以是目录路径或文件路径）
#   task_id: 要评估的task ID (0-9)
#
# 示例：
#   bash examples/embodiment/eval_single_task.sh libero_object_task6_lora_residual_sac_openvlaoft \
#       /absolute/path/to/checkpoint/actor/huggingface_model \
#       2
#
# 工作原理：
#   1. 使用Hydra override来设置：
#      - runner.eval_policy_path: 指定的policy路径
#      - env.eval.specific_reset_id: 指定的task ID
#      - runner.only_eval: 设置为True，只进行评估不训练
#   2. 测试结果会保存到独立的日志目录中
#   3. 输出评估结果和成功率
#
# 输出：
#   - 结果保存在: ${REPO_PATH}/results/single_eval_YYYYMMDD-HH:MM:SS/
#   - 日志文件: eval.log
#   - 汇总信息: evaluation_summary.txt
################################################################################

# 设置错误处理：遇到错误立即退出
set -e

# ============================================================================
# 环境变量设置
# ============================================================================

# 获取脚本所在目录的绝对路径
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取仓库根目录（embodiment目录的父目录的父目录）
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
# 评估脚本路径
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

# MuJoCo渲染设置（使用osmesa进行无头渲染）
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# CUDA和Hydra设置
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# ============================================================================
# 参数解析和验证
# ============================================================================

# 默认配置名称
BASE_CONFIG_NAME=${1:-"libero_object_task6_lora_residual_sac_openvlaoft"}

# 检查参数数量
if [ $# -lt 3 ]; then
    echo "错误: 需要提供至少3个参数"
    echo ""
    echo "实际接收到的参数数量: $#"
    echo "参数列表:"
    for i in $(seq 1 $#); do
        echo "  [$i] ${!i}"
    done
    echo ""
    echo "使用方法:"
    echo "  bash eval_single_task.sh [config_name] [policy_path] [task_id]"
    echo ""
    echo "参数说明:"
    echo "  config_name: Hydra配置文件名（不含.yaml后缀），可选，默认为 libero_object_task6_lora_residual_sac_openvlaoft"
    echo "  policy_path: 要评估的policy路径（可以是目录路径或文件路径），必需"
    echo "  task_id: 要评估的task ID (0-9)，必需"
    echo ""
    echo "示例:"
    echo "  bash examples/embodiment/eval_single_task.sh libero_object_task6_lora_residual_sac_openvlaoft \\"
    echo "      /path/to/policy \\"
    echo "      2"
    echo ""
    echo "注意:"
    echo "  - policy_path可以是目录路径（推荐）或文件路径"
    echo "  - task_id 应该是 0-9 之间的整数"
    exit 1
fi

# 获取参数
POLICY_PATH="${2}"
TASK_ID="${3}"

# 验证task_id是否为有效数字
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "错误: task_id 必须是数字，当前值: ${TASK_ID}"
    exit 1
fi

# 验证task_id范围
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -gt 9 ]; then
    echo "错误: task_id 必须在 0-9 之间，当前值: ${TASK_ID}"
    exit 1
fi

# 验证policy路径是否存在（可以是目录或文件）
if [ ! -d "$POLICY_PATH" ] && [ ! -f "$POLICY_PATH" ]; then
    echo "错误: Policy路径不存在: ${POLICY_PATH}"
    echo "      请检查路径是否正确"
    exit 1
fi

# 显示policy路径类型
if [ -d "$POLICY_PATH" ]; then
    echo "✓ Policy路径有效: ${POLICY_PATH} [目录]"
else
    echo "✓ Policy路径有效: ${POLICY_PATH} [文件]"
fi

# ============================================================================
# 结果目录设置
# ============================================================================

# 创建带时间戳的结果目录
RESULTS_DIR="${REPO_PATH}/results/single_eval_$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${RESULTS_DIR}"

# 汇总文件路径
SUMMARY_FILE="${RESULTS_DIR}/evaluation_summary.txt"

# 初始化汇总文件
{
    echo "=========================================="
    echo "Single Task Evaluation Summary"
    echo "=========================================="
    echo "Base Config: ${BASE_CONFIG_NAME}"
    echo "Policy Path: ${POLICY_PATH}"
    echo "Task ID: ${TASK_ID}"
    echo "Start Time: $(date)"
    echo ""
} > "${SUMMARY_FILE}"

# ============================================================================
# 测试执行
# ============================================================================

echo ""
echo "=========================================="
echo "开始评估"
echo "=========================================="
echo "配置名称: ${BASE_CONFIG_NAME}"
echo "Policy路径: ${POLICY_PATH}"
echo "评估Task ID: ${TASK_ID}"
echo "结果目录: ${RESULTS_DIR}"
echo "=========================================="
echo ""

# ====================================================================
# 构建Hydra命令
# ====================================================================
# 使用Hydra override来修改配置：
# 1. runner.eval_policy_path: 设置为指定的policy路径
# 2. env.eval.specific_reset_id: 设置为指定的task ID
# 3. runner.only_eval: 设置为True，只进行评估不训练
# 4. runner.logger.log_path: 设置为当前测试的日志目录
CMD="python ${SRC_FILE} \
    --config-path ${EMBODIED_PATH}/config/ \
    --config-name ${BASE_CONFIG_NAME} \
    runner.eval_policy_path=\"${POLICY_PATH}\" \
    env.eval.specific_reset_id=${TASK_ID} \
    runner.only_eval=True \
    runner.logger.log_path=${RESULTS_DIR}"

echo "执行命令:"
echo "  ${CMD}"
echo ""

# ====================================================================
# 执行评估
# ====================================================================
LOG_FILE="${RESULTS_DIR}/eval.log"

# 运行评估命令，将输出保存到日志文件
echo "正在运行评估..."
if ${CMD} > "${LOG_FILE}" 2>&1; then
    echo "✓ 评估成功完成"
    
    # 尝试从日志文件中提取成功率（根据实际日志格式调整）
    # 这里使用grep查找包含"success"的行，提取数字
    SUCCESS_RATE=$(grep -iE "success.*rate|success_rate|success rate|success.*:" "${LOG_FILE}" | tail -n 1 | grep -oE "[0-9]+\.[0-9]+|[0-9]+%" | head -n 1 || echo "N/A")
    
    # 记录到汇总文件
    {
        echo "=========================================="
        echo "Evaluation Result"
        echo "=========================================="
        echo "Status: SUCCESS"
        echo "Success Rate: ${SUCCESS_RATE}"
        echo "End Time: $(date)"
        echo ""
        echo "Log File: ${LOG_FILE}"
        echo "=========================================="
    } >> "${SUMMARY_FILE}"
    
    echo ""
    echo "=========================================="
    echo "评估完成！"
    echo "=========================================="
    echo "结果: 成功"
    echo "成功率: ${SUCCESS_RATE}"
    echo "日志文件: ${LOG_FILE}"
    echo "汇总文件: ${SUMMARY_FILE}"
    echo "结果目录: ${RESULTS_DIR}"
    echo "=========================================="
    
    exit 0
else
    echo "✗ 评估失败"
    
    # 记录到汇总文件
    {
        echo "=========================================="
        echo "Evaluation Result"
        echo "=========================================="
        echo "Status: FAILED"
        echo "End Time: $(date)"
        echo ""
        echo "Log File: ${LOG_FILE}"
        echo "请查看日志文件获取详细错误信息"
        echo "=========================================="
    } >> "${SUMMARY_FILE}"
    
    echo ""
    echo "=========================================="
    echo "评估失败！"
    echo "=========================================="
    echo "请查看日志文件获取详细错误信息:"
    echo "  ${LOG_FILE}"
    echo "汇总文件: ${SUMMARY_FILE}"
    echo "结果目录: ${RESULTS_DIR}"
    echo "=========================================="
    
    exit 1
fi
