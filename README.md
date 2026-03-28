# LoRA Intent Classification（意图识别微调项目）

基于 LoRA（Low-Rank Adaptation）对大语言模型进行 SFT 微调，用于意图识别分类任务。

## 项目特点

- **双模式训练**：支持 LlamaFactory CLI（YAML 配置）和纯 Python 脚本两种方式，自由选择
- **模块解耦**：训练、导出、推理、评估各自独立，互不依赖
- **配置驱动**：所有参数通过配置文件管理，零硬编码
- **可扩展**：标签列表可配置，轻松从二分类扩展到多分类
- **可替换模型**：不绑定特定模型，支持 Qwen、Llama、Baichuan、ChatGLM 等

## 环境版本

| 组件 | 版本 |
|------|------|
| LlamaFactory | v0.9.4.dev0 |
| TensorBoard | 2.9.0 |
| Python | 3.10 |
| 工作目录 | `/workspace/lora-intent-clf` |

## 目录结构

```
/workspace/lora-intent-clf/
├── README.md                          # 本文档
├── pyproject.toml                     # ruff 代码检查配置
├── .python-version                    # Python 版本锁定（3.10）
├── .gitignore
│
├── configs/                           # LlamaFactory YAML 配置
│   ├── train_lora_sft.yaml           #   训练配置（带详细注释）
│   ├── export_lora.yaml              #   LoRA 适配器导出/合并配置
│   ├── inference.yaml                #   推理配置（合并模型）
│   └── inference_lora.yaml           #   推理配置（基座 + LoRA 适配器）
│
├── data/                              # 数据目录（也是 dataset_dir 的值）
│   ├── dataset_info.json             #   LlamaFactory 数据集注册文件
│   ├── train.json                    #   训练集（示例数据）
│   ├── val.json                      #   验证集（示例数据）
│   └── test.json                     #   测试集（示例数据）
│
├── scripts/                           # Shell 脚本
│   ├── run_train_and_export.sh       #   一键训练 + 导出
│   ├── run_inference.sh              #   推理测试
│   └── setup_github.sh              #   创建 GitHub 仓库
│
└── src/                               # Python 脚本
    ├── __init__.py
    ├── config.py                     #   统一配置定义（与 YAML 参数对应）
    ├── train.py                      #   Python 训练脚本
    ├── export_model.py               #   Python 模型导出脚本
    ├── inference.py                  #   Python 推理脚本
    └── evaluate.py                   #   Python 评估脚本
```

## 硬件环境

| 项目 | 规格 |
|------|------|
| GPU | 4 × NVIDIA V100 (16GB) |
| 总显存 | 64GB |
| 精度 | FP16（V100 不支持 BF16） |

## DeepSpeed ZeRO-3 说明

Qwen3-8B 的 FP16 权重约 16GB，而单张 V100 只有 16GB 显存。默认 DDP 模式下每张 GPU 都需要加载完整模型副本，所以在加载模型阶段就会 OOM。

本项目使用 **DeepSpeed ZeRO-3** 解决此问题：将模型参数、梯度和优化器状态分片到所有 GPU 上，4 张 V100 各承担约 4GB 的模型权重，彻底解决显存不足问题，且不损失精度。

```
/workspace/lora-intent-clf/configs/deepspeed/
├── ds_z3_config.json           # ZeRO-3 标准版（推荐，4×V100 够用）
└── ds_z3_offload_config.json   # ZeRO-3 + CPU Offload（显存极端紧张时使用，速度较慢）
```

训练 YAML 中已通过 `deepspeed` 参数集成，无需额外操作。如果显存仍然紧张，可以在 YAML 中将 `ds_z3_config.json` 改为 `ds_z3_offload_config.json`。

## 快速开始

### 0. 前提条件

确保以下组件已安装：

```bash
# 检查 LlamaFactory CLI（v0.9.4.dev0）
llamafactory-cli version

# 检查 GPU
nvidia-smi

# 检查 TensorBoard（2.9.0）
tensorboard --version

# 安装 Python 依赖（使用 pip，不依赖 uv）
pip install -r requirements.txt
```

> **关于依赖管理**：本项目使用 `requirements.txt` 管理依赖，无需 uv。
> 服务器已预装的版本见 `requirements.txt` 注释，直接 `pip install -r requirements.txt` 即可。

### 1. 准备数据

替换 `/workspace/lora-intent-clf/data/` 目录下的示例数据为你的实际训练数据。

本项目使用 **Alpaca 格式**，数据格式为 JSON 数组，每条数据包含三个字段：

```json
[
  {
    "instruction": "你是一个意图识别助手。请根据用户输入的文本，判断其意图类别。只能输出以下标签之一：寿险意图、拒识。",
    "input": "我想了解一下终身寿险的保费是多少",
    "output": "寿险意图"
  },
  {
    "instruction": "你是一个意图识别助手。请根据用户输入的文本，判断其意图类别。只能输出以下标签之一：寿险意图、拒识。",
    "input": "今天天气怎么样",
    "output": "拒识"
  }
]
```

字段说明：

| 字段 | 含义 | 说明 |
|------|------|------|
| `instruction` | 系统指令 | 告诉模型任务是什么，包含可选标签列表 |
| `input` | 用户输入 | 需要分类的原始文本 |
| `output` | 标签 | 分类结果，只能是标签列表中的一个 |

**注意事项**：
- `instruction` 中的标签列表需要与实际标签一致
- 如果要扩展为多分类，只需在 `instruction` 中增加标签，并提供对应的训练数据
- 建议训练集和验证集的标签分布尽量均衡

---

## LlamaFactory 数据集格式详解

### 支持的格式类型

LlamaFactory 支持两种主要格式，以及它们的变体：

#### 1. Alpaca 格式（本项目使用）

适用于指令-响应对的场景，是最直观的格式。

**SFT 监督微调（本项目使用）**：
```json
{
  "instruction": "任务描述/系统指令",
  "input": "用户输入/上下文",
  "output": "模型输出/标签",
  "system": "系统提示词（可选）",
  "history": [["之前的问题", "之前的回答"]]
}
```

**预训练（PT）**：
```json
{ "text": "这是一段无标注的文档文本，仅用于语言建模..." }
```

**偏好数据（DPO/ORPO/SimPO）**：
```json
{
  "instruction": "任务描述",
  "input": "用户输入",
  "chosen": "更好的回答",
  "rejected": "较差的回答"
}
```

**KTO 数据**：
```json
{
  "instruction": "任务描述",
  "input": "用户输入",
  "output": "模型回答",
  "kto_tag": true
}
```

#### 2. ShareGPT 格式

适用于多轮对话场景，更接近 OpenAI 的对话格式。

**SFT 监督微调**：
```json
{
  "conversations": [
    {"from": "human", "value": "你好，请帮我分类这段文本"},
    {"from": "gpt", "value": "寿险意图"}
  ]
}
```

**带工具调用的对话**：
```json
{
  "conversations": [
    {"from": "human", "value": "帮我查询天气"},
    {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {}}"},
    {"from": "observation", "value": "{\"temperature\": 25}"},
    {"from": "gpt", "value": "今天温度是25度"}
  ],
  "tools": "[{\"name\": \"get_weather\", ...}]"
}
```

**偏好数据（DPO）**：
```json
{
  "conversations": [
    {"from": "human", "value": "问题"}
  ],
  "chosen": {"from": "gpt", "value": "好的回答"},
  "rejected": {"from": "gpt", "value": "差的回答"}
}
```

**多模态数据（图片/视频/音频）**：
```json
{
  "messages": [
    {"role": "user", "content": "<image>这张图片里是什么？"},
    {"role": "assistant", "content": "这是一只猫"}
  ],
  "images": ["path/to/cat.jpg"]
}
```

### 支持的文件类型

JSON、JSONL、CSV、Parquet、Arrow

### dataset_info.json 完整字段说明

```json
{
  "数据集名称": {
    "file_name": "数据文件路径（相对于 dataset_dir 或绝对路径）",
    "formatting": "alpaca 或 sharegpt（默认 alpaca）",
    "ranking": true,
    "subset": "数据子集名称（HuggingFace 数据集适用）",
    "split": "数据集分割（默认 train）",
    "num_samples": 1000,
    "columns": {
      "prompt": "instruction 字段名",
      "query": "input 字段名",
      "response": "output 字段名",
      "system": "system 字段名",
      "history": "history 字段名",
      "messages": "conversations 字段名（ShareGPT）",
      "chosen": "chosen 字段名",
      "rejected": "rejected 字段名",
      "kto_tag": "kto_tag 字段名",
      "images": "images 字段名",
      "videos": "videos 字段名",
      "audios": "audios 字段名",
      "tools": "tools 字段名"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "observation_tag": "observation",
      "function_tag": "function_call"
    }
  }
}
```

---

## dataset 和 dataset_dir 的关系

这是 LlamaFactory 数据加载中最容易出错的地方，必须理解清楚：

```
dataset_dir (目录)
    │
    ├── dataset_info.json          ← LlamaFactory 在 dataset_dir 下查找此文件
    │       │
    │       ├── "intent_clf_train" ← dataset 参数的值，是此 JSON 的 key
    │       │       └── "file_name": "train.json"  ← 相对于 dataset_dir 的路径
    │       │
    │       ├── "intent_clf_val"
    │       │       └── "file_name": "val.json"
    │       │
    │       └── "intent_clf_test"
    │               └── "file_name": "test.json"
    │
    ├── train.json                 ← 实际数据文件
    ├── val.json
    └── test.json
```

**加载流程**：

1. LlamaFactory 根据 `dataset_dir` 找到 `dataset_info.json`
   - 路径拼接：`{dataset_dir}/dataset_info.json`
   - 本项目：`/workspace/lora-intent-clf/data/dataset_info.json`

2. 根据 `dataset` 参数在 `dataset_info.json` 中查找对应的 key
   - `dataset: intent_clf_train` → 查找 JSON 中 `"intent_clf_train"` 这个 key

3. 根据 key 对应的 `file_name` 加载数据文件
   - 如果 `file_name` 是相对路径 → 相对于 `dataset_dir` 拼接
   - 如果 `file_name` 是绝对路径 → 直接使用

**本项目配置**：

```yaml
# train_lora_sft.yaml 中的设置
dataset_dir: /workspace/lora-intent-clf/data    # ← 绝对路径！
dataset: intent_clf_train                        # ← dataset_info.json 中的 key
eval_dataset: intent_clf_val                     # ← 验证集（也是 dataset_info.json 中的 key）
```

```json
// data/dataset_info.json 中的定义
"intent_clf_train": {
  "file_name": "train.json",     // 相对于 dataset_dir → /workspace/lora-intent-clf/data/train.json
  "formatting": "alpaca"
}
```

**常见错误**：
- `dataset_dir: .` 或 `dataset_dir: ./data` → 使用相对路径时，如果你不在项目根目录执行命令就会报错 `No such file or directory`
- 解决方案：**一律使用绝对路径**

---

## 三个数据集各自的用途

| 数据集 | dataset_info.json 中的 key | 使用时机 | 使用方式 |
|--------|---------------------------|----------|----------|
| train.json | `intent_clf_train` | 训练阶段 | YAML: `dataset: intent_clf_train` |
| val.json | `intent_clf_val` | 训练过程中的评估 | YAML: `eval_dataset: intent_clf_val` |
| test.json | `intent_clf_test` | 训练完成后的最终测试 | Python: `python src/evaluate.py` 或 `llamafactory-cli chat` 手动测试 |

说明：
- **训练集**通过 `dataset` 参数指定，LlamaFactory 用它来更新模型权重
- **验证集**通过 `eval_dataset` 参数指定，在训练过程中定期评估模型表现（每 `eval_steps` 步一次），用于监控过拟合
- **测试集**不参与训练过程，而是在训练全部结束、模型导出之后，通过推理脚本或评估脚本进行最终的性能评估

---

### 2. 方案选择

本项目提供两种完全等价的训练方案：

| 特性 | 方案 A：LlamaFactory CLI | 方案 B：Python 脚本 |
|------|--------------------------|---------------------|
| 配置方式 | YAML 文件 | Python dataclass + JSON |
| 启动方式 | `llamafactory-cli train` | `python src/train.py` |
| 依赖 | LlamaFactory v0.9.4.dev0 | transformers + peft |
| 适用场景 | 快速实验、参数调优 | 需要定制化训练流程 |
| 多 GPU | 自动支持 | 需配合 accelerate/torchrun |

---

## 方案 A：使用 LlamaFactory CLI（推荐快速实验）

### A1. 训练

```bash
# 使用绝对路径指定配置文件，可以在任意目录执行
llamafactory-cli train /workspace/lora-intent-clf/configs/train_lora_sft.yaml
```

### A2. 查看训练日志（TensorBoard 2.9.0）

```bash
# 启动 TensorBoard（训练过程中或训练结束后均可）
tensorboard --logdir /workspace/lora-intent-clf/saves/qwen3-8b/lora/sft --port 6006

# 浏览器访问 http://localhost:6006
```

TensorBoard 中可以查看：训练/验证 loss 曲线、学习率变化、训练步数等。

### A3. 导出合并模型

```bash
# 将 LoRA 适配器合并到基座模型
llamafactory-cli export /workspace/lora-intent-clf/configs/export_lora.yaml
```

导出后的完整模型保存在 `/workspace/lora-intent-clf/models/qwen3-8b-intent-clf/`，可以像普通 HuggingFace 模型一样加载。

### A4. 推理测试

```bash
# 使用合并模型进行交互式推理
llamafactory-cli chat /workspace/lora-intent-clf/configs/inference.yaml

# 或者使用基座模型 + LoRA 适配器推理（无需先导出）
llamafactory-cli chat /workspace/lora-intent-clf/configs/inference_lora.yaml
```

### A5. 一键训练 + 导出

```bash
# 一键执行完整流程：训练 → 导出（可以在任意目录执行）
bash /workspace/lora-intent-clf/scripts/run_train_and_export.sh

# 可选参数：
bash /workspace/lora-intent-clf/scripts/run_train_and_export.sh --skip-train   # 仅导出
bash /workspace/lora-intent-clf/scripts/run_train_and_export.sh --skip-export   # 仅训练
```

### A6. 后台训练（SSH 断开/重启不丢失）

```bash
# 后台启动训练（nohup，断开 SSH 也不中断）
bash /workspace/lora-intent-clf/scripts/train_background.sh

# 查看实时日志
bash /workspace/lora-intent-clf/scripts/train_background.sh --log
# 或
tail -f /workspace/lora-intent-clf/logs/train_latest.log

# 查看运行状态
bash /workspace/lora-intent-clf/scripts/train_background.sh --status

# 终止训练
bash /workspace/lora-intent-clf/scripts/train_background.sh --kill
```

### A7. 查看 TensorBoard

```bash
# 前台启动（Ctrl+C 退出，不影响训练）
bash /workspace/lora-intent-clf/scripts/run_tensorboard.sh

# 后台启动
bash /workspace/lora-intent-clf/scripts/run_tensorboard.sh --background

# 指定端口
bash /workspace/lora-intent-clf/scripts/run_tensorboard.sh --port 6007

# 终止后台 TensorBoard
bash /workspace/lora-intent-clf/scripts/run_tensorboard.sh --kill
```

> **SSH 隧道访问**（服务器无公网端口时）：在本地机器执行：
> ```bash
> ssh -L 6006:localhost:6006 user@<服务器IP>
> ```
> 然后浏览器打开 `http://localhost:6006`

---

## 方案 B：使用 Python 脚本

### B0. 安装 Python 依赖

```bash
cd /workspace/lora-intent-clf
# 安装生产依赖
pip install -r requirements.txt

# 安装开发依赖（含 ruff + pytest）
pip install -r requirements-dev.txt
```

### B1. 训练

> **⚠️ 关于 OOM 问题**：直接用 `python src/train.py` 会在单 GPU 上加载完整模型，8B FP16 ≈ 16GB 直接撑满 V100，必然 OOM。解决方式是用 `torchrun` 启动并开启 DeepSpeed ZeRO-3，原理与 LlamaFactory CLI 方案相同——将模型参数分片到多张 GPU。

```bash
cd /workspace/lora-intent-clf

# ✅ 推荐：多卡 + DeepSpeed ZeRO-3（解决 OOM）
torchrun --nproc_per_node=4 src/train.py

# ✅ 后台运行（防止 SSH 断开丢失任务）
bash scripts/train_background.sh --python

# 命令行覆盖参数（仍需 torchrun 启动）
torchrun --nproc_per_node=4 src/train.py \
    --lora_rank 32 \
    --learning_rate 2e-4 \
    --num_train_epochs 2

# 禁用 DeepSpeed（仅单卡测试小模型时使用）
torchrun --nproc_per_node=1 src/train.py --deepspeed none
```

可用的命令行参数（均有默认值，可按需覆盖）：

| 参数 | 默认值 | 对应 YAML |
|------|--------|-----------|
| `--model_name_or_path` | `/workspace/Qwen3-8B` | `model_name_or_path` |
| `--template` | `qwen` | `template` |
| `--lora_rank` | `16` | `lora_rank` |
| `--lora_alpha` | `32` | `lora_alpha` |
| `--lora_dropout` | `0.05` | `lora_dropout` |
| `--lora_target` | `all` | `lora_target` |
| `--train_file` | `/workspace/.../data/train.json` | `dataset` (in dataset_info) |
| `--val_file` | `/workspace/.../data/val.json` | `eval_dataset` (in dataset_info) |
| `--max_seq_length` | `512` | `cutoff_len` |
| `--output_dir` | `/workspace/.../saves/qwen3-8b/lora/sft` | `output_dir` |
| `--num_train_epochs` | `5.0` | `num_train_epochs` |
| `--per_device_train_batch_size` | `2` | `per_device_train_batch_size` |
| `--gradient_accumulation_steps` | `4` | `gradient_accumulation_steps` |
| `--learning_rate` | `1e-4` | `learning_rate` |
| `--lr_scheduler_type` | `cosine` | `lr_scheduler_type` |
| `--warmup_ratio` | `0.1` | `warmup_ratio` |
| `--weight_decay` | `0.01` | `weight_decay` |
| `--max_grad_norm` | `1.0` | `max_grad_norm` |
| `--fp16` | `True` | `fp16` |
| `--seed` | `42` | `seed` |
| `--logging_steps` | `10` | `logging_steps` |
| `--save_steps` | `500` | `save_steps` |
| `--eval_steps` | `500` | `eval_steps` |
| `--report_to` | `tensorboard` | `report_to` |

### B2. 查看 TensorBoard

```bash
tensorboard --logdir /workspace/lora-intent-clf/saves/qwen3-8b/lora/sft/logs --port 6006
```

### B3. 导出合并模型

```bash
cd /workspace/lora-intent-clf

# 使用默认配置导出
python src/export_model.py

# 指定适配器路径和导出目录
python src/export_model.py \
    --adapter_path /workspace/lora-intent-clf/saves/qwen3-8b/lora/sft \
    --export_dir /workspace/lora-intent-clf/models/qwen3-8b-intent-clf
```

### B4. 推理

```bash
cd /workspace/lora-intent-clf

# 交互式推理（使用合并模型）
python src/inference.py

# 单条推理
python src/inference.py --input "我想买寿险"

# 批量推理
python src/inference.py \
    --input_file /workspace/lora-intent-clf/data/test.json \
    --output_file /workspace/lora-intent-clf/results/predictions.json

# 使用 LoRA 适配器推理（无需先导出）
python src/inference.py \
    --adapter_path /workspace/lora-intent-clf/saves/qwen3-8b/lora/sft \
    --input "我想买寿险"
```

### B5. 批量评估

LlamaFactory CLI 的内置 `eval` 指标只有 loss/perplexity，**不适合分类任务**。推荐以下两种方式：

**方式一（推荐）：LlamaFactory 批量预测 + Python 解析**

```bash
# Step 1：LlamaFactory 在测试集上批量推理，生成预测文件
llamafactory-cli train /workspace/lora-intent-clf/configs/predict_lora.yaml

# Step 2：Python 解析预测文件，输出 accuracy / F1 / 分类报告
python src/evaluate.py \
    --pred_file /workspace/lora-intent-clf/saves/qwen3-8b/lora/predict/generated_predictions.jsonl \
    --output_file /workspace/lora-intent-clf/saves/qwen3-8b/lora/predict/eval_report.json

# 一键脚本（两步合一）
bash scripts/run_evaluate.sh
```

**方式二：Python 脚本直接推理评估**（无 LlamaFactory 时使用）

```bash
# 使用 LoRA 适配器直接推理
python src/evaluate.py \
    --adapter_path /workspace/lora-intent-clf/saves/qwen3-8b/lora/sft \
    --test_file /workspace/lora-intent-clf/data/test.json \
    --output_file /workspace/lora-intent-clf/results/eval_report.json

# 或一键脚本
bash scripts/run_evaluate.sh --python
```

评估输出示例：

```
======================================================================
分类评估报告
======================================================================
标签                 精确率       召回率         F1       支持数
----------------------------------------------------------------------
寿险意图            1.0000     1.0000     1.0000          2
拒识                1.0000     1.0000     1.0000          2
----------------------------------------------------------------------
macro avg           1.0000     1.0000     1.0000          4
weighted avg        1.0000     1.0000     1.0000          4
----------------------------------------------------------------------
准确率: 1.0000 (4/4)
======================================================================
```

### B6. 多 GPU 训练（Python 方案）

Python 脚本方案使用多 GPU 时，需配合 `torchrun` 或 `accelerate`：

```bash
cd /workspace/lora-intent-clf

# 方式一：torchrun（推荐）
torchrun --nproc_per_node=4 src/train.py

# 方式二：accelerate
accelerate launch --num_processes=4 src/train.py
```

---

## 自定义配置

### 使用 JSON 配置文件（Python 方案）

可以将所有参数保存为 JSON 文件：

```bash
cd /workspace/lora-intent-clf

# 生成默认配置模板
python -c "
from src.config import ProjectConfig
cfg = ProjectConfig()
cfg.save('my_config.json')
print('配置已保存到 my_config.json')
"

# 编辑 my_config.json 后使用
python src/train.py --config my_config.json
```

### 更换基座模型

修改以下文件中的模型相关配置：

**YAML 方案**：编辑 `configs/train_lora_sft.yaml`

```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct  # 替换模型
template: llama3                                           # 替换模板
```

**Python 方案**：通过命令行参数

```bash
python src/train.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --template llama3
```

常见模型和模板对照：

| 模型 | template |
|------|----------|
| /workspace/Qwen3-8B (当前) | qwen |
| Qwen/Qwen2.5-7B-Instruct | qwen |
| meta-llama/Meta-Llama-3-8B-Instruct | llama3 |
| baichuan-inc/Baichuan2-7B-Chat | baichuan2 |
| THUDM/glm-4-9b-chat | chatglm4 |

### 扩展为多分类

只需两步：

1. **修改数据**：在 `instruction` 中添加新标签，并提供对应训练数据

```json
{
  "instruction": "...只能输出以下标签之一：寿险意图、车险意图、健康险意图、拒识。",
  "input": "车险怎么理赔",
  "output": "车险意图"
}
```

2. **修改 Python 配置**（如果使用 Python 方案）：编辑 `src/config.py` 中的 `labels` 列表

```python
labels: List[str] = field(
    default_factory=lambda: ["寿险意图", "车险意图", "健康险意图", "拒识"]
)
```

YAML 方案无需修改配置文件，因为标签信息完全在 instruction 中定义。

---

## 大数据量场景的 LoRA 最佳实践（~6万条，二分类）

### 推荐参数配置

| 参数 | 当前值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `num_train_epochs` | 5 | **2~3** | 6万条数据每轮已有充足样本，5轮易过拟合 |
| `lora_rank` | 16 | **16** | 二分类任务 rank=16 足够，无需更高 |
| `lora_alpha` | 32 | **32** | 保持 alpha=2×rank 的经验比例 |
| `lora_target` | all | **all** | 全参数目标对分类效果最优 |
| `learning_rate` | 2e-4 | **2e-4** | 当前值合理，可以试 5e-4（配合短 warmup） |
| `per_device_train_batch_size` | 2 | **2** | V100 16GB FP16 的显存极限，不建议增大 |
| `gradient_accumulation_steps` | 4 | **4~8** | 有效 batch = 2×4×4=32，8 则为 64 |
| `warmup_ratio` | 0.05 | **0.05** | 约 360 步 warmup，合理 |
| `lr_scheduler_type` | cosine | **cosine** | 长训练首选，自然衰减 |
| `cutoff_len` | 512 | **256~512** | 意图识别句子短，256 足够且速度更快 |
| `eval_steps` | 500 | **1000~2000** | 每次 eval 耗时 612s，减少频率节省时间 |

### 关键建议

**epochs 选择原则**：数据量越大，所需 epochs 越少。6万条数据的经验：
- 1 epoch = 快速验证，看 loss 是否收敛方向正确
- 2 epochs = 多数情况已足够（每类3万条，模型见过足够多样本）
- 3 epochs = 保守选择，适合类别不均衡时
- 5 epochs（当前）= 数据量少（<1万条）时才需要，6万条大概率过拟合

**eval_steps 建议**：当前每次验证耗时 612s（≈10min），eval_steps=500 意味着每 500 步停下来验证 10min，5 轮训练共验证 14 次 = 额外 143min。将 eval_steps 改为 2000，仅额外花费 36min，节省近 2 小时。

**cutoff_len 建议**：意图识别的输入文本通常在 50 字以内，instruction 加上 input 合计不超过 200 token。将 cutoff_len 从 512 改为 256，显存占用降低约 30%，速度可能提升 20%。

---

## Step 数量计算公式

理解 step 数量的计算逻辑，才能准确预估训练时间。

### 公式

```
有效 batch 大小  = per_device_train_batch_size × GPU 数量 × gradient_accumulation_steps
每轮步数        = ceil(训练集样本数 ÷ 有效 batch 大小)
总步数          = 每轮步数 × num_train_epochs
```

### 当前配置计算

```
训练集样本数    = 60,000 × 80% = 48,000 条
per_device_batch = 2
GPU 数量        = 4
gradient_accum  = 4
有效 batch      = 2 × 4 × 4 = 32

每轮步数        = ceil(48,000 ÷ 32) = 1,500
总步数          = 1,500 × 5 = 7,500（实际 7,195，误差来自数据集整除尾部丢弃）
```

### 参数变化对步数的影响

| 改变的参数 | 变化方向 | 对步数的影响 |
|-----------|---------|------------|
| `num_train_epochs` | ×N | 步数 ×N（线性） |
| `gradient_accumulation_steps` | ×2 | 步数 ÷2（减半） |
| `per_device_train_batch_size` | ×2 | 步数 ÷2（减半，但受显存限制） |
| `cutoff_len` | 减小 | 步数不变，但每步变快（显存降低） |
| `lora_rank` | 增大 | 步数不变，但每步变慢（参数增多） |

---

## 训练时间估算

基准测量（4×V100 16GB，DeepSpeed ZeRO-3，当前配置）：
- 训练单步耗时：**13.2 s/step**
- 验证单次耗时：**612 s/次**（6,000条验证集，batch=4）

### 不同 epochs 配置的耗时对比

| 场景 | epochs | 总步数 | 纯训练时间 | eval次数<br>(eval_steps=500) | eval总耗时 | **总耗时** |
|------|--------|--------|-----------|---------------------------|-----------|----------|
| 快速验证 | 1 | 1,500 | 5.5 h | 3 次 | 0.5 h | **6.0 h** |
| **推荐** | **2** | **3,000** | **11.0 h** | **6 次** | **1.0 h** | **12.0 h** |
| 保守 | 3 | 4,500 | 16.5 h | 9 次 | 1.5 h | **18.0 h** |
| 当前配置 | 5 | 7,500 | 27.5 h | 14 次 | 2.4 h | **29.9 h** |

### eval_steps 对总时间的影响（以 5 epochs / 7,500 步为例）

| eval_steps | eval次数 | eval总耗时 | 节省时间 |
|-----------|---------|-----------|---------|
| 200 | 37 次 | 6.3 h | — |
| **500（当前）** | **14 次** | **2.4 h** | — |
| 1000 | 7 次 | 1.2 h | **节省 1.2 h** |
| 2000 | 3 次 | 0.5 h | **节省 1.9 h** |
| epoch 末（eval_strategy: epoch） | 5 次 | 0.9 h | **节省 1.5 h** |

> **立竿见影的优化**：当前训练完成后，下次训练只需把 `eval_steps: 500` 改为 `eval_steps: 2000`（或 `eval_strategy: epoch`），在不影响任何模型效果的情况下，直接节省约 2 小时。

### cutoff_len 对速度的影响（估算）

| cutoff_len | 每步耗时(估算) | 相对当前 |
|-----------|------------|---------|
| 128 | ~8 s/step | 快 ~40% |
| 256 | ~10 s/step | 快 ~25% |
| **512（当前）** | **13.2 s/step** | 基准 |

---

## 参数对应关系

YAML 配置和 Python 脚本的参数完全对应，下面是完整的对照表：

| 功能 | YAML 参数 | Python 参数 | 默认值 |
|------|-----------|-------------|--------|
| 基座模型 | `model_name_or_path` | `model_name_or_path` | `/workspace/Qwen3-8B` |
| 对话模板 | `template` | `template` | `qwen` |
| 微调类型 | `finetuning_type` | `finetuning_type` | `lora` |
| LoRA 目标 | `lora_target` | `lora_target` | `all` |
| LoRA 秩 | `lora_rank` | `lora_rank` | `16` |
| LoRA Alpha | `lora_alpha` | `lora_alpha` | `32` |
| LoRA Dropout | `lora_dropout` | `lora_dropout` | `0.05` |
| 序列长度 | `cutoff_len` | `max_seq_length` | `512` |
| 输出目录 | `output_dir` | `output_dir` | `/workspace/.../saves/qwen3-8b/lora/sft` |
| 训练轮数 | `num_train_epochs` | `num_train_epochs` | `5.0` |
| 批次大小 | `per_device_train_batch_size` | `per_device_train_batch_size` | `2` |
| 梯度累积 | `gradient_accumulation_steps` | `gradient_accumulation_steps` | `4` |
| 学习率 | `learning_rate` | `learning_rate` | `1e-4` |
| 调度器 | `lr_scheduler_type` | `lr_scheduler_type` | `cosine` |
| 预热比例 | `warmup_ratio` | `warmup_ratio` | `0.1` |
| 权重衰减 | `weight_decay` | `weight_decay` | `0.01` |
| 梯度裁剪 | `max_grad_norm` | `max_grad_norm` | `1.0` |
| 混合精度 | `fp16` | `fp16` | `true` |
| 优化器 | `optim` | `optim` | `adamw_torch` |
| 随机种子 | `seed` | `seed` | `42` |
| 日志间隔 | `logging_steps` | `logging_steps` | `10` |
| 保存间隔 | `save_steps` | `save_steps` | `500` |
| 保存数量 | `save_total_limit` | `save_total_limit` | `3` |
| 评估策略 | `eval_strategy` | `eval_strategy` | `steps` |
| 评估间隔 | `eval_steps` | `eval_steps` | `500` |
| 梯度检查点 | `gradient_checkpointing` | `gradient_checkpointing` | `true` |
| 日志目标 | `report_to` | `report_to` | `tensorboard` |

---

## 完整工作流程

```
准备数据 → 训练 → TensorBoard 监控 → 导出模型 → 推理测试 → 评估
  │          │           │              │            │          │
  ▼          ▼           ▼              ▼            ▼          ▼
data/    方案 A 或 B   port 6006    合并 LoRA     交互/批量   分类报告
```

### 方案 A 完整流程（LlamaFactory）

```bash
# 1. 准备数据（替换 /workspace/lora-intent-clf/data/*.json）
# 2. 一键训练 + 导出
bash /workspace/lora-intent-clf/scripts/run_train_and_export.sh
# 3. TensorBoard
tensorboard --logdir /workspace/lora-intent-clf/saves/qwen3-8b/lora/sft --port 6006
# 4. 推理测试
llamafactory-cli chat /workspace/lora-intent-clf/configs/inference.yaml
```

### 方案 B 完整流程（Python）

```bash
cd /workspace/lora-intent-clf
# 1. 安装依赖
pip install -r requirements.txt
# 2. 准备数据（替换 data/*.json）
# 3. 训练
python src/train.py
# 4. TensorBoard
tensorboard --logdir /workspace/lora-intent-clf/saves/qwen3-8b/lora/sft/logs --port 6006
# 5. 导出
python src/export_model.py
# 6. 评估
python src/evaluate.py --test_file /workspace/lora-intent-clf/data/test.json --output_file /workspace/lora-intent-clf/results/eval.json
# 7. 推理
python src/inference.py --input "我想买一份寿险"
```

---

## 关于执行目录

**方案 A（LlamaFactory CLI）**：YAML 配置中所有路径均为绝对路径（`/workspace/lora-intent-clf/...`），因此可以在**任意目录**下执行 `llamafactory-cli train`，不受执行目录影响。

**方案 B（Python 脚本）**：Python 的 `config.py` 中所有默认路径也是绝对路径，因此同样不受执行目录影响。建议先 `cd /workspace/lora-intent-clf` 再执行，确保相对 import 正常工作。

---

## 发布到 GitHub

项目已初始化 Git 仓库并完成首次提交。使用以下命令创建公开 GitHub 仓库：

```bash
cd /workspace/lora-intent-clf

# 方式一：一键脚本（推荐）
bash scripts/setup_github.sh lora-intent-clf

# 方式二：手动执行
gh repo create lora-intent-clf --public --source=. --push
```

前提条件：已安装 [GitHub CLI](https://cli.github.com/) 并完成 `gh auth login`。

---

## FAQ

**Q: V100 能否使用 BF16？**
A: 不能。V100 只支持 FP16，项目已默认配置为 `fp16: true`。如果你使用 A100/H100，可以改为 `bf16: true`（YAML）或修改 `config.py` 中的 `fp16=False` 并添加 `bf16=True`。

**Q: 训练时显存不足怎么办？**
A: 可以尝试以下方法（按优先级）：减小 `per_device_train_batch_size`（如改为 1）、减小 `cutoff_len`（如改为 256）、减小 `lora_rank`（如改为 8）、确认 `gradient_checkpointing` 已开启。

**Q: 如何使用挂载或本地路径的模型？**
A: 将 `model_name_or_path` 改为本地或挂载的绝对路径即可，如本项目使用的本地路径 `/workspace/Qwen3-8B`。LlamaFactory 和 PEFT 对路径和 Hub ID 的处理逻辑完全一致。

**Q: 数据集需要多少条才够？**
A: 分类任务对数据量要求不高。经验上，每个类别 500~1000 条即可取得不错的效果。建议确保各标签的数据量相对均衡。

**Q: 如何修改 TensorBoard 端口？**
A: `tensorboard --logdir /workspace/lora-intent-clf/saves/qwen3-8b/lora/sft --port 你想要的端口号`。

**Q: 报错 "No such file or directory './dataset_info.json'" 怎么办？**
A: 这是因为 `dataset_dir` 使用了相对路径。确保 `dataset_dir` 使用绝对路径：`/workspace/lora-intent-clf/data`。详见上方「dataset 和 dataset_dir 的关系」章节。
