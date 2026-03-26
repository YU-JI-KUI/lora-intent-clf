# LoRA Intent Classification（意图识别微调项目）

基于 LoRA（Low-Rank Adaptation）对大语言模型进行 SFT 微调，用于意图识别分类任务。

## 项目特点

- **双模式训练**：支持 LlamaFactory CLI（YAML 配置）和纯 Python 脚本两种方式，自由选择
- **模块解耦**：训练、导出、推理、评估各自独立，互不依赖
- **配置驱动**：所有参数通过配置文件管理，零硬编码
- **可扩展**：标签列表可配置，轻松从二分类扩展到多分类
- **可替换模型**：不绑定特定模型，支持 Qwen、Llama、Baichuan、ChatGLM 等

## 目录结构

```
lora-intent-clf/
├── README.md                          # 本文档
├── pyproject.toml                     # uv 项目配置 & Python 依赖
├── .python-version                    # Python 版本锁定（3.10）
├── .gitignore
│
├── configs/                           # LlamaFactory YAML 配置
│   ├── train_lora_sft.yaml           #   训练配置（带详细注释）
│   ├── export_lora.yaml              #   LoRA 适配器导出/合并配置
│   ├── inference.yaml                #   推理配置（合并模型）
│   └── inference_lora.yaml           #   推理配置（基座 + LoRA 适配器）
│
├── data/                              # 数据目录
│   ├── dataset_info.json             #   LlamaFactory 数据集注册文件
│   ├── train.json                    #   训练集（示例数据）
│   ├── val.json                      #   验证集（示例数据）
│   └── test.json                     #   测试集（示例数据）
│
├── scripts/                           # Shell 脚本
│   ├── run_train_and_export.sh       #   一键训练 + 导出
│   └── run_inference.sh              #   推理测试
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

## 快速开始

### 0. 前提条件

确保以下组件已安装：

```bash
# 检查 LlamaFactory CLI（v0.9.0）
llamafactory-cli version

# 检查 GPU
nvidia-smi

# 检查 uv（Python 脚本方案需要）
uv --version

# 检查 TensorBoard
tensorboard --version
```

### 1. 准备数据

替换 `data/` 目录下的示例数据为你的实际训练数据。数据格式为 JSON 数组，每条数据包含三个字段：

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

### 2. 方案选择

本项目提供两种完全等价的训练方案：

| 特性 | 方案 A：LlamaFactory CLI | 方案 B：Python 脚本 |
|------|--------------------------|---------------------|
| 配置方式 | YAML 文件 | Python dataclass + JSON |
| 启动方式 | `llamafactory-cli train` | `uv run python src/train.py` |
| 依赖 | LlamaFactory v0.9.0 | transformers + peft |
| 适用场景 | 快速实验、参数调优 | 需要定制化训练流程 |
| 多 GPU | 自动支持 | 需配合 accelerate/torchrun |

---

## 方案 A：使用 LlamaFactory CLI（推荐快速实验）

### A1. 训练

```bash
# 直接使用 llamafactory-cli 训练
llamafactory-cli train configs/train_lora_sft.yaml
```

### A2. 查看训练日志（TensorBoard）

```bash
# 启动 TensorBoard（训练过程中或训练结束后均可）
tensorboard --logdir saves/qwen3-8b/lora/sft --port 6006

# 浏览器访问 http://localhost:6006
```

TensorBoard 中可以查看：训练/验证 loss 曲线、学习率变化、训练步数等。

### A3. 导出合并模型

```bash
# 将 LoRA 适配器合并到基座模型
llamafactory-cli export configs/export_lora.yaml
```

导出后的完整模型保存在 `models/qwen3-8b-intent-clf/`，可以像普通 HuggingFace 模型一样加载。

### A4. 推理测试

```bash
# 使用合并模型进行交互式推理
llamafactory-cli chat configs/inference.yaml

# 或者使用基座模型 + LoRA 适配器推理（无需先导出）
llamafactory-cli chat configs/inference_lora.yaml
```

### A5. 一键训练 + 导出

```bash
# 一键执行完整流程：训练 → 导出
bash scripts/run_train_and_export.sh

# 可选参数：
bash scripts/run_train_and_export.sh --skip-train   # 仅导出
bash scripts/run_train_and_export.sh --skip-export   # 仅训练
```

---

## 方案 B：使用 Python 脚本

### B0. 安装 Python 依赖

```bash
# 使用 uv 安装依赖（Python 3.10）
uv sync
```

### B1. 训练

```bash
# 使用默认配置训练
uv run python src/train.py

# 使用自定义配置文件
uv run python src/train.py --config my_config.json

# 命令行覆盖参数
uv run python src/train.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --lora_rank 32 \
    --learning_rate 2e-4 \
    --num_train_epochs 10
```

可用的命令行参数（均有默认值，可按需覆盖）：

| 参数 | 默认值 | 对应 YAML |
|------|--------|-----------|
| `--model_name_or_path` | `Qwen/Qwen3-8B` | `model_name_or_path` |
| `--template` | `qwen` | `template` |
| `--lora_rank` | `16` | `lora_rank` |
| `--lora_alpha` | `32` | `lora_alpha` |
| `--lora_dropout` | `0.05` | `lora_dropout` |
| `--lora_target` | `all` | `lora_target` |
| `--train_file` | `data/train.json` | `dataset` (in dataset_info) |
| `--val_file` | `data/val.json` | `val_size` (auto split) |
| `--max_seq_length` | `512` | `cutoff_len` |
| `--output_dir` | `saves/qwen3-8b/lora/sft` | `output_dir` |
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
tensorboard --logdir saves/qwen3-8b/lora/sft/logs --port 6006
```

### B3. 导出合并模型

```bash
# 使用默认配置导出
uv run python src/export_model.py

# 指定适配器路径和导出目录
uv run python src/export_model.py \
    --adapter_path saves/qwen3-8b/lora/sft \
    --export_dir models/qwen3-8b-intent-clf
```

### B4. 推理

```bash
# 交互式推理（使用合并模型）
uv run python src/inference.py

# 单条推理
uv run python src/inference.py --input "我想买寿险"

# 批量推理
uv run python src/inference.py \
    --input_file data/test.json \
    --output_file results/predictions.json

# 使用 LoRA 适配器推理（无需先导出）
uv run python src/inference.py \
    --adapter_path saves/qwen3-8b/lora/sft \
    --input "我想买寿险"
```

### B5. 评估

```bash
# 在测试集上评估
uv run python src/evaluate.py

# 指定测试集和输出文件
uv run python src/evaluate.py \
    --test_file data/test.json \
    --output_file results/eval_report.json

# 使用 LoRA 适配器评估
uv run python src/evaluate.py \
    --adapter_path saves/qwen3-8b/lora/sft
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
# 方式一：torchrun（推荐）
uv run torchrun --nproc_per_node=4 src/train.py

# 方式二：accelerate
uv run accelerate launch --num_processes=4 src/train.py
```

---

## 自定义配置

### 使用 JSON 配置文件（Python 方案）

可以将所有参数保存为 JSON 文件：

```bash
# 生成默认配置模板
uv run python -c "
from src.config import ProjectConfig
cfg = ProjectConfig()
cfg.save('my_config.json')
print('配置已保存到 my_config.json')
"

# 编辑 my_config.json 后使用
uv run python src/train.py --config my_config.json
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
uv run python src/train.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --template llama3
```

常见模型和模板对照：

| 模型 | template |
|------|----------|
| Qwen/Qwen3-8B | qwen |
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

## 参数对应关系

YAML 配置和 Python 脚本的参数完全对应，下面是完整的对照表：

| 功能 | YAML 参数 | Python 参数 | 默认值 |
|------|-----------|-------------|--------|
| 基座模型 | `model_name_or_path` | `model_name_or_path` | `Qwen/Qwen3-8B` |
| 对话模板 | `template` | `template` | `qwen` |
| 微调类型 | `finetuning_type` | `finetuning_type` | `lora` |
| LoRA 目标 | `lora_target` | `lora_target` | `all` |
| LoRA 秩 | `lora_rank` | `lora_rank` | `16` |
| LoRA Alpha | `lora_alpha` | `lora_alpha` | `32` |
| LoRA Dropout | `lora_dropout` | `lora_dropout` | `0.05` |
| 序列长度 | `cutoff_len` | `max_seq_length` | `512` |
| 输出目录 | `output_dir` | `output_dir` | `saves/qwen3-8b/lora/sft` |
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
# 1. 准备数据（替换 data/*.json）
# 2. 一键训练 + 导出
bash scripts/run_train_and_export.sh
# 3. TensorBoard
tensorboard --logdir saves/qwen3-8b/lora/sft --port 6006
# 4. 推理测试
llamafactory-cli chat configs/inference.yaml
```

### 方案 B 完整流程（Python）

```bash
# 1. 安装依赖
uv sync
# 2. 准备数据（替换 data/*.json）
# 3. 训练
uv run python src/train.py
# 4. TensorBoard
tensorboard --logdir saves/qwen3-8b/lora/sft/logs --port 6006
# 5. 导出
uv run python src/export_model.py
# 6. 评估
uv run python src/evaluate.py --test_file data/test.json --output_file results/eval.json
# 7. 推理
uv run python src/inference.py --input "我想买一份寿险"
```

---

## FAQ

**Q: V100 能否使用 BF16？**
A: 不能。V100 只支持 FP16，项目已默认配置为 `fp16: true`。如果你使用 A100/H100，可以改为 `bf16: true`（YAML）或修改 `config.py` 中的 `fp16=False` 并添加 `bf16=True`。

**Q: 训练时显存不足怎么办？**
A: 可以尝试以下方法（按优先级）：减小 `per_device_train_batch_size`（如改为 1）、减小 `cutoff_len`（如改为 256）、减小 `lora_rank`（如改为 8）、确认 `gradient_checkpointing` 已开启。

**Q: 如何使用已下载的本地模型？**
A: 将 `model_name_or_path` 改为本地模型的绝对路径，如 `/data/models/Qwen3-8B`。

**Q: 数据集需要多少条才够？**
A: 分类任务对数据量要求不高。经验上，每个类别 500~1000 条即可取得不错的效果。建议确保各标签的数据量相对均衡。

**Q: 如何修改 TensorBoard 端口？**
A: `tensorboard --logdir saves/qwen3-8b/lora/sft --port 你想要的端口号`。
