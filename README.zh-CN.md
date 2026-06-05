# Lightweight LLM Harness for Robust Few-Shot Classification

[English](README.md) | [中文](README.zh-CN.md)

本项目实现了一个面向受限上下文窗口的轻量级 LLM Harness，用于研究冻结大模型在少样本文本分类、分布外任务和自然语言选择题中的测试时适应能力。项目最初基于 2026 创智学院 Harness Engineering 考核框架完成，核心实现集中在 `solution.py` 中；本仓库在保留官方评测接口的基础上，补充了更清晰的工程结构、公开配置方式、扩展 benchmark 与实验报告。

## 项目动机

在许多 LLM 应用中，模型权重并不会在每个新任务上重新训练，系统能力主要来自外部 Harness：如何组织 prompt、检索记忆、控制上下文、解析输出、处理异常与注入攻击。本项目关注一个较小但完整的问题：

> 在单次 prompt 不超过 2048 token 的限制下，如何让冻结的 Qwen3-8B 模型从训练流中的少量带标签样本中完成高鲁棒性的 exact-match 分类？

这个问题与 Test-Time Adaptation、LLM 鲁棒性评测、Agent/Harness Engineering 等方向直接相关：模型本身保持冻结，所有任务适应都发生在外部记忆与确定性控制逻辑中。

## 核心设计

`MyHarness` 由四个部分构成：

1. **混合特征动态检索**：使用 word-level token 与 character 3-gram 的多重集合 Jaccard 相似度，从历史样本中选择最相关的 few-shot 示例。
2. **标签空间任务路由**：根据训练标签的形态自动区分普通意图分类与选择题任务，并切换对应 prompt。
3. **结构化推理输出**：约束模型输出 `<thought>` 与 `<label>`，再用确定性解析逻辑提取最终 label。
4. **预算控制与容错兜底**：在调用 LLM 前主动检查 token 数，必要时减少示例数量；当模型输出不规范时，使用 exact match、清洗匹配、子串匹配与最近邻标签兜底。

该设计刻意保持轻量，不依赖 sklearn、torch 或外部索引库，符合原评测对 `solution.py` 的导入限制，也便于读者审计每一步工程决策。

## 仓库结构

```text
.
├── solution.py                 # 主要贡献：MyHarness 的检索、路由、prompt 与解析逻辑
├── harness_base.py             # 官方 Harness 基类接口
├── run.py                      # 官方本地评测入口
├── llm_client.py               # OpenAI-compatible LLM 客户端，使用环境变量配置
├── requirements.txt
├── data/                       # 官方 DEV 数据：train_dev / test_dev
├── tokenizer/                  # 本地 tokenizer，用于精确 token 计数
├── mock-data/                  # 社区/自造扩展评测数据
├── scripts/
│   └── benchmark.py            # 扩展 benchmark 运行脚本
├── docs/
│   ├── assignment-spec-2026-summer.pdf
│   └── exploration_report.md
└── assets/
    ├── benchmark-results.png
    └── benchmark-results-repeat.png
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

配置 OpenAI-compatible API。推荐在本地复制 `.env.example` 后自行加载环境变量，不要提交真实密钥。

PowerShell 示例：

```powershell
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_API_KEY="your-api-key"
$env:LLM_MODEL="qwen3-8b"
$env:LLM_ENABLE_THINKING="false"
```

Bash 示例：

```bash
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="qwen3-8b"
export LLM_ENABLE_THINKING="false"
```

运行官方 DEV 评测：

```bash
python run.py --runs 1
```

运行扩展 benchmark：

```bash
python scripts/benchmark.py
```

## 实验结果

官方 DEV 集包含 231 条训练样本、539 条测试样本和 77 个标签。在 Qwen3-8B 非思考模式下，本方案在本地 DEV split 上约达到 83% accuracy。扩展 benchmark 覆盖同分布分类、OOD 标签空间、自然语言选择题、中文/双语输入和部分垂直领域任务。

![Extended benchmark results](assets/benchmark-results.png)

更详细的方法说明、消融观察和异常数据分析见 [docs/exploration_report.md](docs/exploration_report.md)。

## 来源与贡献边界

为保证可复现性，本仓库保留了原评测的核心文件布局。这样做不是为了弱化项目独立性，而是为了让评测契约、实现边界和实验入口足够清楚。

- 官方/考核框架提供：`harness_base.py`、`run.py`、`data/`、`tokenizer/` 以及原始任务说明。
- 本项目主要实现：`solution.py` 中的 Harness 策略、`llm_client.py` 的公开配置整理、`scripts/benchmark.py` 扩展评测脚本、实验报告与仓库文档。
- 扩展数据：`mock-data/` 来自公开社区数据、考核模拟数据和部分自造数据，具体来源见 [mock-data/SII-26Summer-HE-Data-main/README.md](mock-data/SII-26Summer-HE-Data-main/README.md)。

## 工程取舍

本项目没有重构成复杂 Python package，原因是官方评测环境以根目录下的 `solution.py` / `run.py` 为契约。对于展示项目而言，保留这一结构能让读者直接复现实验，也能清楚地区分官方 scaffold 与个人实现。整理重点放在配置安全、文档可读性、实验脚本和结果归档上，而不是为了形式上的“工程化”牺牲可验证性。

## 安全与隐私

公开仓库中不应包含个人申请材料、真实 API Key 或私有通信内容。本项目通过环境变量读取 API 配置，并用 `.gitignore` 排除本地 `.env` 与个人申请材料。如果真实 API Key 曾经出现在 Git 历史中，建议在推送公开仓库前撤销该密钥并清理提交历史。
