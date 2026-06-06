<h1 align="center">Context-Budgeted LLM Harness for Robust Classification</h1>

<h3 align="center">围绕冻结大模型的动态检索、任务路由与结构化输出控制</h3>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  LLM harness · Context engineering · Dynamic retrieval · Robust parsing · OOD evaluation
</p>

---

本项目实现了一个面向分类任务的轻量级 LLM Harness。系统接收带标签样本流，在外部维护记忆；预测时根据输入动态检索示例、区分普通意图分类与自然语言选择题、控制 prompt token 预算，并通过结构化解析返回可 exact-match 的标签。

项目关注的问题很具体：当底层大模型保持冻结时，外部控制层还能在多大程度上提升推理可靠性？这里的控制层包括检索策略、上下文预算管理、结构化输出约束、确定性解析与扩展鲁棒性评测。

## 项目动机

很多 LLM 应用的可靠性并不只取决于一次模型调用，而取决于模型外围的 Harness：上下文如何选择，prompt 如何组织，输出如何验证，异常如何兜底。这个项目把这一层单独抽出来，在一个小而完整的任务设定中做工程实现和评估。

该设定具备几个明确约束：

- 模型只通过 OpenAI-compatible chat API 调用；
- 不更新模型权重；
- 每次 prompt 必须处在固定 token 预算内；
- 最终输出必须严格匹配已观测标签空间中的某个标签；
- 同一套实现需要同时处理短文本意图分类、分布外标签空间和自然语言选择题。

因此，这不是一个简单 prompt demo，也不是训练分类器，而是一个围绕冻结 LLM 的推理时控制系统。

## 系统设计

核心实现位于 `solution.py` 中的 `MyHarness`。

1. **外部记忆**
   系统通过 `update(text, label)` 接收训练样本，并将其存入轻量级 memory bank。

2. **混合特征动态检索**
   对每个测试输入，系统计算它与历史样本的相似度。特征同时包含 word-level token 和 character 3-gram，使检索对短文本、缩写和噪声表述更稳定。

3. **运行时任务路由**
   系统根据当前标签空间判断任务形态。若标签呈现短选项形式，则切换到选择题 prompt；否则使用意图分类 prompt。这样同一套代码可以覆盖不同任务格式。

4. **受预算约束的 prompt 组装**
   检索到的样本只会在 token 预算允许的情况下进入 few-shot prompt。若超出预算，系统优先移除低相关示例，避免评测运行时发生不可控截断。

5. **结构化输出与容错解析**
   模型被要求输出包含最终 `<label>` 的紧凑 XML-like 格式。解析阶段依次使用精确匹配、清洗匹配、子串匹配和最近邻标签兜底，从而降低模型输出格式漂移带来的 exact-match 失败。

## 实验结果

在本地 DEV split 上，使用 Qwen3-8B 非思考模式时，该 Harness 约达到 83% accuracy。扩展 benchmark 进一步覆盖同分布分类、OOD 标签空间、自然语言选择题、双语输入和部分垂直领域任务。

![Extended benchmark results](assets/benchmark-results.png)

主要观察包括：

- word + 3-gram 混合检索比纯词面匹配更适合短文本和噪声输入；
- prompt 预算控制可以降低长样本导致关键指令被截断的风险；
- 结构化输出约束能减少大模型冗长回答造成的 exact-match 解析失败；
- 某个中文 MCQ 子集的异常低分更接近数据质量问题，而不是一个干净的推理能力评测结论。

更详细的实验记录和误差分析见 [docs/exploration_report.md](docs/exploration_report.md)。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

配置 OpenAI-compatible API。不要提交真实密钥。

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

运行本地评测：

```bash
python run.py --runs 1
```

运行扩展 benchmark：

```bash
python scripts/benchmark.py
```

## 仓库结构

```text
.
├── solution.py                 # 核心 Harness 实现
├── harness_base.py             # 评测使用的最小 Harness 接口
├── run.py                      # 本地评测入口
├── llm_client.py               # 通过环境变量配置的 OpenAI-compatible 客户端
├── requirements.txt
├── data/                       # 本地 DEV split
├── tokenizer/                  # 用于 prompt token 预算统计的 tokenizer
├── mock-data/                  # 扩展 benchmark 数据
├── scripts/
│   └── benchmark.py            # 扩展 benchmark 运行脚本
├── docs/
│   ├── assignment-spec-2026-summer.pdf
│   └── exploration_report.md
└── assets/
    ├── benchmark-results.png
    └── benchmark-results-repeat.png
```

## 范围与来源说明

这个项目起源于一个受限 Harness Engineering 评测场景。仓库保留了原有根目录评测入口和本地 DEV 资产，目的是让实现可以直接运行，并与原评测环境保持可比性。

本项目的主要工作集中在：

- `solution.py`：检索、任务路由、prompt 构造、输出解析和兜底逻辑；
- `llm_client.py`：面向公开仓库的环境变量配置方式；
- `scripts/benchmark.py`：基于扩展数据集的鲁棒性评测脚本；
- `docs/` 与 README：实验整理、方法说明和误差分析。

这里不回避项目的评测来源，但也不把来源当作项目主叙事。更准确的定位是：这是一个小规模、可复现的 LLM Harness 工程项目，用于展示上下文管理、推理时控制、评测组织和失败案例分析能力。

## 安全说明

公开仓库不应包含真实 API Key、个人申请材料或私有通信内容。当前客户端通过环境变量读取凭据，`.gitignore` 也排除了 `.env` 和本地私有文档。如果真实密钥曾经进入 Git 历史，发布前应先轮换密钥。
