<h1 align="center">Context-Budgeted LLM Harness for Robust Classification</h1>

<h3 align="center">面向分类任务的动态检索、上下文预算控制、任务路由与结构化解析</h3>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  LLM Harness | Context Engineering | Dynamic Retrieval | Structured Output | Robust Evaluation
</p>

---

本仓库将分类任务视作一个小型的 LLM 推理时适配问题。系统通过 OpenAI-compatible API 调用冻结大模型，在模型调用外层完成样本记忆、动态检索、prompt 预算控制、任务路由和结构化输出解析，使 `predict(text)` 能稳定返回已知标签空间中的精确匹配标签。

项目按一个轻量级 AI 系统实验来组织：可控变量包括检索策略、prompt 预算管理、任务表述和输出校验；评测部分除了本地准确率，也关注分布外标签、选择题格式、双语输入和领域迁移下的表现。

## 技术定位

这个设计围绕三个具体问题展开：

- 在不更新模型权重的前提下，外部记忆和动态检索能否提供有效的任务适配；
- 将 prompt token 预算作为显式约束后，输出是否更稳定、评测是否更可复现；
- 如何约束并解析生成式回答，使聊天模型满足 exact-match 分类评测。

这些组件共同构成了一个紧凑但完整的流程，覆盖方法设计、工程实现、评测复现和误差分析。

## 工程亮点

- **外部样本记忆**：通过 `update(text, label)` 接收带标签样本，并维护当前可用的标签空间。
- **混合特征检索**：结合 word-level token 与 character 3-gram，为短文本、噪声表述和中英文混合输入检索更相关的 few-shot 示例。
- **运行时任务路由**：根据标签形态区分普通意图分类和选择题任务，复用同一套 Harness 处理不同输入格式。
- **prompt 预算控制**：检索示例只在 token 预算允许时加入 prompt，避免长样本挤占关键指令或触发运行时截断。
- **结构化输出解析**：要求模型输出紧凑的 XML-like 格式，并用精确匹配、清洗匹配、子串匹配和最近邻兜底保证返回合法标签。
- **可复现实验脚本**：根目录评测入口覆盖本地 DEV split，`scripts/benchmark.py` 进一步运行扩展鲁棒性评测。

## 核心流程

核心实现位于 `solution.py` 中的 `MyHarness`。

```text
update(text, label)
    -> 存储样本
    -> 更新标签统计

predict(text)
    -> 检查标签空间
    -> 检索相似示例
    -> 组装受 token 预算约束的 prompt
    -> 调用配置好的 LLM
    -> 解析并校验返回标签
```

这种结构把模型后端和 Harness 控制逻辑分开，便于替换模型、复现实验和定位错误。

## 实验结果

在 Qwen3-8B 非思考模式下，该 Harness 在本地 DEV split 上达到约 **83% accuracy**。扩展 benchmark 覆盖同分布分类、分布外标签空间、选择题任务、双语输入和部分垂直领域数据。

![Extended benchmark results](assets/benchmark-results.png)

实验中的主要观察包括：

- word + 3-gram 混合检索在短文本和噪声输入上比纯词面匹配更稳定；
- 显式 prompt 预算控制可以减少长示例挤占关键指令带来的失败；
- 结构化响应和确定性解析能降低冗长输出造成的 exact-match 错误；
- 某个中文 MCQ 子集的异常低分更适合作为数据质量压力测试来看待。

更详细的实验记录见 [docs/exploration_report.md](docs/exploration_report.md)。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

配置 OpenAI-compatible API。项目从环境变量读取凭据，请不要提交真实密钥。

PowerShell：

```powershell
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_API_KEY="your-api-key"
$env:LLM_MODEL="qwen3-8b"
$env:LLM_ENABLE_THINKING="false"
```

Bash：

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
|-- solution.py          # 核心 Harness 实现
|-- harness_base.py      # Harness 实现的基础接口
|-- run.py               # 本地评测入口
|-- llm_client.py        # 通过环境变量配置的 OpenAI-compatible 客户端
|-- requirements.txt
|-- data/                # 本地开发集
|-- tokenizer/           # 用于 prompt token 预算统计的 tokenizer
|-- mock-data/           # 扩展 benchmark 数据
|-- scripts/
|   `-- benchmark.py     # 扩展 benchmark 运行脚本
|-- docs/
|   `-- exploration_report.md
`-- assets/
    |-- benchmark-results.png
    `-- benchmark-results-repeat.png
```

## 限制

这是一个聚焦分类和选择题场景的推理 Harness，不是通用 Agent 框架。实际效果仍依赖底层模型的指令跟随能力；当前选择题路由使用的是较简单的标签形态启发式规则；项目也没有引入训练好的 embedding 模型或任务专用分类器，而是把适配能力集中在检索、prompt 组织和解析逻辑上。

## 安全说明

API Key 应通过环境变量配置。仓库默认忽略 `.env` 文件，`.env.example` 中列出了可用的配置项名称。
