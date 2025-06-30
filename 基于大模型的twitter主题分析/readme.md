# 基于大模型的twitter主题分析

这是一个基于 DeepSeek 大语言模型的 Twitter 主题分析工具，能够自动识别推文中的主题分布，并提供可视化分析和语义解释。通过结合传统主题建模技术和先进的大语言模型，该工具提供深入的主题洞察力。

## 功能亮点

- **自动化文本预处理**：智能清洗推文数据，去除噪音和非必要元素
- **LDA 主题建模**：使用潜在狄利克雷分配算法识别推文中的主要主题
- **多维度可视化**：
  - 交互式主题模型可视化
  - 主题词云展示
  - 文档-主题分布热力图
- **语义分析**：利用 DeepSeek 大模型解释主题含义
- **错误处理**：增强的 API 错误处理机制

## 安装指南

### 前置要求

- Python 3.7+
- pip (最新版)
- DeepSeek API 密钥

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/YukunDai/2022290220-DaiYukun.git
cd 基于大模型的twitter主题分析
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 安装 NLTK 资源：
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

5. 配置环境变量：
创建 `.env` 文件并添加：
```env
OPENAI_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1
```

## 使用指南

### 准备数据

在 `main.py` 文件中添加推文数据：
```python
tweets = [
    "Just had the best coffee in Seattle! #coffee #morning",
    "AI is transforming the world at an unprecedented pace. #AI #tech",
    # 添加更多推文...
]
```

或从文件加载推文：
```python
with open('tweets.txt', 'r', encoding='utf-8') as f:
    tweets = [line.strip() for line in f.readlines()]
```

### 运行分析

```bash
python main.py
```

### 命令行选项

| 选项               | 描述                | 默认值   |
| ------------------ | ------------------- | -------- |
| `--num_topics NUM` | 设置主题数量        | 3        |
| `--passes NUM`     | 设置LDA训练迭代次数 | 10       |
| `--output_dir DIR` | 设置输出目录        | 当前目录 |

示例：
```bash
python main.py --num_topics 5 --passes 15
```

## 输出示例

### 控制台输出
```
[DEBUG] 原始推文条数: 10
[DEBUG] 文档 0 预处理后 tokens: ['best', 'coffee', 'seattle', 'coffee', 'morning']
...
...
[DEBUG] 预处理后文档示例: [['best', 'coffee', 'seattle', 'coffee', 'morning'], ['transforming', 'world', 'unprecedented', 'pace', 'tech']]
[DEBUG] 原始词典大小: 54
[DEBUG] 过滤后词典大小: 54
[DEBUG] 示例 Bow (文档0): [(0, 1), (1, 2), (2, 1), (3, 1)]
[DEBUG] LDA 训练完成，主题数: 3
[DEBUG] 主题 0 关键词: 0.069*"vote" + 0.039*"remember" + 0.039*"politics" + 0.039*"election" + 0.039*"coming" + 0.039*"pace" + 0.039*"unprecedented" + 0.039*"world" + 0.039*"tech" + 0.039*"transforming"
...
...
[DEBUG] 已保存可视化到: lda_vis.html
[DEBUG] 调用 DeepSeek，prompt: 请根据关键词['vote', 'remember', 'politics', 'election', 'coming']描述主题0核心内容
[DEBUG] DeepSeek 返回: 基于关键词“vote”（投票）、“remember”（记住）、“politics”（政治）、“election”（选举）、“coming”（即将到来），**主题0的核心内容**可以描述为：

> **呼吁公民在即将到来的选举中积极参与投票，并提醒他们牢记政治参与的重要性及其对未来产生的影响。**

**具体解释：**

1.  **`election` 和 `coming`：** 明确点出背景是**一场即将举行的选举**。这是核心事件。
2.  **`vote`：** 这是行动号召的核心。主题的核心是**鼓励或强调投票行为**本身。
3.  **`remember`：** 这是一个关键的提醒或警示。它暗示选民需要**记住某些重要的事情**，这可能是：
    *   投票的权利和责任（来之不易）。
    *   过去的政治事件、政策后果或候选人的承诺/记录。
    *   特定议题（如经济、社会公平、外交政策）对自己生活的重要性。
    *   忽视政治或不投票可能带来的负面结果。
4.  **`politics`：** 这设定了整个讨论的**背景和领域**。主题讨论的不是一般的决定，而是与**政治权力、治理、政策和公共事务**直接相关的决策（即选举投票）。

**总结核心信息：**

*   **事件：** 一场选举即将发生。
*   **核心行动：** 投票 (`vote`)。
*   **关键心态：** 提醒 (`remember`) 选民要重视并负责任地行使投票权。
*   **领域：** 在政治 (`politics`) 领域内，投票结果将决定权力分配和未来政策方向。

**因此，主题0的核心是关于在临近的政治选举中，通过强调“记住”（责任、历史、影响）来激发和动员选民参与投票。** 它超越了简单的“去投票”号召，加入了“深思熟虑”、“承担责任”和“理解后果”的维度。
主题 0 内容分析（DeepSeek）: 基于关键词“vote”（投票）、“remember”（记住）、“politics”（政治）、“election”（选举）、“coming”（即将到来），**主题0的核心内容**可以描述为：

> **呼吁公民在即将到来的选举中积极参与投票，并提醒他们牢记政治参与的重要性及其对未来产生的影响。**

**具体解释：**

1.  **`election` 和 `coming`：** 明确点出背景是**一场即将举行的选举**。这是核心事件。
2.  **`vote`：** 这是行动号召的核心。主题的核心是**鼓励或强调投票行为**本身。
3.  **`remember`：** 这是一个关键的提醒或警示。它暗示选民需要**记住某些重要的事情**，这可能是：
    *   投票的权利和责任（来之不易）。
    *   过去的政治事件、政策后果或候选人的承诺/记录。
    *   特定议题（如经济、社会公平、外交政策）对自己生活的重要性。
    *   忽视政治或不投票可能带来的负面结果。
4.  **`politics`：** 这设定了整个讨论的**背景和领域**。主题讨论的不是一般的决定，而是与**政治权力、治理、政策和公共事务**直接相关的决策（即选举投票）。

**总结核心信息：**

*   **事件：** 一场选举即将发生。
*   **核心行动：** 投票 (`vote`)。
*   **关键心态：** 提醒 (`remember`) 选民要重视并负责任地行使投票权。
*   **领域：** 在政治 (`politics`) 领域内，投票结果将决定权力分配和未来政策方向。

**因此，主题0的核心是关于在临近的政治选举中，通过强调“记住”（责任、历史、影响）来激发和动员选民参与投票。** 它超越了简单的“去投票”号召，加入了“深思熟虑”、“承担责任”和“理解后果”的维度。
...
...
```

## 自定义配置

### 调整主题建模参数
```python
# 在 main.py 中修改
num_topics = 5  # 主题数量
passes = 15     # 训练迭代次数
alpha = 'auto'  # 主题分布超参数
```

### 修改预处理流程
```python
def preprocess(texts):
    # 添加自定义停用词
    custom_stopwords = ['http', 'https', 'com']
    stop_words = set(stopwords.words('english')).union(custom_stopwords)
    
    # 修改词形还原规则
    lemmatizer = WordNetLemmatizer()
    processed = []
    for doc in texts:
        # 自定义清洗规则
        doc_clean = re.sub('[^a-zA-Z]', ' ', doc).lower()
        tokens = [t for t in doc_clean.split() if t not in stop_words and len(t) > 2]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        processed.append(tokens)
    return processed
```

### 使用不同的大模型
```python
def get_completion(prompt, model="deepseek-reasoner"):
    # 替换为其他支持的模型
    # 例如: "deepseek-chat", "deepseek-coder"
    ...
```

