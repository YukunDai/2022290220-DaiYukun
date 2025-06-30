# 基于大模型的情感语义分析

这个Python脚本使用本地大语言模型（通过Ollama）对新闻文章进行真假判别和情感分析，包含两种分析方法：纯真假判别和情感增强判别。

## 功能概述

- **纯真假判别**：通过三步分析新闻可信度
  1. 提取核心事实与实体
  2. 评估新闻可信度
  3. 做出真假判断
  
- **情感增强判别**：额外分析情感因素
  1. 识别情感词及其极性
  2. 判断整体情感倾向
  3. 结合情感分析进行真假判断

- **性能评估**：计算两种方法的准确率并进行对比

## 快速开始

### 前提条件
1. 安装 [Ollama](https://ollama.com/)
2. 下载所需模型（如 deepseek-r1:8B）：
   ```bash
   ollama pull deepseek-r1:8B
   ```
3. 安装Python依赖：
   ```bash
   pip install ollama csv-logging
   ```

### 使用示例

1. **使用内置示例新闻**：
   ```bash
   python main.py
   ```

2. **使用完整数据集**（需先下载[Kaggle News Detection数据集](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)）：
   ```bash
   python main.py --use_dataset
   ```

## 数据集准备

1. 从Kaggle下载数据集：[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. 将CSV文件重命名为`fake_and_real_news.csv`
3. 放置在项目目录中

## 配置选项

在脚本中修改以下常量：

```python
# 使用的大语言模型
MODEL_NAME = 'deepseek-r1:8B'

# 数据集路径
DATA_PATH = 'fake_and_real_news.csv'

# API重试设置
RETRIES = 2          # API调用失败重试次数
BACKOFF = 2.0        # 指数退避基数
```

## 输出示例

```log
[2025-06-28 17:36:35] INFO 加载数据集，共 10 条记录
[2025-06-28 17:36:57] INFO HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
...
...
[2025-06-28 17:47:20] INFO 开始情感增强判别，文本: 
[2025-06-28 17:47:49] INFO HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
[2025-06-28 17:47:49] INFO 提取到的情感词: [{'term': '成功', 'polarity': '+'}, {'term': '失望', 'polarity': '-'}, {'term': '快乐', 'polarity': '+'}, {'term': '激动', 'polarity': '悲伤', 'polarity': '-'}, {'term': '恐惧', 'polarity': '-'}, {'term': '安心', 'polarity': '+'}]
[2025-06-28 17:48:06] INFO HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
[2025-06-28 17:48:06] INFO 整体情感分析结果: {'sentiment': '负面', 'rationale': '由于负面情感词如失望、悲伤、恐惧较多，对整体情感倾向有较大影响。'}
[2025-06-28 17:48:24] INFO HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
[2025-06-28 17:48:24] INFO 情感增强解析后结果: {'emotion_impact': '负面情感词较多，可能导致可信度降低', 'label': 'Real', 'explanation': '由于明确的情感倾向（负面），判断结果为真实'}
...
...
[2025-06-28 17:57:50] INFO 开始情感增强判别，文本: 
[2025-06-28 17:58:10] INFO HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
[2025-06-28 17:58:10] INFO 提取到的情感词: [{'term': '积极', 'polarity': '+'}, {'term': '消极', 'polarity': '-'}, {'term': '正面', 'polarity': '+'}, {'term': '负面', 'polarity': '-'}]
[2025-06-28 17:58:35] INFO HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
[2025-06-28 17:58:35] INFO 整体情感分析结果: {'sentiment': '正面', 'rationale': '包含积极和正面情感词，整体倾向为正面'}
[2025-06-28 17:58:57] INFO HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
[2025-06-28 17:58:57] INFO 情感增强解析后结果: {'emotion_impact': 'Positive sentiment in the article supports its credibility.', 'label': 'Real', 'explanation': 'The positive sin the article aligns with the reasoning provided, indicating the news is likely genuine.'}
[2025-06-28 17:58:57] INFO 纯真假判别 完成，预测分布 Fake=3, Real=7
[2025-06-28 17:58:57] INFO 情感增强判别 完成，预测分布 Fake=1, Real=9
[2025-06-28 17:58:57] INFO 纯真假判别 - Acc Overall: 0.5000, Fake Acc: 0.2500 (n=4), Real Acc: 0.6667 (n=6)
[2025-06-28 17:58:57] INFO 情感增强判别 - Acc Overall: 0.7000, Fake Acc: 0.2500 (n=4), Real Acc: 1.0000 (n=6)
[2025-06-28 17:58:57] INFO 准确率提升 - ΔOverall: 0.2000, ΔFakeAcc: 0.0000, ΔRealAcc: 0.3333
```

## 方法说明

### 纯真假判别流程
1. **事实提取**：识别新闻中的核心实体（人物、地点、时间、事件）
2. **可信度验证**：评估新闻可信度（高/中/低）
3. **真假判定**：基于以上分析给出最终标签（Fake/Real）

### 情感增强判别流程
1. **情感词提取**：识别文本中的情感词并标注极性（正面+/负面-/中立0）
2. **整体情感分析**：判断新闻整体情感倾向（正面/中立/负面）
3. **综合判定**：结合内容和情感分析给出最终判断

## 性能指标

脚本计算并输出以下评估指标：
- **整体准确率**：所有样本中正确预测的比例
- **假新闻准确率**：假新闻样本中正确识别的比例
- **真新闻准确率**：真新闻样本中正确识别的比例
- **提升对比**：情感增强判别相比纯真假判别的准确率提升

