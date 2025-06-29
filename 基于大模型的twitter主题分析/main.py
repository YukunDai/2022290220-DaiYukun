# -*- coding: utf-8 -*-
"""
twitter_topic_analysis.py
基于 DeepSeek 大模型的 Twitter 主题分析项目示例脚本（含调试输出，增强 API 错误处理）
"""

import os
import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim_models
import pyLDAvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import webbrowser
from dotenv import load_dotenv
from openai import OpenAI

# ============================
# 0. NLTK 资源下载（仅需运行一次）
# ============================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ============================
# 1. 客户端初始化（DeepSeek API）
# ============================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("DEEPSEEK_API_URL")

if not api_key or not base_url:
    print("[ERROR] 请检查 .env 中是否正确配置 OPENAI_API_KEY 和 DEEPSEEK_API_URL")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=base_url)


def get_completion(prompt, model="deepseek-reasoner"):
    """
    调用 DeepSeek 进行单轮对话并返回内容（含调试打印及错误处理）
    """
    print(f"[DEBUG] 调用 DeepSeek，prompt: {prompt}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content
        print(f"[DEBUG] DeepSeek 返回: {result}")
        return result
    except Exception as e:
        err = e.__str__()
        if 'Authentication Fails' in err:
            print("[ERROR] DeepSeek 验证失败，请检查 API Key 是否有效")
        else:
            print(f"[ERROR] 调用 DeepSeek API 失败： {err}")
        return "[WARN] 语义分析未执行"

# ============================
# 2. 数据准备
# ============================
# 示例 10 条推文，请替换为实际数据
tweets = [
    "Just had the best coffee in Seattle! #coffee #morning",
    "AI is transforming the world at an unprecedented pace. #AI #tech",
    "Can't believe how hot it is today... #summer #heatwave",
    "The new iPhone release event was disappointing. #Apple #iPhone",
    "Blockchain technology will revolutionize finance. #crypto #blockchain",
    "Watching the Olympics opening ceremony tonight! #Olympics #sports",
    "Elections are coming up, remember to vote. #politics #vote",
    "Excited for the concert this weekend. #music #live",
    "Global warming is a serious threat to our planet. #environment",
    "Learning Python for data science is so much fun! #Python #DataScience"
]
print(f"[DEBUG] 原始推文条数: {len(tweets)}")

# ============================
# 3. 数据预处理
# ============================
stop_words = set(stopwords.words('english'))
def preprocess(texts):
    lemmatizer = WordNetLemmatizer()
    processed = []
    for idx, doc in enumerate(texts):
        doc_clean = re.sub('[^a-zA-Z]', ' ', doc).lower()
        tokens = [t for t in doc_clean.split() if t not in stop_words and len(t) > 2]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        processed.append(tokens)
        print(f"[DEBUG] 文档 {idx} 预处理后 tokens: {tokens}")
    return processed

processed_docs = preprocess(tweets)
print(f"[DEBUG] 预处理后文档示例: {processed_docs[:2]}")

# ============================
# 4. 模型构建与训练
# ============================
dictionary = corpora.Dictionary(processed_docs)
print(f"[DEBUG] 原始词典大小: {len(dictionary)}")
dictionary.filter_extremes(no_below=1, no_above=0.5)
print(f"[DEBUG] 过滤后词典大小: {len(dictionary)}")
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(f"[DEBUG] 示例 Bow (文档0): {corpus[0]}")
num_topics = 3
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=10
)
print(f"[DEBUG] LDA 训练完成，主题数: {num_topics}")
for idx in range(num_topics):
    print(f"[DEBUG] 主题 {idx} 关键词: {lda_model.print_topic(idx, topn=10)}")
for i, bow in enumerate(corpus):
    print(f"[DEBUG] 文档 {i} 主题分布: {lda_model.get_document_topics(bow)}")

# ============================
# 5. 可视化分析
# ============================
vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
try:
    pyLDAvis.show(vis_data)
except Exception:
    html_path = 'lda_vis.html'
    pyLDAvis.save_html(vis_data, html_path)
    print(f"[DEBUG] 已保存可视化到: {html_path}")
    webbrowser.open(html_path)
for topic_id in range(num_topics):
    plt.figure(figsize=(8, 4))
    wc = WordCloud(width=800, height=400)
    wc.generate_from_frequencies(dict(lda_model.show_topic(topic_id, topn=20)))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"主题 {topic_id} 词云")
    plt.show()
prob_matrix = [[dict(dt).get(i, 0) for i in range(num_topics)] for dt in [lda_model.get_document_topics(b) for b in corpus]]
plt.figure()
sns.heatmap(pd.DataFrame(prob_matrix, columns=[f"主题 {i}" for i in range(num_topics)]), annot=True, cmap='Blues')
plt.title("文档-主题概率分布热力图")
plt.show()

# ============================
# 6. 语义分析
# ============================
for topic_id in range(num_topics):
    keywords = [w for w, _ in lda_model.show_topic(topic_id, topn=5)]
    summary = get_completion(f"请根据关键词{keywords}描述主题{topic_id}核心内容")
    print(f"主题 {topic_id} 内容分析（DeepSeek）: {summary}")
