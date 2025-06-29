import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import shap

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 1. 数据加载与预处理
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].apply(lambda x: 1 if x == "Real" else 0)
    return df


# 2. 情感特征提取
sentiment_analyzer = pipeline("sentiment-analysis", model="bert-base-uncased")


def extract_sentiment_features(texts):
    features = []
    for text in tqdm(texts, desc="Extracting Sentiment Features", unit="text"):
        try:
            # Ensure that text length is within acceptable bounds
            text = text[:512]  # BERT supports up to 512 tokens
            res = sentiment_analyzer(text)[0]
            pos = res["score"] if res["label"] == "POSITIVE" else 1 - res["score"]
            neg = 1 - pos
            features.append([pos, neg])
        except Exception as e:
            logging.error(f"Error extracting sentiment for text: {text}. Error: {str(e)}")
            features.append([0, 0])
    return torch.tensor(features, dtype=torch.float)


# 3. 主题特征提取
def train_topic_model(texts, n_topics=10):
    vec = CountVectorizer(max_df=0.85, min_df=5, stop_words="english")  # Adjusted max_df and min_df
    dtm = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    return vec, lda


def extract_topic_features(texts, vectorizer, lda_model):
    dtm = vectorizer.transform(texts)
    topic_dist = lda_model.transform(dtm)
    return torch.tensor(topic_dist, dtype=torch.float)  # Directly return the tensor


# 4. 数据集构造
class NewsDataset(Dataset):
    def __init__(self, texts, senti_feats, topic_feats, labels, tokenizer, max_len=128):
        self.texts = texts
        self.senti_feats = senti_feats
        self.topic_feats = topic_feats
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'sentiment': self.senti_feats[idx],
            'topic': self.topic_feats[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# 5. 模型定义
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, y):
        # Ensure that x and y are 3D tensors [batch_size, 1, dim] for batch matrix multiplication
        q = self.query(x).unsqueeze(1)  # [batch_size, 1, dim]
        k = self.key(y).unsqueeze(1)  # [batch_size, 1, dim]
        v = self.value(y).unsqueeze(1)  # [batch_size, 1, dim]

        # Perform batch matrix multiplication for attention
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [batch_size, 1, 1]
        attn = torch.softmax(attn, dim=-1)  # Apply softmax to get attention weights

        # Multiply attention weights with value
        return torch.bmm(attn, v).squeeze(1)  # [batch_size, dim] after removing the extra dimension


class MultimodalClassifier(nn.Module):
    def __init__(self, bert_model_name, topic_dim, senti_dim=2, hidden_dim=256, use_attention=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        # Linearly project sentiment and topic features to match bert_dim
        self.sentiment_projection = nn.Linear(senti_dim, bert_dim)
        self.topic_projection = nn.Linear(topic_dim, bert_dim)

        fusion_dim = bert_dim * 3  # cls_emb + sentiment + topic (after projection)
        self.use_attention = use_attention
        if use_attention:
            self.attn = CrossModalAttention(bert_dim)
            fusion_dim += bert_dim  # add attended vector dimension

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask, sentiment, topic):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.pooler_output  # Extract CLS embedding

        # Project sentiment and topic to the same dimension as cls_emb (bert_dim)
        sentiment_proj = self.sentiment_projection(sentiment).unsqueeze(1)  # Add extra dimension
        topic_proj = self.topic_projection(topic).unsqueeze(1)  # Add extra dimension

        # Concatenate cls_emb, sentiment_proj, and topic_proj
        features = [cls_emb, sentiment_proj, topic_proj]

        if self.use_attention:
            # Ensure all inputs have the correct shape for attention
            features.append(self.attn(cls_emb, sentiment_proj + topic_proj))  # Adjusted

        # Concatenate features along the last dimension
        fused = torch.cat(features, dim=1)
        return self.classifier(fused)


# 6. 训练与评估流程
def train(model, dataloader, optim, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment = batch['sentiment'].to(device)
        topic = batch['topic'].to(device)
        labels = batch['label'].to(device)

        preds = model(input_ids, attention_mask, sentiment, topic)
        loss = criterion(preds, labels)
        loss.backward()
        optim.step()

        total_loss += loss.item()

        # Record loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            topic = batch['topic'].to(device)
            labels = batch['label'].to(device)
            preds = model(input_ids, attention_mask, sentiment, topic)
            _, pred_idx = torch.max(preds, 1)
            correct += (pred_idx == labels).sum().item()
            total += labels.size(0)
    return correct / total


# 可视化：t-SNE 降维
def visualize_features(sentiment, topic, labels):
    features = torch.cat((sentiment, topic), dim=1).cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Visualization of Features")
    plt.show()


# 可视化：注意力权重
def visualize_attention_weights(cls_emb, sentiment_proj, topic_proj, attn_layer):
    # Ensure that cls_emb, sentiment_proj, and topic_proj are of compatible dimensions
    attn_weights = attn_layer(cls_emb, sentiment_proj + topic_proj).cpu().detach().numpy()

    # 可视化注意力权重
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap="YlGnBu", cbar=True)
    plt.title("Attention Weights Heatmap")
    plt.show()


# 可视化：SHAP
def visualize_shap_values(model, texts):
    explainer = shap.Explainer(model)
    shap_values = explainer(texts)
    shap.summary_plot(shap_values, texts)


# 7. 主函数示例
if __name__ == "__main__":
    # 参数设置
    DATA_PATH = "fake_and_real_news.csv"
    BERT_MODEL = "bert-base-uncased"
    N_TOPICS = 10
    BATCH_SIZE = 8
    EPOCHS = 3
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir="runs/experiment")

    # 1) 加载数据
    df = load_data(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 2) 提取情感特征
    senti_feats = extract_sentiment_features(texts)

    # 3) 训练主题模型并提取主题分布
    vec, lda = train_topic_model(texts, n_topics=N_TOPICS)
    topic_feats = extract_topic_features(texts, vec, lda)

    # 4) 划分数据集
    X_train, X_val, s_train, s_val, t_train, t_val, y_train, y_val = train_test_split(
        texts, senti_feats, topic_feats, labels, test_size=0.2, random_state=42
    )
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train_ds = NewsDataset(X_train, s_train, t_train, y_train, tokenizer)
    val_ds = NewsDataset(X_val, s_val, t_val, y_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 5) 初始化模型
    model = MultimodalClassifier(BERT_MODEL, topic_dim=N_TOPICS, use_attention=True)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 6) 训练与评估
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE, writer, epoch)
        val_acc = evaluate(model, val_loader, DEVICE)
        logging.info(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

        # 可视化注意力权重
        for batch in tqdm(train_loader, desc="Visualizing Attention Weights"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            sentiment = batch['sentiment'].to(DEVICE)
            topic = batch['topic'].to(DEVICE)

            outputs = model(input_ids, attention_mask, sentiment, topic)
            cls_emb = outputs[0]  # Get cls_emb from model output
            sentiment_proj = model.sentiment_projection(sentiment)
            topic_proj = model.topic_projection(topic)

            visualize_attention_weights(cls_emb, sentiment_proj, topic_proj, model.attn)

    # 保存模型
    save_path = "multimodal_model.pt"  # Absolute or relative path
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved at '{save_path}'")

    # 关闭 TensorBoard writer
    writer.close()
