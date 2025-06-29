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
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import warnings
from matplotlib.colors import ListedColormap

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)  # 抑制FutureWarning


# 1. 数据加载与预处理
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].apply(lambda x: 1 if x == "Real" else 0)

    # 文本清洗：去除标点符号、转换为小写
    df["text"] = df["text"].str.replace(r'[^\w\s]', '', regex=True).str.lower()

    # 添加文本长度信息
    df["text_length"] = df["text"].apply(len)

    return df


# 2. 情感特征提取
sentiment_analyzer = pipeline("sentiment-analysis", model="bert-base-uncased",
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 自动检测GPU


def extract_sentiment_features(texts):
    features = []
    for text in tqdm(texts, desc="Extracting Sentiment Features", unit="text"):
        try:
            text = text[:512]  # BERT支持最大512个tokens
            res = sentiment_analyzer(text)[0]
            pos = res["score"] if res["label"] == "POSITIVE" else 1 - res["score"]
            neg = 1 - pos
            features.append([pos, neg])
        except Exception as e:
            logging.error(f"Error extracting sentiment for text: {text}. Error: {str(e)}")
            features.append([0.5, 0.5])  # 中性情感
    return torch.tensor(features, dtype=torch.float)


# 3. 主题特征提取
def train_topic_model(texts, n_topics=10):
    vec = CountVectorizer(max_df=0.85, min_df=5, stop_words="english")
    dtm = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    return vec, lda


def extract_topic_features(texts, vectorizer, lda_model):
    dtm = vectorizer.transform(texts)
    topic_dist = lda_model.transform(dtm)
    return torch.tensor(topic_dist, dtype=torch.float)


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
            'input_ids': inputs['input_ids'].view(-1),  # 展平
            'attention_mask': inputs['attention_mask'].view(-1),
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
        q = self.query(x)  # [batch_size, dim]
        k = self.key(y)  # [batch_size, dim]
        v = self.value(y)  # [batch_size, dim]

        q = q.unsqueeze(1)  # [batch_size, 1, dim]
        k = k.unsqueeze(1)  # [batch_size, 1, dim]
        v = v.unsqueeze(1)  # [batch_size, 1, dim]

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [batch_size, 1, 1]
        attn = torch.softmax(attn, dim=-1)

        return torch.bmm(attn, v).squeeze(1)  # [batch_size, dim]


class MultimodalClassifier(nn.Module):
    def __init__(self, bert_model_name, topic_dim, senti_dim=2, hidden_dim=256, use_attention=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

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
        cls_emb = outputs.pooler_output  # [batch_size, bert_dim]

        sentiment_proj = self.sentiment_projection(sentiment)
        topic_proj = self.topic_projection(topic)

        features = [cls_emb, sentiment_proj, topic_proj]

        if self.use_attention:
            attn_output = self.attn(cls_emb, sentiment_proj + topic_proj)
            features.append(attn_output)

        fused = torch.cat(features, dim=1)

        return self.classifier(fused)


# 6. 训练与评估流程
def train(model, dataloader, optim, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

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

        # 确保正确解包 preds
        if len(preds.shape) == 2:  # 如果 preds 形状为 [batch_size, num_classes]
            _, predicted = torch.max(preds, dim=1)  # 获取预测的类别
        else:
            predicted = preds  # 如果 preds 已经是一个分类结果（例如二分类任务）

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(dataloader) + batch_idx)

    epoch_acc = correct / total
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    return total_loss / len(dataloader), epoch_acc


def evaluate(model, dataloader, device, epoch, writer, mode='val'):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            topic = batch['topic'].to(device)
            labels = batch['label'].to(device)
            preds = model(input_ids, attention_mask, sentiment, topic)

            probs, pred_idx = torch.max(preds, 1)
            all_preds.extend(pred_idx.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(preds[:, 1].cpu().numpy())

            correct += (pred_idx == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    writer.add_scalar(f'Accuracy/{mode}', accuracy, epoch)

    report = classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], output_dict=True)
    writer.add_scalar(f'Precision/{mode}', report['weighted avg']['precision'], epoch)
    writer.add_scalar(f'Recall/{mode}', report['weighted avg']['recall'], epoch)
    writer.add_scalar(f'F1/{mode}', report['weighted avg']['f1-score'], epoch)

    return accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_learning_curves(train_losses, val_accs, title='Learning Curves'):
    """绘制学习曲线"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制训练损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, len(train_losses) + 1), train_losses, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 绘制验证准确率
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(1, len(val_accs) + 1), val_accs, color=color, marker='s', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # 图表标题和布局
    plt.title(title)
    fig.tight_layout()
    plt.legend(loc='best')
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()


def plot_roc_curve(y_true, y_probs, title='ROC Curve'):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

    return roc_auc


def plot_precision_recall_curve(y_true, y_probs, title='Precision-Recall Curve'):
    """绘制精确率-召回率曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    average_precision = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

    return average_precision


# 主函数
if __name__ == "__main__":
    DATA_PATH = "fake_and_real_news.csv"
    BERT_MODEL = "bert-base-uncased"
    N_TOPICS = 10
    BATCH_SIZE = 8
    EPOCHS = 3
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("visualizations", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/experiment")

    df = load_data(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 提取情感特征
    senti_feats = extract_sentiment_features(texts)

    # 训练主题模型
    vec, lda = train_topic_model(texts, n_topics=N_TOPICS)
    topic_feats = extract_topic_features(texts, vec, lda)

    # 数据集划分
    X_train, X_val, s_train, s_val, t_train, t_val, y_train, y_val = train_test_split(
        texts, senti_feats, topic_feats, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train_ds = NewsDataset(X_train, s_train, t_train, y_train, tokenizer)
    val_ds = NewsDataset(X_val, s_val, t_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 初始化模型
    model = MultimodalClassifier(BERT_MODEL, topic_dim=N_TOPICS, use_attention=True)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE, writer, epoch)
        val_acc, val_labels, val_preds, val_probs = evaluate(model, val_loader, DEVICE, epoch, writer, 'val')

        train_losses.append(train_loss)
        val_accs.append(val_acc)

    plot_learning_curves(train_losses, val_accs, "Training Progress")
    val_acc, val_labels, val_preds, val_probs = evaluate(model, val_loader, DEVICE, EPOCHS, writer, 'val_final')

    plot_confusion_matrix(val_labels, val_preds, classes=['Fake', 'Real'], title='Confusion Matrix')
    roc_auc = plot_roc_curve(val_labels, val_probs, "ROC Curve")
    pr_auc = plot_precision_recall_curve(val_labels, val_probs, "Precision-Recall Curve")

    writer.close()
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
