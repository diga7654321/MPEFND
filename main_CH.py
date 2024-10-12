import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import warnings
import logging
from sklearn.model_selection import train_test_split
import torch.nn as nn
from transformers import BertModel
import numpy as np


# 1. 加载数据函数
def load_data(input_file):
    text = []
    judgements = []
    labels = []

    # 打开并读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # 从数据集中提取文本和标签
        for value in data.values():
            # 确保 comments 是一个字符串，如果是列表则将其拼接为字符串
            if isinstance(value['comments'], list):
                comments_str = ' '.join(value['comments'])  # 将列表中的元素用空格连接
            else:
                comments_str = value['comments']  # 如果不是列表，直接使用

            text.append(value['content'] + comments_str)  # 提取新闻文本
            judgements.append(value['judgements'])
            labels.append(value['label'])  # 提取对应的标签

    # 返回文本列表和标签列表
    return text, judgements, labels


# class AttentionLayer(nn.Module):
#     def __init__(self, input_dim):
#         super(AttentionLayer, self).__init__()
#         self.attention = nn.Linear(input_dim, 1)
#
#     def forward(self, x):
#         # x 的形状应该是 [batch_size, feature_dim]
#         attn_weights = torch.softmax(self.attention(x), dim=1)
#         # 返回 x 的加权和，保持 [batch_size, feature_dim]
#         return x * attn_weights


class CustomOutputLayer(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(CustomOutputLayer, self).__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x, count):
        min_value = x.min()
        max_value = x.max()

        # 防止除以零的情况
        range_value = max_value - min_value
        range_value[range_value == 0] = 1e-8  # 避免除以 0

        # 归一化 combined_output
        x = (x - min_value) / range_value

        x = x * count  # 使用调整后的 count 作为权重
        y = x
        x = self.linear(x)

        # print(x)
        return x,y


# 定义自定义分类器
class CustomBertClassifier(nn.Module):
    def __init__(self, model_path, num_labels=2, mlp_hidden_sizes=[1024, 512, 256], dropout_prob=0.1):
        super(CustomBertClassifier, self).__init__()

        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(model_path)

        # 冻结 BERT 参数，只训练下游任务
        for param in self.bert.parameters():
            param.requires_grad = False

        # 解冻 BERT 的最后两层
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 定义 MLP 模块，用于处理文本特征和 judgements 特征
        self.uni_repre = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 增加 MLP 隐藏层的维度和层数
        input_dim = 256 * (25 + 1)  # 25 条 judgements 加上 1 个文本的拼接维度
        self.mlp_layers = nn.ModuleList()
        for hidden_size in mlp_hidden_sizes:
            self.mlp_layers.append(nn.Linear(input_dim, hidden_size))
            self.mlp_layers.append(nn.LayerNorm(hidden_size))
            self.mlp_layers.append(nn.GELU())
            self.mlp_layers.append(nn.Dropout(p=dropout_prob))
            input_dim = hidden_size

        # 输出层
        self.output_layer = CustomOutputLayer(input_dim, num_labels)

    def forward(self, input_ids_text, attention_mask_text, input_ids_judgements, attention_mask_judgements, count,
                judgement_results):
        # 处理 text 部分
        text_outputs = self.bert(input_ids=input_ids_text, attention_mask=attention_mask_text)
        text_pooled_output = text_outputs.pooler_output  # BERT 对 text 的池化输出

        batch_size, num_judgements, max_length = input_ids_judgements.shape
        judgements_pooled_output = []

        # 分别对每条 judgement 进行处理
        for i in range(num_judgements):
            # 对每条 judgement 单独通过 BERT 模型
            judgement_outputs = self.bert(
                input_ids=input_ids_judgements[:, i, :],
                attention_mask=attention_mask_judgements[:, i, :]
            )
            judgements_pooled_output.append(judgement_outputs.pooler_output)

        # 使用 MLP 模块处理 text 特征
        text_prime = self.uni_repre(text_pooled_output)  # [batch_size, 256]

        # 初始化空列表，用于存储加权后的 judgements 特征
        weighted_judgements_list = []

        # 分别处理每条 judgement
        for i in range(num_judgements):
            judgement_prime = self.uni_repre(judgements_pooled_output[i])  # [batch_size, 256]

            # 计算 text 与 judgement 的相似度
            similarity = torch.nn.functional.cosine_similarity(
                text_pooled_output, judgements_pooled_output[i], dim=1
            )

            normalized_similarity = (similarity + 1) / 2

            # 根据判断结果调整相似度
            similarity = torch.where(judgement_results[:, i] == 1, normalized_similarity, 1 - normalized_similarity)

            # 使用相似度加权 judgement 特征
            weighted_judgement = judgement_prime * similarity.unsqueeze(1)  # 扩展维度为 [batch_size, 1]

            # 添加到列表中
            weighted_judgements_list.append(weighted_judgement)

        # 将所有加权后的 judgements 特征拼接，形状为 [batch_size, 256 * 25]
        weighted_judgements_concat = torch.cat(weighted_judgements_list, dim=1)

        # 将 text 特征与 judgements 特征拼接，形状为 [batch_size, 256 * (25 + 1)]
        combined_output = torch.cat([text_prime, weighted_judgements_concat], dim=1)

        # 使用对数变换平滑 count 的影响
        adjusted_count = torch.log1p(count).unsqueeze(1)  # 确保 adjusted_count 的形状是 [batch_size, 1]

        # 通过 MLP 层处理拼接后的特征
        x = combined_output
        for layer in self.mlp_layers:
            x = layer(x)

        # 最终输出
        output,y = self.output_layer(x, adjusted_count)
        self.features = y.clone().detach()
        # print(f"Logits shape: {output.shape}")  # 确保输出形状是 [batch_size, num_labels]

        return output


def custom_collate_fn(batch):
    max_length = 512  # 设置固定的最大长度

    input_ids_judgements = []
    attention_mask_judgements = []
    input_ids_text = []
    attention_mask_text = []
    labels = []
    counts = []
    judgement_results_list = []  # 新增用于存储 judgement_results

    for b in batch:
        # 处理 judgements
        judgements_input_ids_padded = torch.stack([
            torch.cat([judgement, torch.zeros(max_length - judgement.size(0), dtype=torch.long)])
            for judgement in b['input_ids_judgements']
        ])
        judgements_attention_mask_padded = torch.stack([
            torch.cat([judgement, torch.zeros(max_length - judgement.size(0), dtype=torch.long)])
            for judgement in b['attention_mask_judgements']
        ])

        input_ids_judgements.append(judgements_input_ids_padded)
        attention_mask_judgements.append(judgements_attention_mask_padded)

        # 处理 text
        input_ids_text_padded = torch.cat(
            [b['input_ids_text'], torch.zeros(max_length - b['input_ids_text'].size(0), dtype=torch.long)]
        )
        attention_mask_text_padded = torch.cat(
            [b['attention_mask_text'], torch.zeros(max_length - b['attention_mask_text'].size(0), dtype=torch.long)]
        )

        input_ids_text.append(input_ids_text_padded)
        attention_mask_text.append(attention_mask_text_padded)

        labels.append(b['label'])
        counts.append(b['count'])
        judgement_results_list.append(b['judgement_results'])  # 收集 judgement_results

    # 转换为张量
    input_ids_judgements = torch.stack(input_ids_judgements)  # [batch_size, 25, max_length]
    attention_mask_judgements = torch.stack(attention_mask_judgements)  # [batch_size, 25, max_length]
    input_ids_text = torch.stack(input_ids_text)  # [batch_size, max_length]
    attention_mask_text = torch.stack(attention_mask_text)  # [batch_size, max_length]
    labels = torch.stack(labels)  # [batch_size]
    counts = torch.stack(counts)  # [batch_size]
    judgement_results = torch.stack(judgement_results_list)  # [batch_size, 25]

    # 返回所有张量，包括 judgement_results
    return {
        'input_ids_judgements': input_ids_judgements,
        'attention_mask_judgements': attention_mask_judgements,
        'input_ids_text': input_ids_text,
        'attention_mask_text': attention_mask_text,
        'label': labels,
        'count': counts,
        'judgement_results': judgement_results  # 包含 judgement_results
    }


# 定义 Dataset 类
class RumorDataset(Dataset):
    def __init__(self, texts, labels, judgements, tokenizer, device='cuda', max_length=512):
        self.texts = texts
        self.labels = labels
        self.judgements = judgements
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        # 定义每条判断的权重
        self.judgement_weights = {
            1: 0.94, 2: 15.9, 3: 1.38, 4: 5.56, 5: 0.81, 6: 1.51, 7: 1.69, 8: 2.35, 9: 2.34,
            10: 0.59, 11: 2.47, 12: 0.91, 13: 0.89, 14: 173.0, 15: 1.13, 16: 3.26, 17: 3.28,
            18: 6.2, 19: 278.17, 20: 2.4, 21: 9.68, 22: 0.38, 23: 2.49, 24: 0.67, 25: 0.93
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 初始化 count 为 0
        total_count = 0.0

        # 对每条 judgement 进行处理并编码
        judgements_input_ids = []
        judgements_attention_masks = []  # 初始化 count 为 0
        judgement_results = []

        # 遍历每个 judgement_i，计算 count 和 judgement_results
        for i in range(1, 26):
            judgement_key = f"judgement_{i}"
            judgement_text = self.judgements[idx].get(judgement_key, "")

            # 判断是否有 "判断：是"，并根据对应的权重累加
            if "判断：是" in judgement_text:
                total_count += self.judgement_weights[i]
                judgement_results.append(1)
            else:
                judgement_results.append(0)  # 否则添加 0

            # 编码每条 judgement
            encoding_judgement = self.tokenizer(
                judgement_text,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )

            # 将每条判断的 input_ids 和 attention_mask 添加到列表中
            judgements_input_ids.append(encoding_judgement['input_ids'].squeeze(0))  # 去除 batch 维度
            judgements_attention_masks.append(encoding_judgement['attention_mask'].squeeze(0))

        total_count = total_count * 1000000
        # total_count=1

        # 将所有 25 条判断的 input_ids 和 attention_masks 转换成张量
        judgements_input_ids_tensor = torch.stack(judgements_input_ids)
        judgements_attention_masks_tensor = torch.stack(judgements_attention_masks)

        # 编码 text
        encoding_text = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        return {
            'input_ids_text': encoding_text['input_ids'].squeeze(0),
            'attention_mask_text': encoding_text['attention_mask'].squeeze(0),
            'input_ids_judgements': judgements_input_ids_tensor,  # [25, max_length]
            'attention_mask_judgements': judgements_attention_masks_tensor,  # [25, max_length]
            'label': torch.tensor(label, dtype=torch.long),
            'count': torch.tensor(total_count, dtype=torch.float),
            'judgement_results': torch.tensor(judgement_results, dtype=torch.float)  # 返回判断结果
        }


# 训练函数
def train_epoch(model, data_loader, optimizer, device, scheduler, loss_fct):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    loop = tqdm(data_loader, leave=True, desc="Training")
    for batch in loop:
        input_ids_text = batch['input_ids_text'].to(device)
        attention_mask_text = batch['attention_mask_text'].to(device)
        input_ids_judgements = batch['input_ids_judgements'].to(device)
        attention_mask_judgements = batch['attention_mask_judgements'].to(device)
        labels = batch['label'].to(device)
        count = batch['count'].to(device)
        judgement_results = batch['judgement_results'].to(device)

        optimizer.zero_grad()

        # 前向传播，模型内部处理相似度计算和加权
        logits = model(
            input_ids_text=input_ids_text,
            attention_mask_text=attention_mask_text,
            input_ids_judgements=input_ids_judgements,
            attention_mask_judgements=attention_mask_judgements,
            count=count,
            judgement_results=judgement_results
        )

        loss = loss_fct(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy


# 评估函数
def evaluate(model, data_loader, device, loss_fct):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_preds = []
    all_features = []

    loop = tqdm(data_loader, leave=True, desc="Evaluating")

    with torch.no_grad():
        for batch in loop:
            input_ids_text = batch['input_ids_text'].to(device)
            attention_mask_text = batch['attention_mask_text'].to(device)
            input_ids_judgements = batch['input_ids_judgements'].to(device)
            attention_mask_judgements = batch['attention_mask_judgements'].to(device)
            labels = batch['label'].to(device)
            count = batch['count'].to(device)
            judgement_results = batch['judgement_results'].to(device)

            # 前向传播，模型内部处理相似度计算和加权
            logits = model(
                input_ids_text=input_ids_text,
                attention_mask_text=attention_mask_text,
                input_ids_judgements=input_ids_judgements,
                attention_mask_judgements=attention_mask_judgements,
                count=count,
                judgement_results=judgement_results
            )

            loss = loss_fct(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

            # 将标签和预测结果存储到列表中
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # 提取中间特征并保存
            features = model.features
            all_features.append(features.cpu().numpy())  # 确保将特征移到 CPU

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)

        # 将整个 epoch 的特征和标签保存为 .npz 文件
        np.savez(f'features_epoch_{epoch}.npz', features=all_features, labels=all_labels)

    # 计算准确率
    accuracy = correct_preds / total_preds
    avg_loss = total_loss / len(data_loader)

    # 计算混淆矩阵和各项指标
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # torch.save(classifier.state_dict(), f"model_epoch_{epoch + 1}.pt")
    # print(f"Model parameters saved for epoch {epoch + 1}.")

    return accuracy, avg_loss


if __name__ == '__main__':

    # 清除 GPU 缓存
    torch.cuda.empty_cache()

    # 关闭 transformers 库的警告日志
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    # 设置随机种子
    random_seed = 16
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件路径和模型路径
    train_file = r"datasets/OUTPUT/WEIBO.json"
    model_path = "model/chinese-roberta-wwm-ext"

    # 加载模型和分词器
    classifier = CustomBertClassifier(model_path, num_labels=2).to(device)
    # 加载保存的模型参数
    classifier.load_state_dict(torch.load("checkpoint/checkpoint_CH.pt"))

    # 将模型设置为评估模式
    classifier.eval()

    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 加载数据
    train_texts, train_judgements, train_labels = load_data(train_file)

    # 分层抽样
    train_texts, test_texts, train_labels, test_labels, train_judgements, test_judgements = train_test_split(
        train_texts, train_labels, train_judgements, test_size=0.2, random_state=random_seed, stratify=train_labels
    )

    # 创建 Dataset 和 DataLoader
    train_dataset = RumorDataset(train_texts, train_labels, train_judgements, tokenizer, device)
    test_dataset = RumorDataset(test_texts, test_labels, test_judgements, tokenizer, device)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)

    # 优化器和调度器
    optimizer = AdamW(classifier.parameters(), lr=1e-4)
    num_epochs = 50
    num_training_steps = len(train_loader) * num_epochs * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    loss_fct = torch.nn.CrossEntropyLoss()

    # 训练和评估循环
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, device, scheduler, loss_fct)
        # print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

        # if (epoch + 1) % 5 == 0:
        test_acc, avg_loss = evaluate(classifier, test_loader, device, loss_fct)
        print(f"Test accuracy: {test_acc:.4f}, avg_loss: {avg_loss:.4f}")
