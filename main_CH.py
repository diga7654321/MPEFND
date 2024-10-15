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



def load_data(input_file):
    text = []
    judgements = []
    labels = []


    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        
        for value in data.values():
           
            if isinstance(value['comments'], list):
                comments_str = ' '.join(value['comments'])  
            else:
                comments_str = value['comments']  

            text.append(value['content'] + comments_str)  
            judgements.append(value['judgements'])
            labels.append(value['label'])  

    
    return text, judgements, labels



class CustomOutputLayer(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(CustomOutputLayer, self).__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x, count):
        min_value = x.min()
        max_value = x.max()

       
        range_value = max_value - min_value
        range_value[range_value == 0] = 1e-8 

      
        x = (x - min_value) / range_value

        x = x * count  
        y = x
        x = self.linear(x)

        # print(x)
        return x,y



class CustomBertClassifier(nn.Module):
    def __init__(self, model_path, num_labels=2, mlp_hidden_sizes=[1024, 512, 256], dropout_prob=0.1):
        super(CustomBertClassifier, self).__init__()

        
        self.bert = BertModel.from_pretrained(model_path)

        
        for param in self.bert.parameters():
            param.requires_grad = False

        
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        
        self.uni_repre = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

     
        input_dim = 256 * (25 + 1) 
        self.mlp_layers = nn.ModuleList()
        for hidden_size in mlp_hidden_sizes:
            self.mlp_layers.append(nn.Linear(input_dim, hidden_size))
            self.mlp_layers.append(nn.LayerNorm(hidden_size))
            self.mlp_layers.append(nn.GELU())
            self.mlp_layers.append(nn.Dropout(p=dropout_prob))
            input_dim = hidden_size

        self.output_layer = CustomOutputLayer(input_dim, num_labels)

    def forward(self, input_ids_text, attention_mask_text, input_ids_judgements, attention_mask_judgements, count,
                judgement_results):

        text_outputs = self.bert(input_ids=input_ids_text, attention_mask=attention_mask_text)
        text_pooled_output = text_outputs.pooler_output  

        batch_size, num_judgements, max_length = input_ids_judgements.shape
        judgements_pooled_output = []


        for i in range(num_judgements):

            judgement_outputs = self.bert(
                input_ids=input_ids_judgements[:, i, :],
                attention_mask=attention_mask_judgements[:, i, :]
            )
            judgements_pooled_output.append(judgement_outputs.pooler_output)


        text_prime = self.uni_repre(text_pooled_output)  # [batch_size, 256]


        weighted_judgements_list = []


        for i in range(num_judgements):
            judgement_prime = self.uni_repre(judgements_pooled_output[i])  # [batch_size, 256]

  
            similarity = torch.nn.functional.cosine_similarity(
                text_pooled_output, judgements_pooled_output[i], dim=1
            )

            normalized_similarity = (similarity + 1) / 2

   
            similarity = torch.where(judgement_results[:, i] == 1, normalized_similarity, 1 - normalized_similarity)

   
            weighted_judgement = judgement_prime * similarity.unsqueeze(1) 

 
            weighted_judgements_list.append(weighted_judgement)

 
        weighted_judgements_concat = torch.cat(weighted_judgements_list, dim=1)

   
        combined_output = torch.cat([text_prime, weighted_judgements_concat], dim=1)


        adjusted_count = torch.log1p(count).unsqueeze(1) 


        x = combined_output
        for layer in self.mlp_layers:
            x = layer(x)


        output,y = self.output_layer(x, adjusted_count)
        self.features = y.clone().detach()
        # print(f"Logits shape: {output.shape}")

        return output


def custom_collate_fn(batch):
    max_length = 512  
    input_ids_judgements = []
    attention_mask_judgements = []
    input_ids_text = []
    attention_mask_text = []
    labels = []
    counts = []
    judgement_results_list = [] 

    for b in batch:
   
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
        judgement_results_list.append(b['judgement_results']) 


    input_ids_judgements = torch.stack(input_ids_judgements)  # [batch_size, 25, max_length]
    attention_mask_judgements = torch.stack(attention_mask_judgements)  # [batch_size, 25, max_length]
    input_ids_text = torch.stack(input_ids_text)  # [batch_size, max_length]
    attention_mask_text = torch.stack(attention_mask_text)  # [batch_size, max_length]
    labels = torch.stack(labels)  # [batch_size]
    counts = torch.stack(counts)  # [batch_size]
    judgement_results = torch.stack(judgement_results_list)  # [batch_size, 25]


    return {
        'input_ids_judgements': input_ids_judgements,
        'attention_mask_judgements': attention_mask_judgements,
        'input_ids_text': input_ids_text,
        'attention_mask_text': attention_mask_text,
        'label': labels,
        'count': counts,
        'judgement_results': judgement_results  # 包含 judgement_results
    }



class RumorDataset(Dataset):
    def __init__(self, texts, labels, judgements, tokenizer, device='cuda', max_length=512):
        self.texts = texts
        self.labels = labels
        self.judgements = judgements
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device


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


        total_count = 0.0


        judgements_input_ids = []
        judgements_attention_masks = [] 
        judgement_results = []


        for i in range(1, 26):
            judgement_key = f"judgement_{i}"
            judgement_text = self.judgements[idx].get(judgement_key, "")


            if "判断：是" in judgement_text:
                total_count += self.judgement_weights[i]
                judgement_results.append(1)
            else:
                judgement_results.append(0) 


            encoding_judgement = self.tokenizer(
                judgement_text,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )


            judgements_input_ids.append(encoding_judgement['input_ids'].squeeze(0)) 
            judgements_attention_masks.append(encoding_judgement['attention_mask'].squeeze(0))

        total_count = total_count * 1000000
        # total_count=1


        judgements_input_ids_tensor = torch.stack(judgements_input_ids)
        judgements_attention_masks_tensor = torch.stack(judgements_attention_masks)


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
            'judgement_results': torch.tensor(judgement_results, dtype=torch.float) 
        }



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


            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())


            # features = model.features
            # all_features.append(features.cpu().numpy())

        # all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)


        # np.savez(f'features_epoch_{epoch}.npz', features=all_features, labels=all_labels)


    accuracy = correct_preds / total_preds
    avg_loss = total_loss / len(data_loader)


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


    torch.cuda.empty_cache()


    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    random_seed = 16
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_file = r"datasets/OUTPUT/WEIBO.json"
    model_path = "model/chinese-roberta-wwm-ext"


    classifier = CustomBertClassifier(model_path, num_labels=2).to(device)

    classifier.load_state_dict(torch.load("checkpoint/checkpoint_CH.pt"))


    classifier.eval()

    tokenizer = BertTokenizer.from_pretrained(model_path)


    train_texts, train_judgements, train_labels = load_data(train_file)


    train_texts, test_texts, train_labels, test_labels, train_judgements, test_judgements = train_test_split(
        train_texts, train_labels, train_judgements, test_size=0.2, random_state=random_seed, stratify=train_labels
    )


    train_dataset = RumorDataset(train_texts, train_labels, train_judgements, tokenizer, device)
    test_dataset = RumorDataset(test_texts, test_labels, test_judgements, tokenizer, device)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)


    optimizer = AdamW(classifier.parameters(), lr=1e-4)
    num_epochs = 50
    num_training_steps = len(train_loader) * num_epochs * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    loss_fct = torch.nn.CrossEntropyLoss()


    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, device, scheduler, loss_fct)
        # print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

        # if (epoch + 1) % 5 == 0:
        test_acc, avg_loss = evaluate(classifier, test_loader, device, loss_fct)
        print(f"Test accuracy: {test_acc:.4f}, avg_loss: {avg_loss:.4f}")
