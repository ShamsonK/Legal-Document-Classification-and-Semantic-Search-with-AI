import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer, util
import torch
from torch.utils.data import Dataset

# Function to evaluate model
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

# Custom Dataset for BERT
class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to process data and train models
def process_data_and_train_models(df):
    # Clean and categorize labels
    df['Offense'] = df['Offense'].str.split(';').str[0].str.strip()

    # Merge small classes into a common category
    class_counts = df['Offense'].value_counts()
    small_classes = class_counts[class_counts < 2].index
    df['Offense'] = df['Offense'].apply(lambda x: 'Other' if x in small_classes else x)

    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Offense'])

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Raw Content'])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # Evaluation
    accuracy_rf, precision_rf, recall_rf, f1_rf = evaluate_model(y_test, y_pred_rf)
    accuracy_xgb, precision_xgb, recall_xgb, f1_xgb = evaluate_model(y_test, y_pred_xgb)

    # BERT-based Model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(y)))

    train_dataset = LegalDataset(df['Raw Content'].tolist(), y.tolist(), tokenizer, max_len=128)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to='none'  # Disable wandb logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'bert_model': model,
        'accuracy_rf': accuracy_rf,
        'precision_rf': precision_rf,
        'recall_rf': recall_rf,
        'f1_rf': f1_rf,
        'accuracy_xgb': accuracy_xgb,
        'precision_xgb': precision_xgb,
        'recall_xgb': recall_xgb,
        'f1_xgb': f1_xgb
    }

# Function to perform semantic search
def perform_semantic_search(df, query):
    sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embeddings = sentence_model.encode(df['Raw Content'].tolist(), convert_to_tensor=True)
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)
    best_matches = [df['Raw Content'].iloc[idx] for idx in top_results[1]]
    return best_matches

# Example usage
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('path_to_your_dataset.csv')

    # Process data and train models
    results = process_data_and_train_models(df)

    # Print evaluation results
    print(f"Random Forest - Accuracy: {results['accuracy_rf']}, Precision: {results['precision_rf']}, Recall: {results['recall_rf']}, F1 Score: {results['f1_rf']}")
    print(f"XGBoost - Accuracy: {results['accuracy_xgb']}, Precision: {results['precision_xgb']}, Recall: {results['recall_xgb']}, F1 Score: {results['f1_xgb']}")

    # Perform semantic search
    query = "example query"
    best_matches = perform_semantic_search(df, query)
    print(f"Top Results: {best_matches}")
