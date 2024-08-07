from dataclasses import dataclass
from typing import Generator
from typing import List
from typing import Tuple
import torch
from transformers import PreTrainedTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import numpy as np


@dataclass
class DataLoader:
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: str = None

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        # Number of rows
        with open(self.path) as f:
            n_rows = sum(1 for _ in f)
        n_rows -= 1  # Skip header

        # Round up
        return n_rows // self.batch_size + bool(n_rows % self.batch_size)

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        output = []
        if self.padding == "max_length":
            padding = "max_length"
        elif self.padding == "batch":
            padding = "longest"
        else:
            padding = "do_not_pad"
        for text in batch:
            tokenized = self.tokenizer.encode(
                text,
                padding=padding,
                max_length=self.max_length,
                add_special_tokens=True,
                truncation=True,
            )
            output.append(tokenized)
        return output

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text)"""
        index_start = i * self.batch_size
        index_end = (i + 1) * self.batch_size

        texts = []
        labels = []
        with open(self.path) as f:
            _ = next(f)  # Skip header
            for j, line in enumerate(f):
                if index_start <= j < index_end:
                    fields = line.split(",", 4)

                    sentiment = fields[3]
                    if sentiment == "positive":
                        label = 1
                    elif sentiment == "negative":
                        label = -1
                    else:
                        label = 0

                    texts.append(fields[4].strip())
                    labels.append(label)

                if j >= index_end:
                    break

        return texts, labels

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels

def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    mask = [[1 if token != 0 else 0 for token in seq] for seq in padded]
    return mask

def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    # Convert tokens to tensor
    tokens_tensor = torch.tensor(tokens)

    # Create attention mask
    mask = attention_mask(tokens)
    mask_tensor = torch.tensor(mask)

    with torch.no_grad():
        last_hidden_states = model(tokens_tensor, attention_mask=mask_tensor)

    # Embeddings for [CLS]-tokens
    features = last_hidden_states[0][:, 0, :].tolist()
    return features
    
def evaluate(model, embeddings, labels, cv: int = 5) -> List[float]:
    """Evaluate model on embeddings and labels"""
    # Cross-validation
    kf = KFold(n_splits=cv, shuffle=False)
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    losses = []

    for train_index, test_index in kf.split(embeddings):
        # Split data
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict_proba(X_test)
        loss = log_loss(y_test, y_pred)
        losses.append(loss)

    return losses

# Example
from transformers import DistilBertModel, DistilBertTokenizer

MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
bert = DistilBertModel.from_pretrained(MODEL_NAME)

loader = DataLoader('BERT.csv', tokenizer, max_limit=128, padding='batch')

for tokens, labels in loader:
    embeddings = review_embedding(tokens, bert)
    print(embeddings)

model = LogisticRegression(max_iter=1000)  # Initialization of the logistic regression model
loss_scores = evaluate(model, embeddings, labels, cv=5)
print("Cross-Entropy Loss for each fold:", loss_scores)
