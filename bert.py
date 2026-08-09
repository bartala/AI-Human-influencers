from transformers import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Assume you have a list of comment texts called comment_texts
def bert_embed(texts, tokenizer, model, max_len=64):
    embeddings = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token representation
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_vec)
    return np.array(embeddings)

bert_features = bert_embed(X_text_column, tokenizer, model)


from sklearn.linear_model import LogisticRegressionCV
clf_bert = LogisticRegressionCV(cv=10, max_iter=2000, n_jobs=-1)
clf_bert.fit(bert_features, y)
