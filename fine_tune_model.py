from transformers import  AutoTokenizer, BertForSequenceClassification
import torch
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from ultis import *
import underthesea

MAX_LEN = 160
sw = None#load_stopwords()
model_save = BertForSequenceClassification.from_pretrained('tam18902/sentimentanalysis')
tokenizer = AutoTokenizer.from_pretrained('tam18902/sentimentanalysis')
def word_to_tokenize(text, sw):
    for id, line in enumerate(text):
        """
        line = underthesea.word_tokenize(line)
        # Lọc các từ vô nghĩa
        filtered_sentence = [w for w in line if not w in sw]
        # Ghép lại thành câu như cũ sau khi lọc
        line = " ".join(filtered_sentence)
        """
        text[id] = underthesea.word_tokenize(line, format="text")

def text_token(sentences, tokenizer):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    text = [standardize_data(t) for t in sentences]
    word_to_tokenize(text, sw)
    # For every sentence...
    for id, sent in enumerate(sentences):
        encoded_sent = tokenizer.encode(sentences[id],add_special_tokens = True)
        input_ids.append(encoded_sent)
    return input_ids

def predict(sentences):
    input_ids = text_token(sentences, tokenizer)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=tokenizer.pad_token_id, truncating="post", padding="post")
    attention_masks = np.where(input_ids == 1, 0, 1)
    p_inputs = torch.tensor(input_ids)
    p_masks = torch.tensor(attention_masks)
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model_save(p_inputs, token_type_ids=None,
                        attention_mask=p_masks)
    logits = outputs[0].numpy()
    softmax = torch.nn.Softmax(dim=1)
    prob = np.array(softmax(torch.tensor(logits)).numpy(), dtype=float)
    return prob