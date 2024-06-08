from transformers import pipeline
import pandas as pd
from tqdm import tqdm


# Ładdowanie
def load_data():
    train_dataset = pd.read_csv("train/train.tsv", sep="\t", header=None, names=["Label", "Doc"])
    dev_0_dataset = pd.read_csv("dev-0/in.tsv", sep="\t", header=None, names=["Doc"])
    test_A_dataset = pd.read_csv("test-A/in.tsv", sep="\t", header=None, names=["Doc"])
    return train_dataset, dev_0_dataset, test_A_dataset


train_dataset, dev_0_dataset, test_A_dataset = load_data()

def extract_sentences(dataset):
    sentences = []
    for doc in dataset["Doc"]:
        sentences.append(doc.split())
    return sentences


train_sentences = extract_sentences(train_dataset)
dev_sentences = extract_sentences(dev_0_dataset)
test_sentences = extract_sentences(test_A_dataset)

# NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")


# Predykcja
def predict_ner_tags(sentences):
    ner_tags = []
    for sentence in tqdm(sentences, desc="Processing sentences"):
        ner_result = ner_pipeline(" ".join(sentence))
        tags = ["O"] * len(sentence)

        for entity in ner_result:
            entity_type = entity['entity_group']
            start_idx, end_idx = entity['start'], entity['end']

            token_start_idx = None
            token_end_idx = None
            char_idx = 0
            for i, token in enumerate(sentence):
                token_start = char_idx
                token_end = char_idx + len(token)
                char_idx = token_end + 1

                if token_start <= start_idx < token_end:
                    token_start_idx = i
                if token_start < end_idx <= token_end:
                    token_end_idx = i
                    break

            if token_start_idx is not None and token_end_idx is not None:
                tags[token_start_idx] = f"B-{entity_type}"
                for i in range(token_start_idx + 1, token_end_idx + 1):
                    tags[i] = f"I-{entity_type}"

        ner_tags.append(tags)
    return ner_tags


# Odpalanie predykcji dla wszystkich datasetów
train_ner_tags = predict_ner_tags(train_sentences)
dev_ner_tags = predict_ner_tags(dev_sentences)
test_ner_tags = predict_ner_tags(test_sentences)


# Zapis wyników
def save_predictions(tags, file_path):
    with open(file_path, 'w') as f:
        for tag_sequence in tags:
            f.write(" ".join(tag_sequence) + "\n")


save_predictions(train_ner_tags, "train/out.tsv")
save_predictions(dev_ner_tags, "dev-0/out.tsv") # geval wyszedł 0.78769
save_predictions(test_ner_tags, "test-A/out.tsv")
