from datasets import load_dataset
from openprompt.data_utils.utils import InputExample


class QNLI:
    labels = [0, 1]
    label_words = {
        0: "no",
        1: "yes",
    }
    label_mapping = {0: 1, 1: 0}
    
    def __init__(self):
        data = load_dataset('glue', 'qnli')
        self.train_dataset = [
            InputExample(guid=e['idx'], text_a=e['question'], text_b=e['sentence'], label=self.label_mapping[e['label']]) 
            for e in data['train']]

        self.eval_dataset = [
            InputExample(guid=e['idx'], text_a=e['question'], text_b=e['sentence'], label=self.label_mapping[e['label']])
            for e in data['validation']]

class MNLI:
    labels = [0, 1, 2]
    label_words = {
        0: "no",
        1: "yes",
        2: "neutral"
    }
    label_mapping = {0: 1, 1: 2, 2: 0}

    def __init__(self):
        data = load_dataset('glue', 'mnli')
        self.train_dataset = [
            InputExample(guid=e['idx'], text_a=e['premise'], text_b=e['hypothesis'], label=self.label_mapping[e['label']]) 
            for e in data['train']]

        self.eval_dataset = [
            InputExample(guid=e['idx'], text_a=e['premise'], text_b=e['hypothesis'], label=self.label_mapping[e['label']])
            for e in data['validation_mismatched']]

class SNLI:
    labels = [0, 1, 2]
    label_words = {
        0: "no",
        1: "yes",
        2: "neutral"
    }
    label_mapping = {0: 1, 1: 2, 2: 0}
    
    def __init__(self):
        data = load_dataset('snli')
        self.train_dataset = [
            InputExample(guid=i, text_a=e['premise'], text_b=e['hypothesis'], label=self.label_mapping[e['label']]) 
            for i, e in enumerate(data['train']) if e['label'] != -1]

        self.eval_dataset = [
            InputExample(guid=i, text_a=e['premise'], text_b=e['hypothesis'], label=self.label_mapping[e['label']])
            for i, e in enumerate(data['validation']) if e['label'] != -1]

class SST2:
    labels = [0, 1]
    label_words = {
        0: "negative",
        1: "positive",
    }
    label_mapping = {0: 1, 1: 2, 2: 0}
    
    def __init__(self):
        data = load_dataset('glue', 'sst2')
        self.train_dataset = [
            InputExample(guid=i, text_a=e['sentence'], label=e['label']) 
            for i, e in enumerate(data['train'])]

        self.eval_dataset = [
            InputExample(guid=i, text_a=e['sentence'], label=e['label'])
            for i, e in enumerate(data['validation'])]

class RottenTomatoes:
    labels = [0, 1]
    label_words = {
        0: "negative",
        1: "positive",
    }
    label_mapping = {0: 1, 1: 2, 2: 0}
    
    def __init__(self):
        data = load_dataset('rotten_tomatoes')
        self.train_dataset = [
            InputExample(guid=i, text_a=e['text'], label=e['label']) 
            for i, e in enumerate(data['train'])]

        self.eval_dataset = [
            InputExample(guid=i, text_a=e['text'], label=e['label'])
            for i, e in enumerate(data['validation'])]

data_processor_list = {
    'sst2': SST2,
    'rotten_tomatoes': RottenTomatoes,
    'qnli': QNLI,
    'mnli': MNLI,
    'snli': SNLI,
}
