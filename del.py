import torch
#from torchnlp.metrics import get_moses_multi_bleu
from nltk.translate.bleu_score import sentence_bleu

from transformers import T5TokenizerFast
tokenizer = T5TokenizerFast.from_pretrained("T5ForMaskedLM/t5-base")
from nltk.translate.bleu_score import SmoothingFunction
smoother = SmoothingFunction()

def train_bleu(score, label, dataset):
    total_bleu = 0
    length = len(label)
    references = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    hypotheses = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]
    print("======")
    print(hypotheses)
    print("------")
    print(references)
    print("======")
    total_bleu = get_moses_multi_bleu(hypotheses, references, lowercase=True)
    if total_bleu == None:
        total_bleu = 0
    result = round(float(total_bleu/length),4)

    return result



hypotheses=['Pri', 'President of', 'Thomas Jefferson', '', 'a', '-', 'The Season', 'a', 'The first', '1893', 'The ', 'MW', '', 'The Wi', 'a', '-']
references=['Anil Kumble', 'Inga Rhonda King', 'Paul', 'the Tuscan Apennines', 'seafloor spreading', '1871', 'February 28, 2018', 'two', '1969', 'Pratap Chandra Majumdar', 'Rose Royce', '900-MW', 'Chipper Jones', 'November 7, 2017', 'Koine Iwasaki', 'October 31']

#print(len(hypotheses))

for l in range(len(hypotheses)):
    #print(l)
    total_b = 0
    #print(references[l].lower().split())
    #y = references[l].lower().split()
    y = tokenizer.convert_ids_to_tokens(tokenizer.encode(references[l], add_special_tokens=False), skip_special_tokens=True)
    print(y)
    #y = [['john', 'is', 'coming', 'to', "jim's", 'party', 'with', 'marina.', 'he', 'forgot', 'to', 'tell', 'him.']]
    #print(hypotheses[l].lower().split())
    #y_ = hypotheses[l].lower().split()
    y_ = tokenizer.convert_ids_to_tokens(tokenizer.encode(hypotheses[l], add_special_tokens=False), skip_special_tokens=True)
    print(y_)
    #y_ = ['john', 'and']
    #y_ = ['john', 'is', 'coming', 'to', "jim's", 'party', 'with', 'marina.', 'he', 'forgot', 'to', 'tell', 'him.']
    b=0
    if len(y)!=0 and len(y_)!=0:
        print(y)
        print(y_)
        print("!!!!")
        b = sentence_bleu(y, y_, weights=(0.35, 0.35, 0.15, 0.15), smoothing_function=smoother.method1)
        #b = sentence_bleu(y, y_, weights=(1, 0, 0, 0))
    else:
        b = 0
    print(b)
    print(type(b))
    print("====")
    exit()



