import json
import os

path = '/data3/private/xcj/LegalBert/eval_data/ms_ac'
fnames = os.listdir(path)
fnames.sort()
label2num = {}
for fn in fnames:#[int(len(fnames) * 0.7) : ]:
    data = json.load(open(os.path.join(path, fn), 'r'))
    for doc in data:
        if len(doc['SS']) < 10 or len(doc['AJAY']) == 0:
            continue
        for ay in doc['AJAY']:
            if ay not in label2num:
                label2num[ay] = 0
            label2num[ay] += 1

label2num = {key: label2num[key] for key in label2num if label2num[key] >= 60}
label2num = dict(sorted(label2num.items(), reverse=True, key=lambda x:x[1]))
fout = open('ca_label2num.json', 'w')
print(json.dumps(label2num, ensure_ascii=False, indent=2), file=fout)
fout.close()
'''
label2id = {}
for key in label2num:
    if label2num[key] < 5000:
        label2id[key] = len(label2id)
fout = open('label2id.json', 'w')
print(json.dumps(label2id, ensure_ascii=False, indent=2), file=fout)
fout.close()
'''