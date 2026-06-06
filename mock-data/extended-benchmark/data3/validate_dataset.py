import json, sys, re
from pathlib import Path
from collections import Counter

def approx_tokens(text):
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\sA-Za-z0-9_\u4e00-\u9fff]", text))

def load(path):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            d=json.loads(line)
            assert set(d.keys()) == {'text','label'}, (path,line_no,d.keys())
            assert isinstance(d['text'],str) and isinstance(d['label'],str)
            rows.append(d)
    return rows

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('.')
for task_dir in sorted((root/'tasks').iterdir()):
    tr=load(task_dir/'train.jsonl')
    te=load(task_dir/'test.jsonl')
    tr_l=set(r['label'] for r in tr); te_l=set(r['label'] for r in te)
    assert te_l <= tr_l, task_dir
    assert len([r['text'] for r in tr]) == len(set(r['text'] for r in tr)), task_dir
    assert len([r['text'] for r in te]) == len(set(r['text'] for r in te)), task_dir
    print(task_dir.name, 'OK', 'train',len(tr),'test',len(te),'labels',len(tr_l),'max_tokens',max(approx_tokens(r['text']) for r in tr+te))
print('ALL OK')
