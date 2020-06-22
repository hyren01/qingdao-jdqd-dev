import re
import os

src = 'file'
dst = 'articles'

src_txts = os.listdir(src)
dst_txts = os.listdir(dst)

src_contents = []
dst_contents = []

for st in src_txts:
    with open(os.path.join(src, st), 'r', encoding='utf-8') as f:
        src_contents.append(f.read())


for dt in dst_txts:
    with open(os.path.join(dst, dt), 'r', encoding='utf-8') as f:
        dst_contents.append(f.read())

unique_sts = []
for st, sc in zip(src_txts, src_contents):
    for dt, dc in zip(dst_txts, dst_contents):
        if sc == dc:
            break
    unique_sts.append(st)

dst_nums = [int(dt.split('.')[0]) for dt in dst_txts]
src_nums = [int(st.split('.')[0]) for st in src_txts]

dst_max_num = max(dst_nums) + 1

import shutil

for i, st in enumerate(unique_sts):
    shutil.move(f'{src}/{st}', f'{dst}/{dst_max_num + i}.txt')

