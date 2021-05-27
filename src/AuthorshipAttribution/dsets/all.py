# this file is to build a training set that combines all of the other datasets
# basically we want all blog and imdb data combined into one dataset for pretraining, then we will train with pan

# get the imdb dataset
import imdb
import blog
import csv
import sys
import numpy as np


csv.field_size_limit(sys.maxsize)


i_dset = imdb.get_imdb_as_dict('/home/jtyo/Repos/AuthorshipAttribution/data/IMDB/imdb62.txt')
b_dset = blog.get_contrastive_training_set_blogsos('/home/jtyo/Repos/AuthorshipAttribution/data/Blog/blogsOS-train.json')

# combine and write
a = i_dset

all_keys = np.array([int(k) for k in i_dset.keys()])
max_key = np.max(all_keys)
max_key += 1

for auth, text in b_dset.items():
    a[max_key] = text
    max_key += 1

# chunk just incase
for auth, texts in a.items():
    new_texts = []
    for text in texts:
        split_text = text.split(' ')
        if len(split_text) > 500:
            for start_idx in range(0, len(split_text), 500):
                end_idx = start_idx + 500 if start_idx + 500 < len(split_text) else len(split_text) - 1
                new_texts.append(' '.join(split_text[start_idx: end_idx]))
        else:
            new_texts.append(text)
    a[auth] = new_texts

# need to make sure there are no duplicates
for auth, texts in a.items():
    a[auth] = list(set(texts))

# now write this file and call it good - write it as the blogs os and just use it to process it
save_path = '/home/jtyo/Repos/AuthorshipAttribution/data/all.csv'

print(f'saving all train dataset at {save_path}')

rows = 0
with open(save_path, 'w') as f:

    writer = csv.writer(f)
    # write header
    writer.writerow(['auth_id', 'text'])
    for auth, texts in a.items():
        for text in texts:
            writer.writerow([auth, text])
            rows += 1

print(f'done, wrote {rows} rows')
