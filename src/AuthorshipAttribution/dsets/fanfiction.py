# starting with the large dataset
# data looks like json
# try to read it, lets play a bit I assume

# dataset description:
# The train (calibration) and test datasets consists of pairs of (snippets from) two different fanfics,
# that were obtained drawn from fanfiction.net. Each pair was assigned a unique identifier and we distinguish between
# same-author pairs and different-authors pairs. Additionally, we offer metadata on the fandom (i.e. thematic category)
# for each text in the pair (note that fanfic "crossovers" were not included and only single-fandom texts were
# considered). The fandom distribution in these datasets maximally approximates the (long-tail) distribution of the
# fandoms in the original dataset. The test dataset is structured in the exact same way, but participants should expect
# a significant shift in the relation between authors and fandoms.
#
# The training dataset comes in two variants: a smaller dataset, particularly suited for symbolic machine learning
# methods and a large, dataset, suitable for applying data-hungry deep learning algorithms. Participants have to
# specify which of the two datasets was used to train their model. Models using the small set will be evaluated
# separately from models using the large set. We encourage participants to try the small dataset as a challenge,
# though participants can submit separate submissions for either one or both.
#
# Both the small and the large dataset come with two newline delimited JSON files each (*.jsonl). The first file
# contains pairs of texts (each pair has a unique ID) and their fandom labels:
#
# {"id": "6cced668-6e51-5212-873c-717f2bc91ce6", "fandoms": ["Fandom 1", "Fandom 2"], "pair": ["Text 1...", "Text 2..."]}
# {"id": "ae9297e9-2ae5-5e3f-a2ab-ef7c322f2647", "fandoms": ["Fandom 3", "Fandom 4"], "pair": ["Text 3...", "Text 4..."]}
# ...
# The second file, ending in *_truth.jsonl, contains the ground truth for all pairs. The ground truth is composed of a
# boolean flag indicating if texts in a pair are from the same author and the numeric author IDs:
#
# {"id": "6cced668-6e51-5212-873c-717f2bc91ce6", "same": true, "authors": ["1446633", "1446633"]}
# {"id": "ae9297e9-2ae5-5e3f-a2ab-ef7c322f2647", "same": false, "authors": ["1535385", "1998978"]}
# ...
# Data and ground truth are in the same order and can be ingested line-wise in parallel without the need for a
# reshuffle based on the pair ID. The fandom labels will be given in both the training and testing datasets.
# The ground truth file will only be available for the training data.

# Few things to consider:
#   1) How do we construct user embeddings as we only have text pairs?
#   2) I think just keep track of the data for each user, construct pairs manually (becomes normal problem)
#   3)

from tqdm import tqdm
import os
from torch.utils.data import Dataset
from sentence_transformers import InputExample
import json
from typing import List, Tuple
import csv
from csv import reader
import random
import sys
import numpy as np
import copy


csv.field_size_limit(sys.maxsize)


class FFDataset(Dataset):
    """Dataset for Author Verification on the IMDB62 Dataset."""

    def __init__(self,
                 data_file: str,
                 base_rate: float = 0.5
                 ):
        """
        Args:
            data_file (string): the path to the IMDB62 Dataset txt file
        """
        # get the dataset, then break it up into dict key'd on authors with values a list of texts.
        self.per_author_dataset = get_ff_contrastive_training_set(data_file)
        self.base_rate = base_rate

    def __len__(self):
        return sum([len(x) for x in self.per_author_dataset.values()])

    def __getitem__(self, idx):
        # we want this to work with contrastive, so sample on the author level
        auth1 = random.choice(list(self.per_author_dataset.keys()))

        if np.random.uniform() < self.base_rate:
            # this is a same_author sample
            # make sure the auth has multiple samples
            while len(self.per_author_dataset[auth1]) < 2:
                auth1 = random.choice(list(self.per_author_dataset.keys()))

            text1 = text2 = random.choice(self.per_author_dataset[auth1])
            # make sure the texts are different
            counter = 0
            im_confused_counter = 0
            auths_tried = 0
            while text1 == text2:
                text2 = random.choice(self.per_author_dataset[auth1])
                counter += 1
                if counter > 100:
                    # these texts are the same, get a different author
                    while len(self.per_author_dataset[auth1]) < 2:
                        auth1 = random.choice(list(self.per_author_dataset.keys()))
                    auths_tried += 1
                    text1 = text2 = random.choice(self.per_author_dataset[auth1])
                    if auths_tried > 50:
                        assert False, "we've got problems, can't find a different text from same author"
                    counter = 0
                if im_confused_counter > 10000:
                    print(auth1)
                    print(text1)
                    assert False, "we've got problems, stuck in this same-author loop again."
                im_confused_counter += 1

            label = 1

        else:
            # this is a different author sample
            auth2 = auth1
            while auth1 == auth2:
                auth2 = random.choice(list(self.per_author_dataset.keys()))
            # now get a text from both authors
            text1 = random.choice(self.per_author_dataset[auth1])
            text2 = random.choice(self.per_author_dataset[auth2])
            label = 0

        return InputExample(texts=[text1, text2], label=label)


def get_ff_contrastive_training_set(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)


def get_ff_triplet_training(data_path):
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    # now make list of input examples and return
    data = []

    for auth, texts in raw_data.items():
        for text in texts:
            data.append(InputExample(texts=[text], label=int(auth.strip())))

    return data


def get_ff_contrastive_fixed(data_file: str) -> List[InputExample]:
    # open the training and testing files.
    raw_data = get_ff_contrastive_train_or_test(data_file)
    # now make input exmples
    for i in range(len(raw_data)):
        raw_data[i] = InputExample(texts=[raw_data[i][1], raw_data[i][2]], label=int(raw_data[i][0]))

    return raw_data


def get_ff_contrastive_train_or_test(dataset_path: str) -> List[Tuple[str, str]]:
    raw_data = []
    with open(dataset_path, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if row[0] != "same":
                # skip header
                raw_data.append(InputExample(texts=[row[1], row[2]], label=int(row[0])))
    return raw_data


def get_training_for_triplet_loss_ff(data_file: str) -> List[InputExample]:
    raw_data = get_ff_train_or_test(data_file)

    # make list of InputExamples, return
    data = []
    for i in raw_data:
        data.append(InputExample(texts=[i[1]], label=int(i[0])))

    return data


def get_ff_train_or_test(dataset_path: str, as_dict: bool = False) -> List[Tuple[str, str]]:

    raw_data = {}

    with open(dataset_path, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            auth_id = row[0]
            text = row[1]
            if auth_id not in raw_data.keys():
                raw_data[auth_id] = []
            raw_data[auth_id].append(text)

    if as_dict:

        return raw_data

    else:  # return list of (auth_id, text) pairs

        data = []

        for key, val in raw_data.items():
            for v in val:
                data.append((key, v))

        return data


def build_per_author_dataset(data_file: str, ground_truth_file: str) -> dict:

    # make auth_id's incrementing
    auth_id_transformer = {}
    counter = 0

    # first get the ground truth - dict key'd on id
    labels = {}
    with open(ground_truth_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data_point = json.loads(line)
            data_id = data_point['id']
            same = data_point['same']
            auths = data_point['authors']

            if data_id not in labels:
                labels[data_id] = []

            labels[data_id].append((same, auths))

    raw_data = {}

    with open(data_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            data_point = json.loads(line)
            data_id = data_point['id']
            fandoms = data_point['fandoms']
            text_pair = data_point['pair']

            # for raw_data - just have auth id, [text1, text2, . . . ]
            auth1 = labels[data_id][1][0]
            text1 = text_pair[0]
            auth2 = labels[data_id][1][1]
            text2 = text_pair[1]

            if auth1 not in auth_id_transformer.keys():
                auth_id_transformer[auth1] = counter
                counter += 1
            if auth2 not in auth_id_transformer.keys():
                auth_id_transformer[auth2] = counter
                counter += 1

            auth1 = auth_id_transformer[auth1]
            auth2 = auth_id_transformer[auth2]

            if auth1 not in raw_data.keys():
                raw_data[auth1] = []
            if auth2 not in raw_data.keys():
                raw_data[auth2] = []

            raw_data[auth1].append(text1)
            raw_data[auth2].append(text2)

    # this is a dict keyd on author with values being a list of texts
    return raw_data


def get_input_example_list(raw_data):
    data = []

    for key, val in raw_data.items():
        for v in val:
            data.append(InputExample(texts=[v], label=int(key)))

    return data


if __name__ == '__main__':

    import argparse
    import csv

    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of IMDB dataset')

    parser.add_argument('--dataset_path', metavar='dataset_path', type=str,
                        default='/home/jtyo/Repos/AuthorshipAttribution/data/PAN20/',
                        help='Which dataset to use.')
    parser.add_argument('--dataset_save_path', metavar='dataset_save_path', type=str,
                        default='/home/jtyo/Repos/AuthorshipAttribution/data/PAN20/',
                        help='Where to save new training and testing dataset')
    parser.add_argument('--what_for', metavar='what_for', type=str, default='sbert',
                        help='set to sbert, bertaa, or contrastive')
    parser.add_argument('--small_dset', action='store_true',
                        help='set this to do the processing on the small dataset.')
    parser.add_argument('--chunk', action='store_true',
                        help='set this flag to chunk all texts longer than 500 words into subtexts of 500 words')

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    # need to make a version of this for contrastive loss - very easy, just use the pairs provided.

    print('getting truth file')

    which_dataset = 'small' if args.small_dset else 'large'

    print('using the ' + which_dataset + ' dataset')

    truth_id_to_auth = {}
    dset_truth_path = os.path.join(args.dataset_path, 'pan20-authorship-verification-training-'+which_dataset+'/pan20-authorship-verification-training-'+which_dataset+'-truth.jsonl')
    with open(dset_truth_path, 'r') as f:

        lines = f.readlines()

        for line in lines:
            line = json.loads(line)
            truth_id_to_auth[line['id']] = line['authors']

    print('building the dataset for use with contrastive loss')

    # read the files and create a train/test split
    data = {}
    dset_path = os.path.join(args.dataset_path, 'pan20-authorship-verification-training-'+which_dataset+'/pan20-authorship-verification-training-'+which_dataset+'.jsonl')

    raw_data = []
    # auth to raw_data idx
    auth_to_raw_data_idx = {}
    dropped_texts = 0
    with open(dset_path, 'r') as f:

        lines = f.readlines()

        for line in tqdm(lines):
            line = json.loads(line)
            auth0 = truth_id_to_auth[line['id']][0]
            text0 = line['pair'][0]
            auth1 = truth_id_to_auth[line['id']][1]
            text1 = line['pair'][1]

            raw_data.append([auth0, auth1, text0, text1])
            if auth0 not in auth_to_raw_data_idx.keys():
                auth_to_raw_data_idx[auth0] = []
            if auth1 not in auth_to_raw_data_idx.keys():
                auth_to_raw_data_idx[auth1] = []
            # keep track of which idx's from the raw data list each author has text in.
            # use this for building testing set later
            auth_to_raw_data_idx[auth0].append(len(raw_data)-1)
            auth_to_raw_data_idx[auth1].append(len(raw_data)-1)

            for a, t in zip([auth0, auth1], [text0, text1]):
                # this will be slow, but doing now will save some headache
                if a not in data.keys():
                    data[a] = [t]
                else:
                    data[a].append(t)

    # get rid of duplicates
    for auth, texts in data.items():
        data[auth] = list(set(texts))

    print('splitting data')
    all_auths = list(data.keys())
    random.shuffle(all_auths)

    split_point = int(0.95*len(all_auths))
    train_auths = all_auths[:split_point]
    test_auths = all_auths[split_point:]

    print(f'There are {len(train_auths)} train authors and {len(test_auths)} test authors')

    train_data = {}

    # make sure auth_id's are increasing from 0
    auth_id_normalizer = {}
    counter = 0
    for auth in train_auths:
        if auth not in auth_id_normalizer.keys():
            auth_id_normalizer[auth] = counter
            counter += 1
        train_data[auth_id_normalizer[auth]] = data[auth]

    train_data_nc = copy.deepcopy(train_data)

    # chunk if necessary, but only training data
    if args.chunk:
        print('before chunk:')
        num_texts = sum([len(x) for x in train_data.values()])
        num_auths = sum([1 for x in train_data.keys()])
        all_texts, total_len = [], 0
        for texts in train_data.values():
            for text in texts:
                total_len += len(text)
        avg_txt_len = total_len / num_texts
        print(f'\tnum_texts: {num_texts}')
        print(f'\tnum_auths: {num_auths}')
        print(f'\ttext_len: {total_len}')
        print(f'\tavg_len: {avg_txt_len}')
        print('chunking')
        for auth, texts in tqdm(train_data.items()):
            new_texts = []
            for text in texts:
                split_text = text.split(' ')
                if len(split_text) > 500:
                    for start_idx in range(0, len(split_text) - 1, 500):
                        end_idx = start_idx + 500 if start_idx + 500 < len(split_text) - 1 else len(split_text) - 1
                        new_texts.append(' '.join(split_text[start_idx: end_idx]))
                else:
                    new_texts.append(text)
            train_data[auth] = new_texts
        print('after chunk:')
        num_texts = sum([len(x) for x in train_data.values()])
        num_auths = sum([1 for x in train_data.keys()])
        all_texts, total_len = [], 0
        for texts in train_data.values():
            for text in texts:
                total_len += len(text)
        avg_txt_len = total_len / num_texts
        print(f'\tnum_texts: {num_texts}')
        print(f'\tnum_auths: {num_auths}')
        print(f'\ttext_len: {total_len}')
        print(f'\tavg_len: {avg_txt_len}')

    # make sure things add up
    assert len(train_data.keys()) == len(train_auths)

    num_train_texts = sum([len(x) for x in train_data.values()])
    print(f'Thee training dataset consists of {len(list(train_data.keys()))} authors and {num_train_texts} texts')

    test_data = []  # need pairs here
    skipped_same_pairs = 0
    for auth in test_auths:
        # add all pairs from that auth to the test data
        idxs = auth_to_raw_data_idx[auth]
        for locs in idxs:
            label = 1 if raw_data[locs][0] == raw_data[locs][1] else 0
            test_data.append([label, raw_data[locs][2], raw_data[locs][3]])

    # remove duplicates
    test_data = list(set(test_data))
    # this completes the dataset, print some stats
    print(f'test data contains {len(test_data)} datapoints.')

    tmp = [x[0] for x in test_data]
    same_pair_percent = sum(tmp)/len(tmp)
    print(f'the percentage of same_pairs is {same_pair_percent}')
    # want a 50-50 split between same and diff auth pairs
    # get all text pairs from the original pairs that were written by those authors. This is the test set.

    # if len(data[auth]) > 2:
    #     # create a same auth pair
    #     text1 = text2 = random.choice(data[auth])
    #     counter = 0
    #     while text1 == text2:
    #         text2 = random.choice(data[auth])
    #         counter += 1
    #         if counter > 100:
    #             # we have encountered an author with multiple texts, but they are actually the same
    #             # just skip the author and remove one of the texts
    #             assert False, 'we encountered an author with duplicate texts, but this should not happen per prior filtering. . . '
    #     test_data.append((1, text1, text2))
    # else:
    #     skipped_same_pairs += 1

    def get_an_auth_for_same_pair():
        counter = 0
        auth = random.choice(test_auths)
        while len(data[auth]) < 2:
            auth = random.choice(test_auths)
            counter += 1
            if counter > 100:
                assert False, 'there is an issue finding an author with two or more texts?'
        return auth

    # # add back in the skipped pairs
    # print(f'we want a total of {len(test_data) + skipped_same_pairs}, still need {skipped_same_pairs}')
    # print('adding back in skipped pairs')
    # for i in tqdm(range(skipped_same_pairs)):
    #     auth = get_an_auth_for_same_pair()
    #     # get two texts
    #     text1 = text2 = random.choice(data[auth])
    #     counter = 0
    #     while text1 == text2 or ((1, text1, text2) in test_data):
    #         text1 = random.choice(data[auth])
    #         text2 = random.choice(data[auth])
    #         counter += 1
    #         if counter > 100:
    #             auth = get_an_auth_for_same_pair()
    #             counter = 0
    #     test_data.append((1, text1, text2))
    #
    # # now repeat, but get different author pairs
    # for auth in test_auths:
    #     auth2 = auth
    #     counter = 0
    #     text1 = text2 = None
    #     while auth == auth2 or (0, text1, text2) in test_data:
    #         auth2 = random.choice(test_auths)
    #         counter += 1
    #         if counter > 100:
    #             assert False, "you've got problems breh"
    #     text1 = random.choice(data[auth])
    #     text2 = random.choice(data[auth2])
    #     test_data.append((0, text1, text2))


    print(f'writing training file')
    # save the files
    with open(os.path.join(args.dataset_save_path, 'pan20-authorship-verification-training-' + which_dataset + '/pan20_train.json'), 'w') as f:
        json.dump(train_data, f, sort_keys=True, indent=4)

    with open(os.path.join(args.dataset_save_path, 'pan20-authorship-verification-training-' + which_dataset + '/pan20_train_nc.json'), 'w') as f:
        json.dump(train_data_nc, f, sort_keys=True, indent=4)

    print(f'writing testing file')
    with open(os.path.join(args.dataset_save_path, 'pan20-authorship-verification-training-' + which_dataset + '/pan20_test.csv'), 'w') as f:

        # create the csv writer
        writer = csv.writer(f)

        # write header
        writer.writerow(["same", "text1", "text2"])
        # write contents
        for same, text1, text2 in test_data:
            writer.writerow([same, text1, text2])
