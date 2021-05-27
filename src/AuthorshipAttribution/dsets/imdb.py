import random, os
import numpy as np
from torch.utils.data import Dataset
from sentence_transformers import InputExample
import csv
from typing import List, Tuple, Optional, Union
import copy


class IMDB62AvDataset(Dataset):
    """Dataset for Author Verification on the IMDB62 Dataset."""

    def __init__(self,
                 data_file: str,
                 dataset_size: int = None,
                 use_cache: bool = True):
        """
        Args:
            data_file (string): the path to the IMDB62 Dataset txt file
        """
        self.data_file = data_file
        self.dataset_size = dataset_size

        # get data file
        raw_data = get_imdb_train_or_test(self.data_file)

        # now process the individual files into an actual dataset
        # add all data to a new object, then sample negative pairs to reach self.dataset_size
        self.data = transform_data_classification_to_siamese(raw_data, self.dataset_size)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IMDB62ContrastiveDataset(Dataset):
    """Dataset for Author Verification on the IMDB62 Dataset."""

    def __init__(self,
                 data_file: str,
                 base_rate: float = 0.2
                 ):
        """
        Args:
            data_file (string): the path to the IMDB62 Dataset txt file
        """
        self.data_file = data_file
        self.base_rate = base_rate

        # get data file
        self.data = get_imdb_train_or_test(self.data_file, as_dict=True)

    def __len__(self):
        return sum([len(x) for x in self.data.values()])

    def __getitem__(self, idx):
        # just sample randomly every time, with the desired precentage of same/diff auth
        # draw the first author
        auth1 = random.choice(list(self.data.keys()))
        text1 = random.choice(self.data[auth1])
        if np.random.uniform() <= self.base_rate:
            # this needs to be a same author pair
            text2 = None
            while text1 == text2 or text2 is None:
                text2 = random.choice(self.data[auth1])
            label = 1
        else:
            auth2 = random.choice(list(self.data.keys()))
            text2 = random.choice(self.data[auth2])
            label = 0
        return InputExample(texts=[text1, text2], label=label)


def get_training_for_contrastive_loss_imdb(data_file: str) -> List[InputExample]:
    raw_data = get_imdb_contrastive_train_file(data_file)

    # now make list of input examples
    for i in range(len(raw_data)):
        tmp = raw_data[i]
        raw_data[i] = InputExample(texts=[tmp[1], tmp[2]], label=int(tmp[0]))

    return raw_data


def get_imdb_contrastive_train_file(data_file: str) -> List:
    raw_data = []
    with open(data_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[0] == "same":
                # skip header
                continue
            raw_data.append([row[0], row[1], row[2]])
    return raw_data

def get_training_for_triplet_loss_imdb(data_file: str) -> List[InputExample]:

    raw_data = get_imdb_train_or_test(data_file)

    # make list of InputExamples, return
    data = []
    for i in raw_data:
        data.append(i)  # InputExample(texts=[i[1]], label=int(i[0])))

    return data


def transform_data_classification_to_siamese(raw_data: List[Tuple[str, str]], dataset_size: int,
                                             output_type: str = 'InputExample') \
        -> List[Union[Tuple[int, str], InputExample]]:

    data = []

    for i in range(int(len(raw_data) / 2)):
        # get next two datapoints, and determine label based off of them being the same author or not
        dp1 = raw_data[2 * i]
        dp2 = raw_data[2 * i + 1]
        label = float(1) if dp1[0] == dp2[0] else float(0)

        # wrap data in proper output class
        if output_type == 'InputExample':
            data.append(InputExample(texts=[dp1[1], dp2[1]], label=label))
        elif output_type == 'str':
            data.append((dp1[1], dp2[1], label))
        else:
            assert False, 'transform_data_classification_to_siamese output type not known'

    # now randomly sample to increase dataset size - there are 2^62,000 combinations, so just randomly sample
    for i in range(dataset_size - len(data)):

        dp1 = random.choice(raw_data)
        dp2 = random.choice(raw_data)

        # make sure points aren't the same
        while (dp1[0] == dp2[0] and dp1[1] == dp2[1]):
            dp1 = random.choice(raw_data)
            dp2 = random.choice(raw_data)

        # add to the dataset
        label = float(1) if dp1[0] == dp2[0] else float(0)
        # wrap data in proper output class
        if output_type == 'InputExample':
            data.append(InputExample(texts=[dp1[1], dp2[1]], label=label))
        elif output_type == 'str':
            data.append((dp1[1], dp2[1], label))
        else:
            assert False, 'transform_data_classification_to_siamese output type not known'
    
    return data 


def get_imdb_test_for_sbert_evaluator(dataset_path: str, dataset_size: int) -> \
        List[Union[Tuple[int, str], InputExample]]:
    
    sbert_evaluation_data = transform_data_classification_to_siamese(dataset_path, dataset_size, 'str')
    
    return sbert_evaluation_data


def get_imdb_train_or_test(dataset_path: str, as_dict: bool = False) -> List[Tuple[str, str]]:

    data = {} if as_dict else []

    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            if as_dict:
                if line[0] in data.keys():
                    data[line[0]].append(line[1])
                else:
                    data[line[0]] = [line[1]]
            else:
                data.append((line[0], line[1]))

    return data


def get_imdb_as_dict(dataset_path: str) -> dict:

    # get the dataset as a dict, with key being author id and value being a list of contents
    raw_data = {}

    # transform user_id's as well
    label_transformer = {}
    label_count = 0

    with open(dataset_path, 'r') as f:

        lines = f.readlines()

        for line in lines:

            line = line.split('\t')

            # we want to make sure that splitting on tab is the right thing, so check len of resultant object
            assert len(line) == 6, 'The split line, from the imdb62 dataset, has not given in the right num of ojbects'

            user_id = line[1]
            text = line[-1]

            if user_id not in label_transformer.keys():
                label_transformer[user_id] = label_count
                label_count += 1

            # change user_id to incrementing int
            user_id = str(label_transformer[user_id])

            if user_id not in raw_data.keys():
                raw_data[user_id] = [text]
            else:
                raw_data[user_id].append(text)

    return raw_data


if __name__ == "__main__":
    # we want to split up the IMDB62 dataset into a train/test split
    # exactly how this is done differs widely, so we will just go with the classic 80/20 random split.
    # for each author, use 800 posts for training, and 200 for testing.
    import argparse

    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of IMDB dataset')

    parser.add_argument('--dataset_path', metavar='dataset_path', type=str,
                        help='Which dataset to use.')
    parser.add_argument('--dataset_save_path', metavar='dataset_save_path', type=str,
                        help='Where to save new training and testing dataset')
    parser.add_argument('--what_for', metavar='what_for', type=str, default='triplet',
                        help='set to contrastive to build dataset for contrastive loss, or triplet for triplet loss')
    parser.add_argument('--chunk', action='store_true',
                        help='set this flag to chunk examples into lengths of 500 words')

    args = parser.parse_args()

    print(f'getting original dataset at {args.dataset_path}')
    # read the csv file, and becasue the dataset isn't very big, keep everything in memory
    data = get_imdb_as_dict(args.dataset_path)

    print('splitting dataset')
    # now split this into a train and test dict (for simplicity, just take first 800 training, last 200 test
    training_data, test_data = {}, {}
    for k, v in data.items():
        training_data[k] = v[:800]
        test_data[k] = v[800:]

    training_data_nc = copy.deepcopy(training_data)

    # now chunk training if necessary
    if args.chunk:
        print('before chunk:')
        num_texts = sum([len(x) for x in training_data.values()])
        num_auths = sum([1 for x in training_data.keys()])
        all_texts, total_len = [], 0
        for texts in training_data.values():
            for text in texts:
                total_len += len(text)
        avg_txt_len = total_len / num_texts
        print(f'\tnum_texts: {num_texts}')
        print(f'\tnum_auths: {num_auths}')
        print(f'\ttext_len: {total_len}')
        print(f'\tavg_len: {avg_txt_len}')

        for auth, texts in training_data.items():
            new_texts = []
            for text in texts:
                split_text = text.split(' ')
                if len(split_text) > 500:
                    for start_idx in range(0, len(split_text), 500):
                        end_idx = start_idx + 500 if start_idx + 500 < len(split_text) else len(split_text) - 1
                        new_texts.append(' '.join(split_text[start_idx: end_idx]))
                else:
                    new_texts.append(text)
            training_data[auth] = new_texts

        print('after chunk:')
        num_texts = sum([len(x) for x in training_data.values()])
        num_auths = sum([1 for x in training_data.keys()])
        all_texts, total_len = [], 0
        for texts in training_data.values():
            for text in texts:
                total_len += len(text)
        avg_txt_len = total_len / num_texts
        print(f'\tnum_texts: {num_texts}')
        print(f'\tnum_auths: {num_auths}')
        print(f'\ttext_len: {total_len}')
        print(f'\tavg_len: {avg_txt_len}')

    # now write this to a new file, imdb62-train.tsv and imdb62-test.tsv
    training_save_path = os.path.join(args.dataset_save_path, 'imdb62-train.txt')
    training_nc_save_path = os.path.join(args.dataset_save_path, 'imdb62-train-nc.txt')
    testing_save_path = os.path.join(args.dataset_save_path, 'imdb62-test.txt')

    if args.what_for == 'contrastive':

        # need to make a bunch of pairs somehow?
        # form these pairs into same, text1, text2
        # how do we do this in a reasonable manor?
        # but we can still use the AaEvaluator to see how it does on a non split train set?

        # we have a dict, key'd on authors, and values is a list of texts, For this, I'll just create a bunch of tuples and start sampling? till I hit some desired number?
        # say we will select about .5 to be same author, and .5 to be different author?
        # but this is not really good in terms of base rate if we evaluate with the normal metrics?
        # actually randomly sampling should give an approximately 1/62 being same author, so should be good? but hten just predict not same and get pretty high accuracy
        # however, because this is a closed set identification task, I don't want to change evaluation, so I will just
        # pick a middle ground between 1/62 and 50%, say 25% being same author pairs, and run with it
        len_training_data = sum([len(v) for k, v in training_data.items()])

        # how many samples do we want? A bit hard to tell, but lets say 4x the dataset size for now
        num_desired_pairs = 4*len_training_data
        # 25% being same author pairs
        num_same_pairs = int(0.25*num_desired_pairs)
        num_diff_paris = num_desired_pairs - num_same_pairs

        training_pairs = []
        for i in range(num_same_pairs):
            auth = random.sample(training_data.keys(), 1)[0]
            # now sample two texts from that auth
            texts = random.sample(training_data[auth], 2)
            training_pairs.append([1, texts[0], texts[1]])

        # now add different pairs
        auth1 = auth2 = None
        for i in range(num_diff_paris):
            while auth1 == auth2:
                auth1 = random.sample(training_data.keys(), 1)[0]
                auth2 = random.sample(training_data.keys(), 1)[0]
            text1 = random.sample(training_data[auth1], 1)[0]
            text2 = random.sample(training_data[auth2], 1)[0]
            training_pairs.append([0, text1, text2])

        # now write this training file
        training_save_path = os.path.join(args.dataset_save_path, 'imdb62-contrastive-train.txt')
        testing_save_path = os.path.join(args.dataset_save_path, 'imdb62-contrastive-test.txt')

        print(f'saving new contrastive training dataset at {training_save_path}')
        with open(training_save_path, 'w') as f:

            writer = csv.writer(f)
            # write header
            writer.writerow(['same', 'text1', 'text2'])
            for dp in training_pairs:
                writer.writerow(dp)

        # do the same for testing
        len_test_data = sum([len(v) for k, v in test_data.items()])

        # how many samples do we want? A bit hard to tell, but lets say 4x the dataset size for now
        num_desired_pairs = 4 * len_test_data
        # 25% being same author pairs
        num_same_pairs = int(0.25 * num_desired_pairs)
        num_diff_paris = num_desired_pairs - num_same_pairs

        testing_pairs = []
        for i in range(num_same_pairs):
            auth = random.sample(test_data.keys(), 1)[0]
            # now sample two texts from that auth
            texts = random.sample(test_data[auth], 2)
            testing_pairs.append([1, texts[0], texts[1]])

        # now add different pairs
        auth1 = auth2 = None
        for i in range(num_diff_paris):
            while auth1 == auth2:
                auth1 = random.sample(test_data.keys(), 1)[0]
                auth2 = random.sample(test_data.keys(), 1)[0]
            text1 = random.sample(test_data[auth1], 1)[0]
            text2 = random.sample(test_data[auth2], 1)[0]
            testing_pairs.append([0, text1, text2])

        print(f'saving new contrastive test dataset at {testing_save_path}')
        with open(testing_save_path, 'w') as f:

            writer = csv.writer(f)
            # write header
            writer.writerow(['same', 'text1', 'text2'])
            for dp in testing_pairs:
                writer.writerow(dp)

    else:

        training_rows_written = 0

        print(f'saving new training dataset at {training_save_path}')
        with open(training_save_path, 'w') as f:

            for k, v in training_data.items():
                for content in v:
                    f.write(k + '\t' + content)
                    training_rows_written += 1

        print(f'saving new training dataset at {training_nc_save_path}')
        with open(training_nc_save_path, 'w') as f:

            for k, v in training_data_nc.items():
                for content in v:
                    f.write(k + '\t' + content)

        # this no longer holds (i.e. chunking)
        # assert training_rows_written == 49_600, f'There were {training_rows_written} rows written, should have been 49,600.'

        testing_rows_written = 0

        print(f'saving new testing dataset at {testing_save_path}')
        with open(testing_save_path, 'w') as f:

            for k, v in test_data.items():
                for content in v:
                    f.write(k + '\t' + content)
                    testing_rows_written += 1

        assert testing_rows_written == 12_400, f'There were {testing_rows_written} rows written, should have been 12,400.'
