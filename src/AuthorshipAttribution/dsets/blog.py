import copy
import random, os
from torch.utils.data import Dataset
from sentence_transformers import InputExample
from tqdm import tqdm
import json
import numpy as np
from typing import List, Tuple, Optional, Union
import csv
import copy


class BlogsContrastiveDataset(Dataset):
    """Dataset for Author Verification on the IMDB62 Dataset."""

    def __init__(self,
                 data_file: str,
                 base_rate: float = 0.5
                 ):
        """
        Args:
            data_file (string): the path to the IMDB62 Dataset txt file
        """
        self.data_file = data_file
        self.base_rate = base_rate

        # get data file
        self.data = get_blogs_train(self.data_file)

    def __len__(self):
        return sum([len(x) for x in self.data.values()])

    def __getitem__(self, idx):
        # we want this to work with contrastive, so sample on the author level
        auth1 = random.choice(list(self.data.keys()))
        i_give_up = False
        rand_num = np.random.uniform()

        if rand_num < self.base_rate:
            # this is a same_author sample
            # make sure the auth has multiple samples
            while len(self.data[auth1]) < 2:
                auth1 = random.choice(list(self.data.keys()))

            text1 = text2 = random.choice(self.data[auth1])
            # make sure the texts are different
            counter = 0
            im_confused_counter = 0
            auths_tried = 0
            while text1 == text2:
                text1 = random.choice(self.data[auth1])
                text2 = random.choice(self.data[auth1])
                counter += 1
                if counter > 100:
                    # these texts are the same, get a different author
                    while len(self.data[auth1]) < 2:
                        auth1 = random.choice(list(self.data.keys()))
                    auths_tried += 1
                    text1 = random.choice(self.data[auth1])
                    text2 = random.choice(self.data[auth1])
                    if auths_tried > 50:
                        assert False, "we've got problems, can't find a different text from same author"
                    counter = 0
                if im_confused_counter > 10000:
                    # just give up and serve a same author pair?
                    # print(auth1)
                    # print(text1)
                    # assert False, "we've got problems, stuck in this same-author loop again."
                    i_give_up = True
                    break
                im_confused_counter += 1

            label = 1

        if not i_give_up and rand_num < self.base_rate:

            return InputExample(texts=[text1, text2], label=label)

        else:
            # this is a different author sample
            auth2 = auth1
            while auth1 == auth2:
                auth2 = random.choice(list(self.data.keys()))
            # now get a text from both authors
            text1 = random.choice(self.data[auth1])
            text2 = random.choice(self.data[auth2])
            label = 0
            return InputExample(texts=[text1, text2], label=label)


class BlogsOSContrastiveDataset(Dataset):
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
        self.data = get_contrastive_training_set_blogsos(self.data_file, as_dict=True)

    def __len__(self):
        return sum([len(x) for x in self.data.values()])

    def __getitem__(self, idx):
        # we want this to work with contrastive, so sample on the author level
        auth1 = random.choice(list(self.data.keys()))

        if np.random.uniform() < self.base_rate:
            # this is a same_author sample
            # make sure the auth has multiple samples
            while len(self.data[auth1]) < 2:
                auth1 = random.choice(list(self.data.keys()))

            text1 = random.choice(self.data[auth1])
            text2 = random.choice(self.data[auth1])
            # make sure the texts are different
            counter = 0
            im_confused_counter = 0
            auths_tried = 0
            while text1 == text2:
                text1 = random.choice(self.data[auth1])
                text2 = random.choice(self.data[auth1])
                counter += 1
                if counter > 100:
                    # these texts are the same, get a different author
                    while len(self.data[auth1]) < 2:
                        auth1 = random.choice(list(self.data.keys()))
                    auths_tried += 1
                    text1 = random.choice(self.data[auth1])
                    text2 = random.choice(self.data[auth1])
                    if auths_tried > 100:
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
                auth2 = random.choice(list(self.data.keys()))
            # now get a text from both authors
            text1 = random.choice(self.data[auth1])
            text2 = random.choice(self.data[auth2])
            label = 0

        return InputExample(texts=[text1, text2], label=label)


def get_training_for_triplet_loss_blogs(data_file: str) -> List[InputExample]:
    raw_data = get_blogs_train_or_test(data_file)

    # make list of InputExamples, return
    data = []
    for i in raw_data:
        data.append(InputExample(texts=[i[1]], label=int(i[0])))

    return data


def get_training_for_triplet_loss_blogsos(data_file: str) -> List[InputExample]:

    raw_data = get_contrastive_training_set_blogsos(data_file, as_dict=False)
    # raw data is a list of samples, make input examples and good to go

    # make list of InputExamples, return
    data = []
    for i in raw_data:
        data.append(InputExample(texts=[i[1]], label=int(i[0])))

    return data


def get_blogs_train_or_test(dataset_path: str, as_dict: bool = False) -> List[Tuple[str, str]]:

    with open(dataset_path, 'r') as f:

        raw_data = json.load(f)

    if as_dict:

        return raw_data

    else:  # return list of (auth_id, text) pairs

        data = []

        for key, val in raw_data.items():
            for v in val:
                data.append((key, v))

        return data


def init_blogs_as_dict(dataset_path: str, chunk: bool = False) -> dict:
    # get the dataset as a dic, with key being author id and value being a list of contents
    raw_data = {}

    # transform user_id's as well
    label_transformer = {}
    label_count = 0

    # for every blog (i.e. every file in the directory
    for directory, subdirectories, files in os.walk(dataset_path):

        for file in tqdm(files):

            # get the author id
            auth_id = file.split('.')[0]
            if auth_id not in label_transformer.keys():
                label_transformer[auth_id] = label_count
                label_count += 1

            with open(os.path.join(directory, file), 'r', errors='ignore') as f:

                # we want just the data between the <post> and </post> tags as content
                lines = f.readlines()

                auth_content = []

                get_content = False

                this_line_content = ''

                for line in lines:
                    if get_content:

                        # if line is not empty
                        if not line.isspace():
                            this_line_content += line.strip()

                        if '</post>' in line:
                            get_content = False
                            auth_content.append(this_line_content)
                            this_line_content = ''

                    else:

                        # only get what is between these lines. XML parsing may be easier but this will work
                        if '<post>' in line:

                            get_content = True

                # we have all the lines, add the content to the author
                if label_transformer[auth_id] in raw_data.keys():
                    for content in auth_content:
                        if content not in raw_data[label_transformer[auth_id]]:
                            raw_data[label_transformer[auth_id]].append(content)
                else:
                    raw_data[label_transformer[auth_id]] = []
                    for content in auth_content:
                        raw_data[label_transformer[auth_id]].append(content)

        return raw_data


def init_blogs_as_dict_bertaa(dataset_path: str) -> dict:
    # get the dataset as a dic, with key being author id and value being a list of contents
    raw_data = {}

    # transform user_id's as well
    label_transformer = {}
    label_count = 0

    # for every blog (i.e. every file in the directory
    for directory, subdirectories, files in os.walk(dataset_path):

        for file in tqdm(files):

            # get the author id
            meta_info = file.split('.')
            auth_id = meta_info[0]
            gender = meta_info[1]
            age = meta_info[2]
            topic = meta_info[3]
            sign = meta_info[4]

            if auth_id not in label_transformer.keys():
                label_transformer[auth_id] = label_count
                label_count += 1

            with open(os.path.join(directory, file), 'r', errors='ignore') as f:

                # we want just the data between the <post> and </post> tags as content
                lines = f.readlines()

                auth_content = []

                get_content = False

                this_line_content = ''

                for line in lines:

                    # check date
                    if '<date>' in line:
                        date = line.strip().replace('<date>', '').replace('</date>', '').strip()

                    if get_content:

                        # if line is not empty
                        if not line.isspace():
                            this_line_content += (line.strip() + ' ').replace('\0','')

                        if '</post>' in line:
                            get_content = False
                            auth_content.append((this_line_content, date))
                            this_line_content = ''

                    else:

                        # only get what is between these lines. XML parsing may be easier but this will work
                        if '<post>' in line:

                            get_content = True

                # we have all the lines, add the content to the author
                for auth_c in auth_content:
                    if label_transformer[auth_id] in raw_data.keys():
                        raw_data[label_transformer[auth_id]].append((auth_c[0], auth_id, gender, age, topic, sign, auth_c[1]))
                    else:
                        raw_data[label_transformer[auth_id]] = [(auth_c[0], auth_id, gender, age, topic, sign, auth_c[1])]

        return raw_data


def get_contrastive_training_set_blogsos(data_file: str, as_dict: bool = True) -> dict:

    raw_data = {} if as_dict else []
    with open(data_file, 'r') as f:
        #csv_reader = csv.reader(f)
        csv_reader = csv.reader((line.replace('\0','') for line in f))
        for row in csv_reader:
            if row[0] == "auth_id":
                # skip header
                continue

            if as_dict:
                auth_id = int(row[0].strip())
                if auth_id not in raw_data.keys():
                    raw_data[auth_id] = [row[1]]
                else:
                    raw_data[auth_id].append(row[1])
            else:
                raw_data.append((row[0], row[1]))

    return raw_data

def get_blogsos_contrastive_test(data_file: str) -> List[InputExample]:

    raw_data = []

    with open(data_file, 'r') as f:
        csv_reader = csv.reader((line.replace('\0', '') for line in f))
        #csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[0] == "same":
                # skip header
                continue
            raw_data.append(InputExample(texts=[row[1], row[2]], label=int(row[0])))

    return raw_data

def get_blogs_train(data_file):

    raw_data = []

    with open(data_file, 'r') as f:
        d = json.load(f)

    return d


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
    parser.add_argument('--what_for', metavar='what_for', type=str,
                        help='set to sbert or bertaa')
    parser.add_argument('--chunk', action='store_true',
                        help='set this flag to chunk examples into lengths of 500 words')

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    if args.what_for == 'sbert':

        print(f'getting original dataset at {args.dataset_path}')
        # read the csv file, and becasue the dataset isn't very big, keep everything in memory
        data = init_blogs_as_dict(args.dataset_path)

        print(f'there are {len(data.keys())} authors')

        print('splitting dataset')
        # now split this into a train and test dict (for simplicity, just take first 800 training, last 200 test
        training_data, test_data = {}, {}
        for k, v in data.items():
            # how many items does this author have?
            num_samps = len(v)
            if num_samps > 2:
                split_idx = int(0.8*num_samps)
                tmp_train = v[:split_idx]
                tmp_test = v[split_idx:]
                if len(tmp_train) > 1 and len(tmp_test) > 1:
                    training_data[k] = tmp_train
                    test_data[k] = tmp_test

        # print some stats about the data
        print(f'There are {len(training_data.keys())} authors in the training set')
        print(f'There are {len(test_data.keys())} authors in the testing set')

        def get_article_counts(d):
            article_counts = [len(v) for k, v in d.items()]
            max_art, min_art = np.max(article_counts), np.min(article_counts)
            mean_art, median_art = np.mean(article_counts), np.median(article_counts)
            return mean_art, median_art, max_art, min_art

        mean_articles, median_articles, max_articles, min_articles = get_article_counts(training_data)
        print(f'The mean, median, max, and min articles per author in the training set is {mean_articles}, {median_articles}, {max_articles}, {min_articles}')

        mean_articles, median_articles, max_articles, min_articles = get_article_counts(test_data)
        print(f'The mean, median, max, and min articles per author in the testing set is {mean_articles}, {median_articles}, {max_articles}, {min_articles}')

        # now write this to a new file, blogs-train.json and blogs-test.json
        training_save_path = os.path.join(args.dataset_save_path, 'blogs-train.json')
        testing_save_path = os.path.join(args.dataset_save_path, 'blogs-test.json')

        print(f'saving new training dataset at {training_save_path}')
        with open(training_save_path, 'w') as f:
            json.dump(training_data, f, sort_keys=True, indent=4)

        print(f'saving new testing dataset at {testing_save_path}')
        with open(testing_save_path, 'w') as f:

            json.dump(test_data, f, sort_keys=True, indent=4)


        # replicate the top 50 and top 100 datasets
        # base on counts
        def get_top_n_auth_dicts(d, n=100):
            article_counts = {k: len(v) for k, v in d.items()}
            topnauths = {key: d[key] for key in sorted(article_counts, key=article_counts.get, reverse=True)[:n]}
            return topnauths

        top100auths = get_top_n_auth_dicts(data, n=100)
        top50auths = get_top_n_auth_dicts(data, n=50)

        # split into train test
        def split_into_train_test(d, percent_train=0.8):
            training_data, test_data = {}, {}
            for k, v in d.items():
                # how many items does this author have?
                num_samps = len(v)
                if num_samps > 2:
                    split_idx = int(0.8 * num_samps)
                    tmp_train = v[:split_idx]
                    tmp_test = v[split_idx:]
                    if len(tmp_train) > 1 and len(tmp_test) > 1:
                        training_data[k] = tmp_train
                        test_data[k] = tmp_test
            return training_data, test_data

        top100_train, top100_test = split_into_train_test(top100auths, 0.8)
        top50_train, top50_test = split_into_train_test(top50auths, 0.8)

        top100_train_nc = copy.deepcopy(top100_train)

        # now chunk if needed:
        # chunk texts if necessary
        if args.chunk:
            print('before chunk:')
            num_texts = sum([len(x) for x in top100_train.values()])
            num_auths = sum([1 for x in top100_train.keys()])
            all_texts, total_len = [], 0
            for texts in top100_train.values():
                for text in texts:
                    total_len += len(text)
            avg_txt_len = total_len / num_texts
            print(f'\tnum_texts: {num_texts}')
            print(f'\tnum_auths: {num_auths}')
            print(f'\ttext_len: {total_len}')
            print(f'\tavg_len: {avg_txt_len}')

            for auth, texts in top100_train.items():
                new_texts = []
                for text in texts:
                    split_text = text.split(' ')
                    if len(split_text) > 500:
                        for start_idx in range(0, len(split_text), 500):
                            end_idx = start_idx + 500 if start_idx + 500 < len(split_text) else len(split_text) - 1
                            new_texts.append(' '.join(split_text[start_idx: end_idx]))
                    else:
                        new_texts.append(text)
                top100_train[auth] = new_texts

            print('after chunk:')
            num_texts = sum([len(x) for x in top100_train.values()])
            num_auths = sum([1 for x in top100_train.keys()])
            all_texts, total_len = [], 0
            for texts in top100_train.values():
                for text in texts:
                    total_len += len(text)
            avg_txt_len = total_len / num_texts
            print(f'\tnum_texts: {num_texts}')
            print(f'\tnum_auths: {num_auths}')
            print(f'\ttext_len: {total_len}')
            print(f'\tavg_len: {avg_txt_len}')

        assert len(top100_train.keys()) == 100, 'problem with the top 100 train'
        assert len(top100_test.keys()) == 100, 'problem with the top 100 test'
        assert len(top50_train.keys()) == 50, 'problem with the top 50 train'
        assert len(top50_test.keys()) == 50, 'problem with the top 100 test'

        # save these splits
        top100_train_save_path = os.path.join(args.dataset_save_path, 'blogs-top100-train.json')
        top100_test_save_path = os.path.join(args.dataset_save_path, 'blogs-top100-test.json')

        print(f'saving new 100 auth testing dataset at {top100_test_save_path}')
        with open(top100_test_save_path, 'w') as f:
            json.dump(top100_test, f, sort_keys=True, indent=4)

        print(f'saving new 100 auth train dataset at {top100_train_save_path}')
        with open(top100_train_save_path, 'w') as f:
            json.dump(top100_train, f, sort_keys=True, indent=4)

        top100_nc_train_save_path = os.path.join(args.dataset_save_path, 'blogs-top100-train-nc.json')
        print(f'saving new nc 100 auth train dataset at {top100_nc_train_save_path}')
        with open(top100_nc_train_save_path, 'w') as f:
            json.dump(top100_train_nc, f, sort_keys=True, indent=4)

        top50_train_save_path = os.path.join(args.dataset_save_path, 'blogs-top50-train.json')
        top50_test_save_path = os.path.join(args.dataset_save_path, 'blogs-top50-test.json')

        print(f'saving new 50 auth testing dataset at {top50_test_save_path}')
        with open(top50_test_save_path, 'w') as f:
            json.dump(top50_test, f, sort_keys=True, indent=4)

        print(f'saving new 50 auth train dataset at {top50_train_save_path}')
        with open(top50_train_save_path, 'w') as f:
            json.dump(top50_train, f, sort_keys=True, indent=4)

        # get some more stats
        num_articles = [len(v) for k, v in top100_train.items()]
        mn, mx, md, me = np.min(num_articles), np.max(num_articles), np.median(num_articles), np.mean(num_articles)
        print(f'The mean, median, max, and min number of articles for each auth in top 100 is: {me}, {md}, {mx}, {mn}.')
        auth_word_count = {}
        for k, v in top100_train.items():
            words = 0
            for post in v:
                words += len(post.split())
            auth_word_count[k] = words
        num_words = [v for k, v in auth_word_count.items()]
        mn, mx, md, me = np.min(num_words), np.max(num_words), np.median(num_words), np.mean(num_words)
        print(f'The mean, median, max, and min word count for each auth in top 100 is: {me}, {md}, {mx}, {mn}.')

    elif args.what_for == 'bertaa':

        # read all of the xml files
        print(f'getting original dataset at {args.dataset_path}')

        # this gives a dick where they key is the normalized author id, the value is a tuple of
        # (auth_content, auth_id, gender, age, topic, sign, date)
        data = init_blogs_as_dict_bertaa(args.dataset_path)

        print(f'there are {len(data.keys())} authors')

        print('splitting dataset')
        # now split this into a train and test dict (for simplicity, just take first 800 training, last 200 test
        training_data, test_data = {}, {}
        for k, v in data.items():
            # how many items does this author have?
            num_samps = len(v)
            if num_samps > 2:
                split_idx = int(0.8 * num_samps)
                tmp_train = v[:split_idx]
                tmp_test = v[split_idx:]
                if len(tmp_train) > 1 and len(tmp_test) > 1:
                    training_data[k] = tmp_train
                    test_data[k] = tmp_test

        # print some stats about the data
        print(f'There are {len(training_data.keys())} authors in the training set')
        print(f'There are {len(test_data.keys())} authors in the testing set')


        def get_article_counts(d):
            article_counts = [len(v) for k, v in d.items()]
            max_art, min_art = np.max(article_counts), np.min(article_counts)
            mean_art, median_art = np.mean(article_counts), np.median(article_counts)
            return mean_art, median_art, max_art, min_art


        mean_articles, median_articles, max_articles, min_articles = get_article_counts(training_data)
        print(
            f'The mean, median, max, and min articles per author in the training set is {mean_articles}, {median_articles}, {max_articles}, {min_articles}')

        mean_articles, median_articles, max_articles, min_articles = get_article_counts(test_data)
        print(
            f'The mean, median, max, and min articles per author in the testing set is {mean_articles}, {median_articles}, {max_articles}, {min_articles}')

        # now write this to a csv for processing by bertaa
        import csv

        # open the file in the write mode
        with open(os.path.join(args.dataset_save_path, 'blogs.csv'), 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write header
            writer.writerow(["From", "Gender", "Age", "Topic", "Sign", "Date", "content"])
            # write contents
            for auth, vals in data.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])

        # now write this to a new file, blogs-train.json and blogs-test.json
        training_save_path = os.path.join(args.dataset_save_path, 'blogs-train.csv')
        testing_save_path = os.path.join(args.dataset_save_path, 'blogs-test.csv')

        print(f'saving new training dataset at {training_save_path}')
        with open(training_save_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write header
            writer.writerow(["From", "Gender", "Age", "Topic", "Sign", "Date", "content"])
            # write contents
            for auth, vals in training_data.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])

        print(f'saving new testing dataset at {testing_save_path}')
        with open(testing_save_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write header
            writer.writerow(["From", "Gender", "Age", "Topic", "Sign", "Date", "content"])
            # write contents
            for auth, vals in test_data.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])

        # replicate the top 50 and top 100 datasets
        # base on counts
        def get_top_n_auth_dicts(d, n=100):
            article_counts = {k: len(v) for k, v in d.items()}
            topnauths = {key: d[key] for key in sorted(article_counts, key=article_counts.get, reverse=True)[:n]}
            return topnauths


        top100auths = get_top_n_auth_dicts(data, n=100)
        top50auths = get_top_n_auth_dicts(data, n=50)


        # split into train test
        def split_into_train_test(d, percent_train=0.8):
            training_data, test_data = {}, {}
            for k, v in d.items():
                # how many items does this author have?
                num_samps = len(v)
                if num_samps > 2:
                    split_idx = int(0.8 * num_samps)
                    tmp_train = v[:split_idx]
                    tmp_test = v[split_idx:]
                    if len(tmp_train) > 1 and len(tmp_test) > 1:
                        training_data[k] = tmp_train
                        test_data[k] = tmp_test
            return training_data, test_data


        top100_train, top100_test = split_into_train_test(top100auths, 0.8)
        top50_train, top50_test = split_into_train_test(top50auths, 0.8)

        assert len(top100_train.keys()) == 100, 'problem with the top 100 train'
        assert len(top100_test.keys()) == 100, 'problem with the top 100 test'
        assert len(top50_train.keys()) == 50, 'problem with the top 50 train'
        assert len(top50_test.keys()) == 50, 'problem with the top 100 test'

        # save these splits
        top100_save_path = os.path.join(args.dataset_save_path, 'blogs-top100.csv')
        top100_train_save_path = os.path.join(args.dataset_save_path, 'blogs-top100-train.csv')
        top100_test_save_path = os.path.join(args.dataset_save_path, 'blogs-top100-test.csv')

        print(f'saving new 100 auth dataset at {top100_save_path}')
        with open(top100_save_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write header
            writer.writerow(["From", "Gender", "Age", "Topic", "Sign", "Date", "content"])
            # write contents
            for auth, vals in top100_train.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])
            for auth, vals in top100_test.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])

        print(f'saving new 100 auth testing dataset at {top100_test_save_path}')
        with open(top100_test_save_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write header
            writer.writerow(["From", "Gender", "Age", "Topic", "Sign", "Date", "content"])
            # write contents
            for auth, vals in top100_test.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])

        print(f'saving new 100 auth train dataset at {top100_train_save_path}')
        with open(top100_train_save_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write header
            writer.writerow(["From", "Gender", "Age", "Topic", "Sign", "Date", "content"])
            # write contents
            for auth, vals in top100_train.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])

        top50_train_save_path = os.path.join(args.dataset_save_path, 'blogs-top50-train.csv')
        top50_test_save_path = os.path.join(args.dataset_save_path, 'blogs-top50-test.csv')

        print(f'saving new 50 auth testing dataset at {top50_test_save_path}')
        with open(top50_test_save_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write header
            writer.writerow(["From", "Gender", "Age", "Topic", "Sign", "Date", "content"])
            # write contents
            for auth, vals in top50_test.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])

        print(f'saving new 50 auth train dataset at {top50_train_save_path}')
        with open(top50_train_save_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write header
            writer.writerow(["From", "Gender", "Age", "Topic", "Sign", "Date", "content"])
            # write contents
            for auth, vals in top50_train.items():
                for val in vals:
                    writer.writerow([val[1], val[2], val[3], val[4], val[5], val[6], val[0]])

        # get some more stats
        num_articles = [len(v) for k, v in top100_train.items()]
        mn, mx, md, me = np.min(num_articles), np.max(num_articles), np.median(num_articles), np.mean(num_articles)
        print(f'The mean, median, max, and min number of articles for each auth in top 100 is: {me}, {md}, {mx}, {mn}.')
        auth_word_count = {}
        for k, v in top100_train.items():
            words = 0
            for post in v:
                post = post[0]
                words += len(post.split())
            auth_word_count[k] = words
        num_words = [v for k, v in auth_word_count.items()]
        mn, mx, md, me = np.min(num_words), np.max(num_words), np.median(num_words), np.mean(num_words)
        print(f'The mean, median, max, and min word count for each auth in top 100 is: {me}, {md}, {mx}, {mn}.')

    elif args.what_for == 'BlogsOS':

        # need to build a dataset based for open set evaluation based on the blogs corpus. Not sure how to really go about this
        # I want to just do some filtering, then have the training set be just as any other contrastive/triplet loss setup, then
        # have the test set just a bunch of pairs pre-cretaed.
        # it will be important to also get bert aa working in this type of "open verification" setup. This will likely entail just using the model for bert-100,
        # then compare the similarity of two texts with that, then we can just predict same or different.
        # however, before doing this, we need to come up with a way to filter I think, because the dataset is quite large.
        # so first, lets notebook it to get a handle of it, and complete the proper analysis before splitting.
        # Alright, after initial investigation, we should just randomly split on author, build pairs, call it good.

        # get the initial set:
        print(f'getting original dataset at {args.dataset_path}')
        # read the csv file, and becasue the dataset isn't very big, keep everything in memory
        data = init_blogs_as_dict(args.dataset_path, chunk=args.chunk)

        # get rid of duplicates
        for auth, texts in data.items():
            data[auth] = list(set(texts))

        # now we want 20% authors test, 80% train
        all_authors = list(data.keys())
        random.shuffle(all_authors)

        divider = int(0.9 * len(all_authors))
        train_authors = all_authors[:divider]
        test_authors = all_authors[divider:]

        train_data = dict((auth, data[auth]) for auth in train_authors)
        test_data = dict((auth, data[auth]) for auth in test_authors)

        non_chunked_train_data = copy.deepcopy(train_data)

        # chunk texts if necessary
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

            for auth, texts in train_data.items():
                new_texts = []
                for text in texts:
                    split_text = text.split(' ')
                    if len(split_text) > 500:
                        for start_idx in range(0, len(split_text), 500):
                            end_idx = start_idx + 500 if start_idx + 500 < len(split_text) else len(split_text) - 1
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

        num_texts_train = sum([len(x) for x in train_data.values()])
        num_texts_test = sum([len(x) for x in test_data.values()])

        print(f'there are {num_texts_train} training texts, and {num_texts_test} testing texts, so {num_texts_train/(num_texts_train+num_texts_test)} percent is training.')

        # now create the pairs for testing
        # for every text in the test set, make 1 same-author pair, and 3 different-author pairs
        # make sure to correct for authors with only one text
        # just keep a count of how many authors only have one text - then just sample the same-author pairs after that
        test_pairs = []
        single_text_auths = 0
        for auth, texts in tqdm(test_data.items()):
            if len(texts) > 2:
                # we can make some pairs here
                for text1 in texts:
                    text2 = random.choice(texts)
                    while (1, text1, text2) in test_pairs:
                        text2 = random.choice(texts)
                    # add the point
                    test_pairs.append((1, text1, text2))
            else:
                # just gotta count it
                single_text_auths += 1

            # add one diff auth text for every text we made a pair for
            for _ in range(len(texts) - 1):
                # now get a diff author pair
                text1 = random.choice(texts)
                auth2 = random.choice(test_authors)
                while auth == auth2:
                    auth2 = random.choice(test_authors)
                text2 = random.choice(test_data[auth2])
                while (0, text1, text2) in test_pairs:
                    auth2 = random.choice(test_authors)
                    while auth == auth2:
                        auth2 = random.choice(test_authors)
                    text2 = random.choice(test_data[auth2])
                test_pairs.append((0, text1, text2))


        # now add in the number of same auth pairs as was auths with single text
        for _ in tqdm(range(single_text_auths)):
            # select an author
            auth = random.choice(test_authors)
            while len(test_data[auth]) < 2:
                auth = random.choice(test_authors)
            text1, text2 = random.choice(test_data[auth]), random.choice(test_data[auth])
            while (1, text1, text2) in test_pairs:
                auth = random.choice(test_authors)
                while len(test_data[auth]) < 2:
                    auth = random.choice(test_authors)
                text1, text2 = random.choice(test_data[auth]), random.choice(test_data[auth])
            test_pairs.append((1, text1, text2))

        # now save the test pairs
        training_save_path = os.path.join(args.dataset_save_path, 'blogsOS-train.json')
        testing_save_path = os.path.join(args.dataset_save_path, 'blogsOS-test.csv')

        print(f'saving Blogs OS contrastive test dataset at {testing_save_path}')
        with open(testing_save_path, 'w') as f:

            writer = csv.writer(f)
            # write header
            writer.writerow(['same', 'text1', 'text2'])
            for dp in test_pairs:
                writer.writerow(dp)

        print(f'saved, test set has {len(test_pairs)} pairs')

        print(f'saving BlogsOS train dataset at {training_save_path}')
        with open(training_save_path, 'w') as f:

            writer = csv.writer(f)
            # write header
            writer.writerow(['auth_id', 'text'])
            for auth, texts in train_data.items():
                for text in texts:
                    writer.writerow([auth, text])

        nc_training_save_path = os.path.join(args.dataset_save_path, 'blogsOS-train-non-chunked.json')

        print(f'saving BlogsOS train dataset at {nc_training_save_path}')
        with open(nc_training_save_path, 'w') as f:

            writer = csv.writer(f)
            # write header
            writer.writerow(['auth_id', 'text'])
            for auth, texts in non_chunked_train_data.items():
                for text in texts:
                    writer.writerow([auth, text])

        print('done')

