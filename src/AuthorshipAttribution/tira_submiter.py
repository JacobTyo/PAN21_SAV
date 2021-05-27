"""
This year, each participating team is allowed a maximum of two submissions: one submission trained on the small
calibration dataset and one for the large calibration data set. (Participants can also choose to submit just one
system.)

Contrary to previous editions, submissions are not expected to be trained on TIRA (i.e. there will no new calibration
dataset for the testing phase). The submisisons should therefore contain already fully calibrated models that should only be deployed on TIRA for the actual testing.

We ask you to prepare your software so that it can be executed via command line calls. The command shall take as
input (i) an absolute path to the EVALUATION-DIRECTORY and (ii) an absolute path to an OUTPUT-DIRECTORY:

 > mySoftware -i EVALUATION-DIRECTORY -o OUTPUT-DIRECTORY
Within EVALUATION-DIRECTORY, a single jsonl-formatted file will be included (pairs.jsonl), containing the text pairs
(analogously to the calibration data that was released). The answers should be written to a jsonl-formatted file
(answers.jsonl) under EVALUATION-DIRECTORY. Each line should contain a single json-formatted answer,
using the following syntax:

{"id": "c04fdf1e-ddf5-5542-96e7-13ce18cae176", "value": 0.4921}
{"id": "49dc4cae-3d32-5b4d-b240-a080a1dbb659", "value": 0.5}
{"id": "f326fe7c-fc10-566f-a70f-0f36e3f92399", "value": 0.5}
{"id": "16daa0d1-61b8-5650-b7ee-5e265bd40910", "value": 0.9333}
{"id": "08b536a8-4fed-5f62-97bb-e57f79e841d2", "value": 0.0751}
...
Note: Each verification problem should be solved independently of other problems in the collection.

You can choose freely among the available programming languages and among the operating systems Microsoft Windows and
Ubuntu. We will ask you to deploy your software onto a virtual machine that will be made accessible to you after
registration. You will be able to reach the virtual machine via ssh and via remote desktop. More information about how
to access the virtual machines can be found in the user guide linked above.

Once deployed in your virtual machine, we ask you to access TIRA, where you can self-evaluate your software on the test
data.

Note: By submitting your software you retain full copyrights. You agree to grant us usage rights only for the purpose
of the PAN competition. We agree not to share your software with a third party or use it for other purposes than the
PAN competition.
"""


import torch
from torch.utils.data import DataLoader
from torch import nn
from dsets.imdb import get_imdb_train_or_test, get_training_for_triplet_loss_imdb, IMDB62ContrastiveDataset
from dsets.blog import get_blogs_train_or_test, get_training_for_triplet_loss_blogs, BlogsContrastiveDataset, \
    BlogsOSContrastiveDataset, get_training_for_triplet_loss_blogsos, get_blogsos_contrastive_test
from dsets.fanfiction import get_ff_triplet_training, FFDataset, get_ff_contrastive_train_or_test
from utils.AaEvaluator import AaEvaluator, ContrastiveEvaluator
from sentence_transformers import SentenceTransformer, losses, models, SentencesDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from utils.char_tokenizer import CharTokenizer, MyTokenizerFromTransformers
from utils.ModifiedContrastiveLoss import ModifiedContrastiveLoss
from transformers import AutoTokenizer
from utils.parallel import Transformer
from utils.DistributedSentenceTransformer import DistributedSentenceTransformer

import mlflow
import argparse
import time
import os
import numpy as np
import random
import json
from tqdm import tqdm




def main(args):

        print('starting')

        print('building model')

        word_embedding_model = models.Transformer('bert-base-cased',
                                                  max_seq_length=args.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                   out_features=256,
                                   activation_function=nn.ReLU())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                    device=args.device)

        model.load_state_dict(torch.load(args.model_path))

        print(f'setting distance metric: {args.distance_metric}')
        # which loss fn are are using?
        if args.distance_metric == 'euclidean':
            distance_metric = SiameseDistanceMetric.EUCLIDEAN
        elif args.distance_metric == 'cosine':
            distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        else:
            assert False, f'{args.distance_metric} is not defined'

        # determine if they gave us the dir or the file
        if args.eval_dir.endswith('pairs.jsonl'):
            eval_file = args.eval_dir
        else:
            eval_file = os.path.join(args.eval_dir, 'pairs.jsonl')
        print(f'starting evaluation of: {eval_file}')

        predictions = []
        identities = []
        with open(eval_file, 'r') as f:
            for line in tqdm(f):
                l = json.loads(line.strip())

                identity = l['id']
                text0 = l['pair'][0]
                text1 = l['pair'][1]

                if args.chunk:

                    if len(text0) > 500:
                        # chunk the sample and get separate predictions - just average them together for now
                        txt1_embeddings = []
                        for i in range(0, len(text0), 500):
                            end_idx = i + 500 if i + 500 < len(text0) else len(text0)
                            txt1_embeddings.append(model.encode(text0[i:end_idx]))

                        enc0 = np.mean(txt1_embeddings, axis=0)
                    else:
                        enc0 = model.encode(text0)

                    if len(text1) > 500:
                        # chunk the sample and get separate predictions - just average them together for now
                        txt2_embeddings = []
                        for i in range(0, len(text1), 500):
                            end_idx = i + 500 if i + 500 < len(text1) else len(text1)
                            txt2_embeddings.append(model.encode(text1[i:end_idx]))

                        enc1 = np.mean(txt2_embeddings, axis=0)
                    else:
                        enc1 = model.encode(text1)

                else:
                    enc0 = model.encode(text0)
                    enc1 = model.encode(text1)

                # now determine if these are of the same author
                predictions.append(distance_metric(torch.tensor([enc0]), torch.tensor([enc1])))
                identities.append(identity)

        # when it is finished, apply the thresholding and write the scores and normalize
        predictions = np.array(predictions, dtype=np.float32)
        predictions = np.max(predictions) - predictions
        # now need to swap everything - these are distances not similarities
        predictions = predictions / np.max(predictions)

        og_predictions = predictions.copy()

        # overall: 0.5996974027950318
        # lower thresh: 0.49948979591836734
        # upper thresh: 0.5836734693877551
        margin = args.margin
        same_margin = args.same_margin
        predictions = my_thresholder(predictions, low_thresh=same_margin, high_thresh=margin)

        print('checking the unanswered pairs and oh sh*t scenarios')

        def check_unanswered(preds):
            tmp2 = preds.copy()
            tmp2[tmp2 != 0.5] = 0
            return sum(tmp2) * 2 / len(tmp2)

        def check_bal(preds, m):
            tmp = predictions.copy()
            tmp[tmp > m] = 1
            tmp[tmp <= m] = 0
            return sum(tmp)/len(tmp)

        # make sure nothing went wrong
        bal = check_bal(predictions, margin)

        if bal < 0.35 or bal > 0.65:
            t = sorted(og_predictions, reverse=True)[int(len(og_predictions)/2)]

            same_margin = t-0.025
            margin = t+0.025
            predictions = my_thresholder(og_predictions, low_thresh=same_margin, high_thresh=margin)

        checker = check_unanswered(predictions)
        while checker > 0.07:
            same_margin = same_margin * 1.05
            margin = margin * 0.95
            predictions = my_thresholder(og_predictions, low_thresh=same_margin, high_thresh=margin)
            checker = check_unanswered(predictions)

        # now write the predictions to the output directory
        if args.output_dir.endswith('answers.jsonl'):
            output_file = args.output_dir
        else:
            output_file = os.path.join(args.output_dir, 'answers.jsonl')
        with open(output_file, 'w') as f:
            for i, pred in zip(identities, predictions):
                o = {'id': i, 'value': float(pred)}
                f.write(json.dumps(o))
                f.write('\n')

        print(f'finished writing to {output_file}')


def my_thresholder(sims, low_thresh, high_thresh, epsilon=1e-6, min_decision=-1, max_decision=-1):
    if min_decision == -1:
        min_decision = low_thresh - epsilon

    if max_decision == -1:
        max_decision = high_thresh + epsilon

    fixed_sims = sims.copy()
    # need to make sure this is < 0.5 - just normalize to [0, 0.5)

    fixed_sims[sims < low_thresh] = sims[sims < low_thresh] / ((1 / min_decision) * low_thresh)
    # need to make sure samples above high thresh is normalized to be in (0.5, 1]

    fixed_sims[sims > high_thresh] = max_decision + ((1 - max_decision) * (sims[sims > high_thresh] - high_thresh)) / (
                1 - high_thresh)

    fixed_sims[np.logical_and(sims >= low_thresh, sims <= high_thresh)] = 0.5
    # in_range_indices = np.where(boolean_array)[0]
    #
    # fixed_sims[low_thresh <= sims and sims <= high_thresh] = 0.5
    return fixed_sims


if __name__ == "__main__":

    # get command line args
    parser = argparse.ArgumentParser(description='Get args for SiameseBert Authorship Attribution Evaluation')

    parser.add_argument('--model_path', metavar='model_path', type=str, default=None,
                        help='The path to the model to evaluate')

    parser.add_argument('--device', metavar='device', type=str, default='cpu', help='which device to use.')

    parser.add_argument('--max_seq_length', metavar='max_seq_length', type=int, default=512,
                        help='What is the maximum sequence length to use with the model?')

    parser.add_argument('--distance_metric', metavar='distance_metric', type=str, default='euclidean',
                        help='which distance metric to use')

    parser.add_argument('--seed', metavar='seed', type=int, default=0,
                        help='set the seeds for randomness')

    parser.add_argument('--margin', metavar='margin', type=float, default=0.5530612244897959,
                        help='the margin to use, for modified contrastive loss this is the different margin')

    parser.add_argument('--same_margin', metavar='same_margin', type=float, default=0.4701530612244898,
                        help='only used in modified contrastive loss, the neighborhood for same author samples to be in')

    parser.add_argument('--chunk', action='store_true')

    parser.add_argument('--eval_dir', metavar='eval_dir')

    parser.add_argument('--output_dir', metavar='output_dir')

    args = parser.parse_args()

    # set the seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
