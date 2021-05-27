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


def main(args):
    mlflow.set_tracking_uri('http://gs18196.sp.cs.cmu.edu:6460')
    mlflow.set_experiment("aa")

    with mlflow.start_run(run_name=args.run_name):

        # log all of the args to mlflow
        try:
            for key, val in vars(args).items():
                mlflow.log_param(key, val)
        except Exception as e:
            for key, val in args.items():
                mlflow.log_param(key, val)

        print('starting')

        # get the training dataset, and a training evaluator
        if args.which_dataset == 'imdb':

            if 'Contrastive' in args.which_loss_fn:
                train_dataset = IMDB62ContrastiveDataset(args.train_dataset)

            else:
                train_dataset = get_training_for_triplet_loss_imdb(args.train_dataset)
            auth_emb_dataset = get_imdb_train_or_test(args.auth_emb_dataset, as_dict=True)
            test_dataset = get_imdb_train_or_test(args.test_dataset, as_dict=True)

        elif args.which_dataset == 'blogs':

            if 'Contrastive' in args.which_loss_fn:
                train_dataset = BlogsContrastiveDataset(args.train_dataset)
            else:
                train_dataset = get_training_for_triplet_loss_blogs(args.train_dataset)

            auth_emb_dataset = get_blogs_train_or_test(args.train_dataset, as_dict=True)
            test_dataset = get_blogs_train_or_test(args.test_dataset, as_dict=True)

        # need to set everthing up for BlogsOS
        elif args.which_dataset == 'blogsos':

            if 'Contrastive' in args.which_loss_fn:
                train_dataset = BlogsOSContrastiveDataset(args.train_dataset)
            else:
                train_dataset = get_training_for_triplet_loss_blogsos(args.train_dataset)

            # now just need a testing set - no auth_emb set needed
            test_dataset = get_blogsos_contrastive_test(args.test_dataset)

        elif args.which_dataset == 'fanfiction':

            if 'Contrastive' in args.which_loss_fn:
                train_dataset = FFDataset(args.train_dataset)

            else:
                train_dataset = get_ff_triplet_training(args.train_dataset)

            test_dataset = get_ff_contrastive_train_or_test(args.test_dataset)

        else:
            assert False, 'Only the imdb and blogs datasets are supported at the moment.'

        print('building model')

        if args.distributed:
            assert args.model_type == 'transformer', 'distributed training is only supported with transformers'

        # now build a model
        if not args.model_path:
            # what kind of model can we build?
            if args.model_type == 'transformer':
                if args.distributed:
                    device_ids = args.device_ids
                    word_embedding_model = Transformer('bert-base-cased',
                                                       max_seq_length=args.max_seq_length,
                                                       device_ids=device_ids)
                    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                               out_features=256,
                                               activation_function=nn.ReLU())
                    model = DistributedSentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                                           device=args.device)

                else:
                    word_embedding_model = models.Transformer('bert-base-cased',
                                                              max_seq_length=args.max_seq_length)
                    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                               out_features=256,
                                               activation_function=nn.ReLU())  # I wish I would have used ReLU()

                    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                                device=args.device)

            elif 'cnn' in args.model_type:
                if args.model_type == 'cnn_char':
                    char_tokenizer = CharTokenizer()
                    word_embedding_model = models.WordEmbeddings(char_tokenizer,
                                                                 np.random.random((len(char_tokenizer.get_vocab(None)),
                                                                                   args.word_embedding_dim)),
                                                                 update_embeddings=True)
                elif args.model_type == 'cnn_glove':
                    word_embedding_model = models.WordEmbeddings.from_text_file('glove.6B.300d.txt.gz')

                elif args.model_type == 'cnn_bert':

                    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
                    tokenizer = MyTokenizerFromTransformers(tokenizer)

                    word_embedding_model = models.WordEmbeddings(tokenizer,
                                                                 np.random.random(
                                                                     (len(tokenizer.get_vocab(None).keys()),
                                                                      args.word_embedding_dim)),
                                                                 update_embeddings=True)

                else:
                    assert False, 'this cnn type is not defined'

                cnn = models.CNN(word_embedding_model.get_word_embedding_dimension(), kernel_sizes=[1, 3, 5, 7, 9])
                pooling_model = models.Pooling(cnn.get_word_embedding_dimension())
                dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                           out_features=256,
                                           activation_function=nn.ReLU())
                # if args.device == 'cuda':
                #     model = DistributedSentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                #                                            device=args.device)
                # else:
                model = SentenceTransformer(modules=[word_embedding_model, cnn, pooling_model, dense_model],
                                            device=args.device)

        else:
            # if args.device == 'cuda':
            #     model = DistributedSentenceTransformer(modules=args.model_path,
            #                                            device=args.device)
            # else:
            if args.distributed:
                device_ids = args.device_ids
                word_embedding_model = Transformer('bert-base-cased',
                                                   max_seq_length=args.max_seq_length,
                                                   device_ids=device_ids)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                           out_features=256,
                                           activation_function=nn.ReLU())
                model = DistributedSentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                                       device=args.device)
                model.load_state_dict(torch.load(args.model_path + '/state_dict.pt'))

            else:
                if args.disregard_distributed:
                    word_embedding_model = models.Transformer('bert-base-cased',
                                                              max_seq_length=args.max_seq_length)
                    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                               out_features=256,
                                               activation_function=nn.ReLU())
                    word_embedding_model.load_state_dict(torch.load(args.model_path+'/0_Transformer/state_dict.pt'))
                    pooling_model.load_state_dict(torch.load(args.model_path + '/1_Pooling/state_dict.pt'))
                    dense_model.load_state_dict(torch.load(args.model_path + '/2_Dense/state_dict.pt'))
                    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                                device=args.device)
                else:
                    model = SentenceTransformer(args.model_path,
                                                device=args.device)

        # build the training dataloader
        train_sdataset = SentencesDataset(train_dataset, model)
        batch_size = args.batch_size if not args.distributed else torch.cuda.device_count() * args.batch_size
        train_dataloader = DataLoader(train_sdataset, shuffle=True, batch_size=batch_size)

        print('building datasets')

        # which loss fn are are using?
        if args.distance_metric == 'euclidean':
            contrastive_dist_metric = SiameseDistanceMetric.EUCLIDEAN
            triplet_dist_metric = BatchHardTripletLossDistanceFunction.eucledian_distance
        elif args.distance_metric == 'cosine':
            contrastive_dist_metric = SiameseDistanceMetric.COSINE_DISTANCE
            triplet_dist_metric = BatchHardTripletLossDistanceFunction.cosine_distance
        else:
            assert False, f'{args.distance_metric} is not defined'

        # set up an evaluator
        margin = args.margin if args.which_loss_fn != 'ModifiedContrastiveLoss' else (
                                                                                                 args.margin + args.same_margin) / 2
        if args.which_loss_fn == 'ContrastiveLoss' and args.which_dataset != 'imdb' and args.which_dataset != 'blogs' or args.which_dataset == 'blogsos' or args.which_dataset == 'fanfiction':
            evaluator = ContrastiveEvaluator(test_dataset,
                                             batch_size=args.eval_batch_size,
                                             distance_metric=contrastive_dist_metric,
                                             threshold=margin)
        else:
            evaluator = AaEvaluator(auth_emb_dataset,
                                    test_dataset,
                                    batch_size=args.eval_batch_size,
                                    distance_metric=triplet_dist_metric)

        if args.which_loss_fn == 'BatchAllTripletLoss':
            train_loss = losses.BatchAllTripletLoss(model=model, distance_metric=triplet_dist_metric,
                                                    margin=args.margin)
        elif args.which_loss_fn == 'BatchHardTripletLoss':
            train_loss = losses.BatchHardTripletLoss(model=model, distance_metric=triplet_dist_metric,
                                                     margin=args.margin)
        elif args.which_loss_fn == 'BatchSemiHardTripletLoss':
            train_loss = losses.BatchSemiHardTripletLoss(model=model, distance_metric=triplet_dist_metric,
                                                         margin=args.margin)
        elif args.which_loss_fn == 'BatchHardSoftMarginTripletLoss':
            train_loss = losses.BatchHardSoftMarginTripletLoss(model=model, distance_metric=triplet_dist_metric)
        elif args.which_loss_fn == 'ContrastiveLoss':
            train_loss = losses.ContrastiveLoss(model=model, distance_metric=contrastive_dist_metric,
                                                margin=args.margin)
        elif args.which_loss_fn == 'ModifiedContrastiveLoss':
            train_loss = ModifiedContrastiveLoss(model=model, distance_metric=contrastive_dist_metric,
                                                 different_margin=args.margin,
                                                 same_margin=args.same_margin)
        else:
            assert False, 'The selected loss function is not recognized.'

        if args.test_eval_time:
            start_time = time.time()
            print('running an initial evaluation')
            evaluator(model)
            elapsed_time = time.time() - start_time
            print(f'evaluation finished, it took {elapsed_time} seconds, or {elapsed_time / 60} minutes')

        print('training')

        # train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=args.epochs,
                  scheduler=args.scheduler,
                  warmup_steps=args.warmup_steps,
                  save_best_model=True,
                  output_path=os.path.join(args.output_path, args.run_name),
                  evaluator=evaluator,
                  evaluation_steps=args.evaluation_steps)

        print('finished')


if __name__ == "__main__":

    # get command line args
    parser = argparse.ArgumentParser(description='Get args for SiameseBert Authorship Attribution Evaluation')

    parser.add_argument('--run_name', metavar='run_name', type=str, help='The name to use for MLFlow logging')
    parser.add_argument('--model_path', metavar='model_path', type=str, default=None,
                        help='The path to the model to evaluate')
    parser.add_argument('--model_type', metavar='model_type', type=str, default='transformer',
                        help='Which model type to use, supported now are cnn_glove, cnn_char, and transformer')
    parser.add_argument('--train_dataset', metavar='train_dataset', type=str,
                        default='/home/jtyo/Repos/AuthorshipAttribution/data/IMDB/imdb62-train.txt',
                        help='The path to the dataset to be used (currently the IMDB62 dataset)')
    parser.add_argument('--test_dataset', metavar='test_dataset', type=str,
                        default='/home/jtyo/Repos/AuthorshipAttribution/data/IMDB/imdb62-test.txt',
                        help='The path to the dataset for testing')
    parser.add_argument('--output_path', metavar='output_path', type=str,
                        default='.', help='Where to save output from this run')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=5,
                        help='how many epochs to train for')
    parser.add_argument('--scheduler', metavar='scheduler', type=str, default='warmuplinear',
                        help='The learning rate scheduler, available are: constantlr, warmupconstant, warmuplinear,'
                             'warmupcosine, warmupcosinenewwithhardrestarts')
    parser.add_argument('--warmup_steps', metavar='warmup_steps', type=int, default=10000,
                        help='corresponds to learning rate schedule. ')
    parser.add_argument('--device', metavar='device', type=str, default='cuda', help='which device to use.')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=10, help='what batch size to use')
    parser.add_argument('--eval_batch_size', metavar='eval_batch_size', type=int, default=10,
                        help='what batch size to use for eval')
    parser.add_argument('--which_dataset', metavar='which_dataset', type=str, default='imdb',
                        help='Which dataset to use - just to simplify things for now')
    parser.add_argument('--which_loss_fn', metavar='which_loss_fn', type=str, default='BatchHardTripletLoss',
                        help='Options: BatchAllTripletLoss, BatchHardTripletLoss, BatchSemiHardTripletLoss,'
                             'BatchHardSoftMarginTripletLoss')
    parser.add_argument('--max_seq_length', metavar='max_seq_length', type=int, default=512,
                        help='What is the maximum sequence length to use with the model?')
    parser.add_argument('--auth_emb_dataset', metavar='auth_emb_dataset', type=str,
                        help='the path to the dataset to use to build the author embeddings during evaluation')

    parser.add_argument('--contrastive_fixed', action='store_true',
                        help='set if the contrastive training set be fixed instead of sampled')

    parser.add_argument('--test_eval_time', action='store_true',
                        help='run evaluation before training and report the time it takes')

    parser.add_argument('--multigpu', action='store_true',
                        help='try to use multiple gpus')

    parser.add_argument('--evaluation_steps', metavar='evaluation_steps', type=int, default=0,
                        help='if > 0, then evaluate every this number of steps')

    parser.add_argument('--distance_metric', metavar='distance_metric', type=str, default='euclidean',
                        help='which distance metric to use')

    parser.add_argument('--seed', metavar='seed', type=int, default=0,
                        help='set the seeds for randomness')

    parser.add_argument('--word_embedding_dim', metavar='word_embedding_dim', type=int, default=512,
                        help='the dimensionality of the word embeddings to use')

    parser.add_argument('--margin', metavar='margin', type=float, default=0.5,
                        help='the margin to use, for modified contrastive loss this is the different margin')
    parser.add_argument('--same_margin', metavar='same_margin', type=float, default=0.1,
                        help='only used in modified contrastive loss, the neighborhood for same author samples to be in')

    parser.add_argument('--distributed', action='store_true',
                        help='set this to try and use multiple GPUs')

    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='the devices to use when using multiple GPUs, or when loading a previously distributed model')

    parser.add_argument('--disregard_distributed', action='store_true',
                        help='for multi gpu to single gpu use')

    args = parser.parse_args()

    if args.auth_emb_dataset is None:
        args.auth_emb_dataset = args.train_dataset

    # set the seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
