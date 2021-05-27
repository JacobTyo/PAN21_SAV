from torch.utils.data import DataLoader
from torch import nn
from dsets.imdb import get_imdb_train_or_test, get_training_for_triplet_loss_imdb
from dsets.blog import get_blogs_train_or_test, get_training_for_triplet_loss_blogs
from dsets.fanfiction import get_ff_contrastive_fixed, get_training_for_triplet_loss_ff, get_ff_train_or_test
from AaEvaluator import AaEvaluator, ContrastiveEvaluator
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, evaluation

import mlflow
import argparse
import time
import os



def main(args):

    print('starting')

    # get the training dataset, and a training evaluator
    if args.which_dataset == 'imdb':
        get_triplet_loss = get_training_for_triplet_loss_imdb
        get_train_or_test = get_imdb_train_or_test
    elif args.which_dataset == 'blogs':
        get_triplet_loss = get_training_for_triplet_loss_blogs
        get_train_or_test = get_blogs_train_or_test
    elif args.which_dataset == 'fanfiction':
        if args.which_loss_fn == 'ContrastiveLoss':
            get_triplet_loss = get_ff_contrastive_fixed
            get_train_or_test = get_ff_contrastive_fixed
        else:
            get_triplet_loss = get_training_for_triplet_loss_ff
            get_train_or_test = get_ff_train_or_test
    else:
        assert False, 'Only the imdb and blogs dataasets are supported at the moment'

    # train_dataset = get_triplet_loss(args.train_dataset)
    # train_eval_dataset = get_train_or_test(args.train_dataset)
    # test_dataset = get_train_or_test(args.test_dataset)
    # if args.which_loss_fn == 'ContrastiveLoss':
    test_dataset = get_triplet_loss(args.test_dataset)
    print(f'the dataset is of length: {len(test_dataset)}')

    print('building model')

    # now build a model
    # now build a model
    if not args.model_path:
        word_embedding_model = models.Transformer('bert-base-cased', max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=args.device)
    else:
        model = SentenceTransformer(args.model_path, device=args.device)

    # set up an evaluator
    # set up an evaluator
    if args.which_loss_fn == 'ContrastiveLoss':
        evaluator = ContrastiveEvaluator(test_dataset)
    else:
        evaluator = AaEvaluator(train_eval_dataset, test_dataset)


    print('evaluating')
    print('samps/emb\t\taccuracy\t\telapsed time')

    # for samps_per_auth_embedding in [1]:
    # tmp = samps_per_auth_embedding if samps_per_auth_embedding is not None else -1
    start_time = time.time()
    # batch_size = samps_per_auth_embedding if samps_per_auth_embedding < 128 else 128
    results = evaluator(model=model)  #, output_path='testing', batch_size=batch_size, num_samples_per_auth_embedding=samps_per_auth_embedding)
    end_time = time.time()
    print(f'Took {end_time - start_time} seconds')
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for SiameseBert Authorship Attribution Evaluation')
    parser.add_argument('--run_name', metavar='run_name', type=str, help='The name to use for MLFlow logging')
    parser.add_argument('--model_path', metavar='model_path', type=str, default=None, help='The path to the model to evaluate')
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
    parser.add_argument('--which_dataset', metavar='which_dataset', type=str, default='imdb', help='which dataset to use')
    parser.add_argument('--which_loss_fn', metavar='which_loss_fn', type=str, default='BatchHardTripletLoss',
                        help='Options: BatchAllTripletLoss, BatchHardTripletLoss, BatchSemiHardTripletLoss,'
                             'BatchHardSoftMarginTripletLoss')
    args = parser.parse_args()

    main(args)
