from sentence_transformers.evaluation import SentenceEvaluator
import torch
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
import mlflow
import math
import time
import random
import torch.nn.functional as F
from utils import eval_metrics
from sklearn import metrics
from tqdm import tqdm
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
import numpy as np
from matplotlib import pyplot as plt


class AaEvaluator(SentenceEvaluator):

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 batch_size=1,
                 distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance
                 ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.distance_metric = distance_metric

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, batch_size: int = -1,
                 num_samples_per_auth_embedding: int = 1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.
        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """

        if batch_size == -1:
            batch_size = self.batch_size

        og_batch_size = batch_size
        batch_size = num_samples_per_auth_embedding if num_samples_per_auth_embedding < batch_size else batch_size

        start_time = time.time()

        auth_embeddings = {}
        auth_to_key, key_to_auth = {}, {}
        key_num = 0
        print('building author embeddings')
        for auth, texts in tqdm(self.train_dataset.items()):
            auth_to_key[auth] = key_num
            key_to_auth[key_num] = auth
            key_num += 1

            samples_used_for_this_auth = 0
            auth_outputs = []

            for i in range(int(math.ceil(len(texts) / batch_size))):

                end_idx = batch_size * (i + 1) if batch_size * (i + 1) < len(texts) else len(texts)
                # FOR TESTING
                if batch_size == 1:
                    i = random.randint(0, len(texts))
                    end_idx = i + 1
                output = model.encode(texts[i * batch_size:end_idx], convert_to_numpy=True)

                auth_outputs.append(output)
                samples_used_for_this_auth += batch_size

                if num_samples_per_auth_embedding is not None:

                    if samples_used_for_this_auth >= num_samples_per_auth_embedding:
                        # skip the rest of the samples for this author
                        break

            # now I can compute the average embedding for this author
            # flatten the outputs
            # if  num_samples_per_auth_embedding is one, then no average
            if num_samples_per_auth_embedding == 1:
                auth_embeddings[auth_to_key[auth]] = np.array(auth_outputs[0])
            else:
                tmp = []
                for x in auth_outputs:
                    try:
                        tmp.append(x[1])
                    except Exception:
                        pass
                auth_outputs = np.array(tmp)
                auth_embeddings[auth_to_key[auth]] = np.mean(auth_outputs, axis=0)

        # reset batch size:
        batch_size = og_batch_size

        # now that this is completed, we will run each test text through the embedding, then compare cosine distance,
        # and predict the author who's user embedding is closest
        number_predictions = 0
        number_correct_predictions = 0

        top_n_performance = dict([[n, 0] for n in range(len(auth_embeddings.keys()))])

        # conf_matrix_n = len(self.train_dataset.keys())
        # confusion_matrix = np.zeros((conf_matrix_n, conf_matrix_n))

        # keep track of perforamnce per author, format is dict with key author and tuple value of num_right, total
        macro_averaged = {}

        print('evaluating on the test dataset')

        for auth, texts in tqdm(self.test_dataset.items()):

            # for text in texts:
            for i in range(int(math.ceil(len(texts) / batch_size))):

                end_idx = batch_size * (i + 1) if batch_size * (i + 1) < len(texts) else len(texts)
                text_embedding = model.encode(texts[batch_size * i:end_idx], convert_to_numpy=True)

                # now for each predition, predict which author it should be
                conf_matrix_keys = {}

                # build matrix for distance calculations
                auth_distance_calc_matrix = np.zeros((len(self.train_dataset.keys()) + 1, text_embedding[0].shape[0]))
                
                auths = list(self.train_dataset.keys())

                for j in range(len(auths)):
                    conf_matrix_keys[j] = auths[j]
                    try:
                        auth_distance_calc_matrix[j, :] = auth_embeddings[auth_to_key[auths[j]]]
                    except:
                        # not sure why this happens occasionally
                        auth_distance_calc_matrix[j, :] = 99999*np.ones_like(auth_distance_calc_matrix[j, :])

                # is there a way I can do this without looping?
                for pred in text_embedding:

                    # now add text embedding as last entry in matrix
                    auth_distance_calc_matrix[-1, :] = pred

                    scrs = self.distance_metric(torch.Tensor(auth_distance_calc_matrix))

                    # we only need the last row
                    scrs = scrs[-1, :]

                    # ignore the zero entry - that is comparison with itself (should be the last entry)
                    assert scrs[-1] > -1e06 and scrs[-1] < 1e-6, 'the zeroth entry was not the last one. .. '
                    scrs = scrs[:-1]

                    # scrs[scrs == 0] += np.max(scrs.numpy())  # I should just add the max here instead

                    # get predicted author by whoever has smallest score, ignoring the distance from itself
                    minidx = np.argmin(scrs.numpy())
                    # pred_dist = scrs.numpy()[minidx]
                    # # how do we convert the dist to a score? just store dists for now
                    # Actually, do these metric make since? In my case I think no. I think these only make since
                    # when using the predefined pairs as they do in the compeition. This complicates things.
                    # probably need to write a new evaluator for them.
                    # Probably best to use two evaluations then, one on preset pairs, another on a regular test set.
                    predicted_auth = conf_matrix_keys[minidx]

                    # instead of just predicting the most probable author, return a ranking
                    scrs = [[s, conf_matrix_keys[i]] for i, s in enumerate(scrs.numpy().tolist())]
                    ranking = sorted(scrs, reverse=True)
                    ranked_auths = [r[1] for r in ranking]

                    number_predictions += 1
                    if auth not in macro_averaged.keys():
                        macro_averaged[auth] = [0, 0]
                    macro_averaged[auth][1] += 1

                    # were we correct in our prediction?
                    if predicted_auth == auth:
                        number_correct_predictions += 1
                        macro_averaged[auth][0] += 1

                    # track top n performance
                    # what index of the ranking is the true author?
                    n = ranked_auths.index(auth)

                    for k in range(n, len(top_n_performance.keys())):
                        top_n_performance[k] += 1

                    # update confusion matrix - format is [predicted, true]
                    # confusion_matrix[auth_to_key[predicted_auth]][auth_to_key[auth]] += 1

        accuracy = number_correct_predictions / number_predictions

        # now change top_n_performance into percentages
        for key, val in top_n_performance.items():
            top_n_performance[key] = val / number_predictions

        print('top n stuff')
        for key, val in top_n_performance.items():
            print(key, val)

        # get accuracy per author, then average
        macro_averaged_accuracy = np.mean([x[0]/x[1] for auth, x in macro_averaged.items()])

        # add the other metrics - should store y_true and y_pred functions, then just pass to the functions

        print('evaluation finished:')
        print(f'Epoch: {epoch}, \tStep: {steps}, \tAccuracy: {accuracy}, Macro Average: {macro_averaged_accuracy}, '
              f'elapsed time: {time.time() - start_time}')

        try:
            # check epoch first?
            if epoch > -1:
                log_step = epoch
            elif isinstance(steps, int):
                log_step = steps
            elif isinstance(steps, list):
                log_step = steps[-1]
            else:
                log_step = -1

            mlflow.log_metric('eval_accuracy', accuracy, step=log_step)
            mlflow.log_metric('eval_macro_accuracy', macro_averaged_accuracy, step=log_step)
            mlflow.log_metric('top1', top_n_performance[1], step=log_step)
            mlflow.log_metric('top2', top_n_performance[2], step=log_step)
            mlflow.log_metric('top3', top_n_performance[3], step=log_step)
            mlflow.log_metric('top4', top_n_performance[4], step=log_step)
            mlflow.log_metric('top5', top_n_performance[5], step=log_step)
            mlflow.log_metric('top10', top_n_performance[10], step=log_step)

        except Exception as e:
            print('exception trying to log metrics')
            print(e)

        return accuracy



class ContrastiveEvaluator(SentenceEvaluator):

    def __init__(self, eval_dataset, batch_size: int = 100,
                 distance_metric=SiameseDistanceMetric.EUCLIDEAN,
                 threshold: float = 0.5, find_threshold: bool = False,
                 name: str = ''):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.distance_metric = distance_metric  # lambda x, y: 1-F.cosine_similarity(x, y)
        self.batch_size = batch_size
        self.threshold = threshold
        self.find_threshold = find_threshold
        self.name = name

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, batch_size: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.
        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """

        # just encode all of the samples, then use some comparison to determine if they are the same or different?
        # Can I just wrap this with the contrastive loss thing somehow? I think so, try it

        if batch_size == -1:
            batch_size = self.batch_size
        print('evaluating')
        start_time = time.time()
        truth = []
        emb1, emb2 = [], []

        with torch.no_grad():
            if batch_size > 1:
                for i in tqdm(range(int(math.ceil(len(self.eval_dataset)/batch_size)))):
                    additional = batch_size if (i+1)*batch_size < len(self.eval_dataset) else len(self.eval_dataset)-1
                    batch1 = [sample.texts[0] for sample in self.eval_dataset[i*batch_size:i*batch_size+additional]]
                    batch2 = [sample.texts[1] for sample in self.eval_dataset[i*batch_size:i*batch_size+additional]]
                    labels = [sample.label for sample in self.eval_dataset[i*batch_size:i*batch_size+additional]]
                    emb1.extend(model.encode(batch1))
                    emb2.extend(model.encode(batch2))
                    truth.extend(labels)
            else:
                for sample in tqdm(self.eval_dataset):
                    emb1.append(model.encode(sample.texts[0]))
                    emb2.append(model.encode(sample.texts[1]))
                    truth.append(sample.label)

            # at this point, we have two lists of embeddings. We want to compare each embeddings, elementwise, to
            # determine if they were written by the same author or not (i.e. get the distance between them)
            # we also want to keep track of all of them for a ranking?
            predictions = self.distance_metric(torch.tensor(emb1), torch.tensor(emb2))

        # need to add some calibration here - especially using eucledian as the output magnitude will be quite different
        # the predictions here are good I think, but they need to be changed to a similarity score, for now just do this by max(x) - x
        similarities = torch.max(predictions) - predictions
        normalized_similarities = similarities/torch.max(similarities)
        normalized_similarities = normalized_similarities.numpy()

        # for now, just assume 50% same auth and 50% different auth - this seems like cheating but what can you do?
        # threshold = sorted(normalized_similarities)[int(len(normalized_similarities)/2)]
        threshold = self.threshold

        if self.find_threshold:
            # save this numpy array for testing later
            print('saving similairities and truth values')
            np.save('sims' + self.name + '.npy', normalized_similarities)

            np.save('truth' + self.name + '.npy', np.array(truth))
            print('predictions saved')

            # first plot the score distribution: how do we make a histogram here?
            pos_data, neg_data = [], []
            for sim, lbl in zip(normalized_similarities, truth):
                if lbl == 1:
                    pos_data.append(sim)
                elif lbl == 0:
                    neg_data.append(sim)
                else:
                    assert False, 'problems'

            plt.hist(pos_data, bins=25, alpha=0.5, label='same_author')
            plt.hist(neg_data, bins=25, alpha=0.5, label='diff_author')
            plt.title('Score distributions for same and different author pairs')
            plt.xlabel('score')
            plt.ylabel('count')

            plt.savefig('hist_differences')

            # set some ranges to sweep, and shuffle them - maybe see more interesting results before hand?
            all_thresholds = [x for x in np.linspace(0.0001, 1, 100)]
            all_possible_scores = []
            for threshold in all_thresholds:
                results = {
                    'auc': metrics.roc_auc_score(truth, normalized_similarities),
                    'c@1': eval_metrics.c_at_1(truth, normalized_similarities, threshold),
                    'f_05_u': eval_metrics.f_05_u_score(truth, normalized_similarities, threshold=threshold),
                    'F1': eval_metrics.f1(truth, normalized_similarities)
                }
                results['overall'] = np.mean(list(results.values()))
                binarized_predictions = eval_metrics.binarize(normalized_similarities, threshold)
                correct_predictions = [1 if x == y else 0 for x, y in zip(truth, binarized_predictions)]
                results['accuracy'] = sum(correct_predictions) / len(correct_predictions)

                all_possible_scores.append((threshold, results))
            # now log all of the results to mlflow
            best_auc = [0, 0]
            best_acc = [0, 0]
            best_overall = [0, 0]
            best_f05 = [0, 0]
            best_ca1 = [0, 0]
            best_f1 = [0, 0]
            for thresh, res in all_possible_scores:
                mlflow.log_metric('threshtest/accuracy', res['accuracy'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/auc', res['auc'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/ca1', res['c@1'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/f_05_u', res['f_05_u'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/F1', res['F1'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/overall', res['overall'], step=int(1000*thresh))

                # track optimal measures
                best_auc = best_auc if best_auc[0] > res['auc'] else [res['auc'], thresh]
                best_acc = best_acc if best_acc[0] > res['accuracy'] else [res['accuracy'], thresh]
                best_overall = best_overall if best_overall[0] > res['overall'] else [res['overall'], thresh]
                best_f05 = best_f05 if best_f05[0] > res['f_05_u'] else [res['f_05_u'], thresh]
                best_ca1 = best_ca1 if best_ca1[0] > res['c@1'] else [res['c@1'], thresh]
                best_f1 = best_f1 if best_f1[0] > res['F1'] else [res['F1'], thresh]

            mlflow.log_metric('threshtest/opt_accuracy', best_acc[0], step=int(1000*best_acc[1]))
            mlflow.log_metric('threshtest/opt_auc', best_auc[0], step=int(1000*best_auc[1]))
            mlflow.log_metric('threshtest/opt_ca1', best_ca1[0], step=int(1000*best_ca1[1]))
            mlflow.log_metric('threshtest/opt_f_05_u', best_f05[0], step=int(1000*best_f05[1]))
            mlflow.log_metric('threshtest/opt_F1', best_f1[0], step=int(1000*best_f1[1]))
            mlflow.log_metric('threshtest/opt_overall', best_overall[0], step=int(1000*best_overall[1]))

            return best_auc

        # now select best threshold for final reporting?

        binarized_predictions = eval_metrics.binarize(normalized_similarities, threshold)

        # this doesn't work with raw scores
        # results = eval_metrics.evaluate_all(truth, binarized_predictions)
        # for now just give raw distances as input
        results = {
            'auc': metrics.roc_auc_score(truth, normalized_similarities),
            'c@1': eval_metrics.c_at_1(truth, normalized_similarities, threshold),
            'f_05_u': eval_metrics.f_05_u_score(truth, normalized_similarities, threshold=threshold),
            'F1': eval_metrics.f1(truth, normalized_similarities)
        }
        results['overall'] = np.mean(list(results.values()))

        correct_predictions = [1 if x == y else 0 for x, y in zip(truth, binarized_predictions)]
        results['accuracy'] = sum(correct_predictions)/len(correct_predictions)
        print(f'Evaluation took {time.time() - start_time} seconds. Scores at step {steps} and epoch {epoch} are:')
        print(f"accuracy: {results['accuracy']}, auc: {results['auc']}, c@1: {results['c@1']}, f_05_u: {results['f_05_u']}, F1: {results['F1']}, overall: {results['overall']}")

        try:
            if epoch > -1:
                log_step = epoch
            elif isinstance(steps, int):
                log_step = steps
            elif isinstance(steps, list):
                log_step = steps[-1]
            else:
                log_step = -1

            mlflow.log_metric('accuracy', results['accuracy'], step=log_step)
            mlflow.log_metric('auc', results['auc'], step=log_step)
            mlflow.log_metric('ca1', results['c@1'], step=log_step)
            mlflow.log_metric('f_05_u', results['f_05_u'], step=log_step)
            mlflow.log_metric('F1', results['F1'], step=log_step)
            mlflow.log_metric('overall', results['overall'], step=log_step)

        except Exception as e:
            print('exception trying to log metrics')
            print(e)

        return results['auc']

    def set_threshold(self, threshold):
        self.threshold = threshold




class ContrastiveChunkerEvaluator(SentenceEvaluator):

    def __init__(self, eval_dataset, batch_size: int = 100,
                 distance_metric=SiameseDistanceMetric.EUCLIDEAN,
                 threshold: float = 0.5, find_threshold: bool = False,
                 name: str = ''):
        self.eval_dataset = eval_dataset
        self.distance_metric = distance_metric  # lambda x, y: 1-F.cosine_similarity(x, y)
        self.batch_size = batch_size
        self.threshold = threshold
        self.find_threshold = find_threshold
        self.name = name

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, batch_size: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.
        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """

        # just encode all of the samples, then use some comparison to determine if they are the same or different?
        # Can I just wrap this with the contrastive loss thing somehow? I think so, try it

        if batch_size == -1:
            batch_size = self.batch_size
        print('evaluating')
        start_time = time.time()
        truth = []
        emb1, emb2 = [], []

        with torch.no_grad():
            for sample in tqdm(self.eval_dataset):
                if len(sample.texts[0]) > 500:
                    # chunk the sample and get separate predictions - just average them together for now
                    txt1_embeddings = []
                    for i in range(0, len(sample.texts[0]), 500):
                        end_idx = i + 500 if i + 500 < len(sample.texts[0]) else len(sample.texts[0])
                        txt1_embeddings.append(model.encode(sample.texts[0][i:end_idx]))

                    txt1_emb = np.mean(txt1_embeddings, axis=0)
                else:
                    txt1_emb = model.encode(sample.texts[0])

                if len(sample.texts[1]) > 500:
                    # chunk the sample and get separate predictions - just average them together for now
                    txt2_embeddings = []
                    for i in range(0, len(sample.texts[1]), 500):
                        end_idx = i + 500 if i + 500 < len(sample.texts[1]) else len(sample.texts[1])
                        txt2_embeddings.append(model.encode(sample.texts[1][i:end_idx]))

                    txt2_emb = np.mean(txt2_embeddings, axis=0)
                else:
                    txt2_emb = model.encode(sample.texts[1])


                emb1.append(txt1_emb)
                emb2.append(txt2_emb)
                truth.append(sample.label)

            # at this point, we have two lists of embeddings. We want to compare each embeddings, elementwise, to
            # determine if they were written by the same author or not (i.e. get the distance between them)
            # we also want to keep track of all of them for a ranking?
            predictions = self.distance_metric(torch.tensor(emb1), torch.tensor(emb2))

        # need to add some calibration here - especially using eucledian as the output magnitude will be quite different
        # the predictions here are good I think, but they need to be changed to a similarity score, for now just do this by max(x) - x
        similarities = torch.max(predictions) - predictions
        normalized_similarities = similarities/torch.max(similarities)
        normalized_similarities = normalized_similarities.numpy()

        # for now, just assume 50% same auth and 50% different auth - this seems like cheating but what can you do?
        # threshold = sorted(normalized_similarities)[int(len(normalized_similarities)/2)]
        threshold = self.threshold

        if self.find_threshold:
            # save this numpy array for testing later
            print('saving similairities and truth values')
            np.save('sims' + self.name + '.npy', normalized_similarities)

            np.save('truth' + self.name + '.npy', np.array(truth))
            print('predictions saved')

            # first plot the score distribution: how do we make a histogram here?
            pos_data, neg_data = [], []
            for sim, lbl in zip(normalized_similarities, truth):
                if lbl == 1:
                    pos_data.append(sim)
                elif lbl == 0:
                    neg_data.append(sim)
                else:
                    assert False, 'problems'

            plt.hist(pos_data, bins=25, alpha=0.5, label='same_author')
            plt.hist(neg_data, bins=25, alpha=0.5, label='diff_author')
            plt.title('Score distributions for same and different author pairs')
            plt.xlabel('score')
            plt.ylabel('count')

            plt.savefig('hist_differences')

            # set some ranges to sweep, and shuffle them - maybe see more interesting results before hand?
            all_thresholds = [x for x in np.linspace(0.0001, 1, 100)]
            all_possible_scores = []
            for threshold in all_thresholds:
                results = {
                    'auc': metrics.roc_auc_score(truth, normalized_similarities),
                    'c@1': eval_metrics.c_at_1(truth, normalized_similarities, threshold),
                    'f_05_u': eval_metrics.f_05_u_score(truth, normalized_similarities, threshold=threshold),
                    'F1': eval_metrics.f1(truth, normalized_similarities)
                }
                results['overall'] = np.mean(list(results.values()))
                binarized_predictions = eval_metrics.binarize(normalized_similarities, threshold)
                correct_predictions = [1 if x == y else 0 for x, y in zip(truth, binarized_predictions)]
                results['accuracy'] = sum(correct_predictions) / len(correct_predictions)

                all_possible_scores.append((threshold, results))
            # now log all of the results to mlflow
            best_auc = [0, 0]
            best_acc = [0, 0]
            best_overall = [0, 0]
            best_f05 = [0, 0]
            best_ca1 = [0, 0]
            best_f1 = [0, 0]
            for thresh, res in all_possible_scores:
                mlflow.log_metric('threshtest/accuracy', res['accuracy'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/auc', res['auc'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/ca1', res['c@1'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/f_05_u', res['f_05_u'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/F1', res['F1'], step=int(1000*thresh))
                mlflow.log_metric('threshtest/overall', res['overall'], step=int(1000*thresh))

                # track optimal measures
                best_auc = best_auc if best_auc[0] > res['auc'] else [res['auc'], thresh]
                best_acc = best_acc if best_acc[0] > res['accuracy'] else [res['accuracy'], thresh]
                best_overall = best_overall if best_overall[0] > res['overall'] else [res['overall'], thresh]
                best_f05 = best_f05 if best_f05[0] > res['f_05_u'] else [res['f_05_u'], thresh]
                best_ca1 = best_ca1 if best_ca1[0] > res['c@1'] else [res['c@1'], thresh]
                best_f1 = best_f1 if best_f1[0] > res['F1'] else [res['F1'], thresh]

            mlflow.log_metric('threshtest/opt_accuracy', best_acc[0], step=int(1000*best_acc[1]))
            mlflow.log_metric('threshtest/opt_auc', best_auc[0], step=int(1000*best_auc[1]))
            mlflow.log_metric('threshtest/opt_ca1', best_ca1[0], step=int(1000*best_ca1[1]))
            mlflow.log_metric('threshtest/opt_f_05_u', best_f05[0], step=int(1000*best_f05[1]))
            mlflow.log_metric('threshtest/opt_F1', best_f1[0], step=int(1000*best_f1[1]))
            mlflow.log_metric('threshtest/opt_overall', best_overall[0], step=int(1000*best_overall[1]))

            return best_auc

        # now select best threshold for final reporting?

        binarized_predictions = eval_metrics.binarize(normalized_similarities, threshold)

        # this doesn't work with raw scores
        # results = eval_metrics.evaluate_all(truth, binarized_predictions)
        # for now just give raw distances as input
        results = {
            'auc': metrics.roc_auc_score(truth, normalized_similarities),
            'c@1': eval_metrics.c_at_1(truth, normalized_similarities, threshold),
            'f_05_u': eval_metrics.f_05_u_score(truth, normalized_similarities, threshold=threshold),
            'F1': eval_metrics.f1(truth, normalized_similarities)
        }
        results['overall'] = np.mean(list(results.values()))

        correct_predictions = [1 if x == y else 0 for x, y in zip(truth, binarized_predictions)]
        results['accuracy'] = sum(correct_predictions)/len(correct_predictions)
        print(f'Evaluation took {time.time() - start_time} seconds. Scores at step {steps} and epoch {epoch} are:')
        print(f"accuracy: {results['accuracy']}, auc: {results['auc']}, c@1: {results['c@1']}, f_05_u: {results['f_05_u']}, F1: {results['F1']}, overall: {results['overall']}")

        try:
            if epoch > -1:
                log_step = epoch
            elif isinstance(steps, int):
                log_step = steps
            elif isinstance(steps, list):
                log_step = steps[-1]
            else:
                log_step = -1

            mlflow.log_metric('accuracy', results['accuracy'], step=log_step)
            mlflow.log_metric('auc', results['auc'], step=log_step)
            mlflow.log_metric('ca1', results['c@1'], step=log_step)
            mlflow.log_metric('f_05_u', results['f_05_u'], step=log_step)
            mlflow.log_metric('F1', results['F1'], step=log_step)
            mlflow.log_metric('overall', results['overall'], step=log_step)

        except Exception as e:
            print('exception trying to log metrics')
            print(e)

        return results['auc']

