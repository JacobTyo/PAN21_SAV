import time
import numpy as np
import torch
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from sklearn import metrics

bd = BatchHardTripletLossDistanceFunction.eucledian_distance
sd = SiameseDistanceMetric.EUCLIDEAN


a = torch.rand((100, 10))  # 100 embeddings of dimension 10
b = torch.rand((100, 10))

# if we want to compare a_i and b_i = then we will get 100 embeddings, two methods
# using siamese distance metric

start = time.time()
sdd = sd(a, b)
end = time.time()

print(f'the sd method took {end - start} seconds')
print(sdd)
print(len(sdd))

start = time.time()
c = torch.cat((a, b), dim=0)
bdd = bd(c)
these_i = np.array([x for x in range(len(a))])
these_j = these_i + len(a)
bdd = [bdd[i, j] for i, j in zip(these_i, these_j)]
end = time.time()
print(f'the bd method took {end - start} seconds')
print(bdd)
print(len(bdd))

# test the roc_auc_score
y_true = np.array([0, 0, 0, 1, 1, 1])
# the prediction needs to correspond to the probability that the label is of the greater class
y_pred1 = np.array([1, 5, 2, 2, 2, 4])
y_pred2 = y_pred1/np.max(y_pred1)
not_normalized = metrics.roc_auc_score(y_true, y_pred1)
normalized = metrics.roc_auc_score(y_true, y_pred2)

print(f'not_normlalized: {not_normalized}')
print(f'normalized: {normalized}')


print('f1')

y3 = [1 if x > 0.5 else 0 for x in y_pred2]
#not_normalized = metrics.f1_score(y_true, y_pred1)
normalized = metrics.f1_score(y_true, y3)

#print(f'not_normlalized: {not_normalized}')
print(f'normalized: {normalized}')