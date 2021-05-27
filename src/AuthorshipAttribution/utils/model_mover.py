import torch
from collections import OrderedDict
from sentence_transformers import models, SentenceTransformer
import argparse


# get command line args
parser = argparse.ArgumentParser(description='Get args for SiameseBert Authorship Attribution Evaluation')

parser.add_argument('--input_model', metavar='input_model', type=str, help='the distributed model to start from')
parser.add_argument('--output_model', metavar='output_model', type=str, help='where to save the output model')

args = parser.parse_args()

input_model = args.input_model
output_model = args.output_model

print(f'building initial model from: {input_model}')
word_embedding_model = models.Transformer('bert-base-cased')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                           out_features=256,
                           activation_function=torch.nn.ReLU())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device='cpu')

print('getting the state dict')
state_dict = torch.load(input_model)

print('building the new state dict')
new_state_dict = OrderedDict()
for key in state_dict.keys():
    if '0.model' not in key:
        new_state_dict[key] = state_dict[key]

print('loading the new state dict into the model')
model.load_state_dict(new_state_dict)

print('ensuring model is on cpu')
model.to('cpu')

print(f'saving model to: {output_model}')
torch.save(model.state_dict(), output_model)
