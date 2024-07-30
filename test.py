# coding=gbk
import os
import torch
from models import *
from functions import *
from Bio import SeqIO

# ----------------------------hyper paramters----------------------------
checkpoint_path = './checkpoint/checkpoint.pth.tar'
input_path = './input_fasta/sequence.fasta'
output_path = './output/prediction.txt'
threshold = 0.4

# ----------------------------initialize trained model----------------------------
model = Network(feature_dim=136, class_num=216)
if torch.cuda.is_available():
    model_dict = torch.load(checkpoint_path)
    model = model.cuda()
    model.load_state_dict(model_dict)
    model.eval()
else:
    model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_dict)
    model.eval()
    
# ----------------------------preprocess model input----------------------------
records = list(SeqIO.parse(input_path, "fasta"))
model_input = []
for i in range(len(records)):
    sequence = str(records[i].seq)
    kmer_frequency = convert_kmer_reverse_complement(sequence, 4)
    model_input.append(list(kmer_frequency.values()))
model_input = torch.Tensor(model_input)

if torch.cuda.is_available():
    model_input = model_input.cuda()

# ----------------------------output model prediction----------------------------
index_path = './genus_index.txt'
dict_genus = {}
data = open(index_path, 'r').readlines()
for line in data:
    dict_genus[line.rstrip().split('\t')[1]] = line.rstrip().split('\t')[0]

_, output = model(model_input)
ones = torch.ones_like(output)
zeros = torch.zeros_like(output)
predictions = torch.where(output>=threshold, ones, zeros)

output_file = []

for i in range(len(records)):
    temp = str(records[i].id) + '\t'
    for j in range(predictions.size(1)):
        if predictions[i,j]==1:
            temp += dict_genus[str(j)] + '\t'
    output_file.append(temp + '\n')
    
f = open(output_path, 'w')
for line in output_file:
    f.write(line)
f.close()