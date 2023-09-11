import numpy as np
import pandas as pd
import operator
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from itertools import combinations
import re
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split

# DATALOADER
print('loading...')
data = torch.load('/home/gaia/progetto/dblp_v14/graph_data.pt')


print('Heterodata: ', data)
print('metadata: ', data.metadata())
print('nodi paper: ', data['paper'].num_nodes)
print('noti autori: ', data['author'].num_nodes)
print('nodi topic: ', data['topic'].num_nodes)
num_edges_cites = data.num_edges_dict[('paper', 'cites', 'paper')]
num_edges_writes = data.num_edges_dict[('author', 'writes', 'paper')]
num_edges_about = data.num_edges_dict[('paper', 'about', 'topic')]
print("edge writes", num_edges_writes)
print("edge cites", num_edges_cites)
print("edge about", num_edges_about)

author_attributes = data['author'].keys()
print("Attributes for 'author' node type:", author_attributes)

print('loading complete. Creating subgraph functions...')
k = 5  # num folds

def filter_data_by_year(dataset, year) -> HeteroData:
    papers = dataset['paper']
    mask = papers.year < year

    # Extract subgraph with filtered papers and their authors and topics and the papers they cited
    ids_papers = torch.where(mask)[0]
    filtered_by_year = dataset.subgraph({'paper': ids_papers})

    return filtered_by_year

# example
filtered = filter_data_by_year(data, year = 1973)