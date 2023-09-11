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

print('Reading csv files...')
papers = pd.read_csv('/home/gaia/progetto/dblp_v14/id_paper.csv', dtype={'p_id': int})
authors = pd.read_csv('/home/gaia/progetto/dblp_v14/id_author.csv')
topics = pd.read_csv('/home/gaia/progetto/dblp_v14/id_topic.csv')
writes = pd.read_csv('/home/gaia/progetto/dblp_v14/writes.csv')
about = pd.read_csv('/home/gaia/progetto/dblp_v14/about.csv')
cites = pd.read_csv('/home/gaia/progetto/dblp_v14/cites.csv')

print('Creating first merged db for topics and papers...')
papers_about_topics= papers.merge(about, on='p_id').merge(topics, on='t_id')
print(papers_about_topics.head())

print('Creating second merged db for authors and papers...')
authors_write_papers= papers.merge(writes, on='p_id').merge(authors, on='a_id')
print(authors_write_papers.head())

print('Creating third merged db for papers cited...')
papers_cite_papers = papers.merge(cites, left_on='p_id', right_on='p_id', how='inner').merge(papers, left_on='c_id', right_on='p_id', how='inner')
papers_cite_papers = papers_cite_papers.drop('c_id', axis=1).rename(columns={
    'p_id_x': 'p_id',
    'original_id_x': 'original_id',
    'title_x': 'title',
    'n_citations_x': 'n_citations',
    'year_x': 'year',
    'p_id_y': 'cited_p_id',
    'original_id_y': 'cited_original_id',
    'title_y': 'cited_title',
    'n_citations_y': 'cited_n_citations',
    'year_y': 'cited_year'
})
print(papers_cite_papers.head())

autori=list(set(authors['a_id']))
autori1=list(set(authors_write_papers['a_id']))
print(list(set(autori) - set(autori1)))

paper1=list(set(papers['p_id']))
paper2=list(set(authors_write_papers['p_id']))
print(list(set(paper1) - set(paper2))[:100])

# GRAFO AUTORI
print('Start graph creation...')
# Sort dataframe
authors_write_papers = authors_write_papers.sort_values(by="title", ascending=True)
print(authors_write_papers.head())

# CREAZIONE HETERODATA
data = HeteroData()

data['paper'].p_id = torch.tensor(papers['p_id'].values)
data['author'].a_id = torch.tensor(authors['a_id'].values)
data['topic'].t_id = torch.tensor(topics['t_id'].values)

# data['paper'].title = papers['title'].values
# data['author'].name = authors['name'].values
# data['topic'].topic_name = topics['topic_name'].values

data['paper'].num_citations = torch.tensor(papers['n_citations'].values)
data['paper'].year = torch.tensor(papers['year'].values)

data['paper', 'cites', 'paper'].edge_index = torch.tensor(cites[['p_id', 'c_id']].values.T)
data['author', 'writes', 'paper'].edge_index = torch.tensor(writes[['a_id', 'p_id']].values.T)
data['paper', 'about', 'topic'].edge_index = torch.tensor(about[['p_id', 't_id']].values.T)

# Set num_nodes attribute for each node type
data['paper'].num_nodes = data['paper']['p_id'].shape[0]
data['author'].num_nodes = data['author']['a_id'].shape[0]
data['topic'].num_nodes = data['topic']['t_id'].shape[0]

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

# funzione check grafo
def check_dataset(dataset): 
    # Remove invalid edges from edge_index
    for edge_type, edge_index in dataset.edge_index_dict.items():
        print("Removing invalid edges from edge_index...")
        num_nodes = dataset[edge_type[0]].num_nodes
        invalid_indices = torch.any(edge_index >= num_nodes, dim=0)
        dataset.edge_index_dict[edge_type] = edge_index[:, ~invalid_indices]

    # Remove duplicate edges from edge_index
    for edge_type, edge_index in dataset.edge_index_dict.items():
        print("Removing duplicate edges from edge_index...")
        unique_indices = torch.unique(edge_index, dim=1, return_inverse=True)[1]
        dataset.edge_index_dict[edge_type] = edge_index[:, unique_indices]

    # Remove nodes without labels
    for node_type, node_labels in dataset.node_attr_dict.items():
        print("Removing nodes without labels...")
        if node_labels is None or not torch.is_tensor(node_labels):
            print('Node without label')
            dataset[node_type].num_nodes = 0

    # Remove edges without labels
    for edge_type, edge_labels in dataset.edge_attr_dict.items():
        print("Removing edges without labels...")
        if edge_labels is None or not torch.is_tensor(edge_labels):
            print('Edge without label')
            dataset[edge_type[0]].num_edges[edge_type[1:]] = 0

    # Check node label shapes
    for node_type, node_labels in dataset.node_attr_dict.items():
        if node_labels.shape[0] != dataset[node_type].num_nodes:
            print(f"Inconsistent node label shape: {node_type}")

    # Check edge label shapes
    for edge_type, edge_labels in dataset.edge_attr_dict.items():
        if edge_labels.shape[1] != dataset.num_edges(edge_type):
            print(f"Inconsistent edge label shape: {edge_type}")


check_dataset(data)

# Heterodata:  HeteroData(
#   paper={
#     p_id=[5259857],
#     num_citations=[5259857],
#     year=[5259857],
#     num_nodes=5259857
#   },
#   author={
#     a_id=[3976339],
#     num_nodes=3976339
#   },
#   topic={
#     t_id=[320616],
#     num_nodes=320616
#   },
#   (paper, cites, paper)={ edge_index=[2, 32719868] },
#   (author, writes, paper)={ edge_index=[2, 16855406] },
#   (paper, about, topic)={ edge_index=[2, 36000986] }
# )
# metadata:  (['paper', 'author', 'topic'], [('paper', 'cites', 'paper'), ('author', 'writes', 'paper'), ('paper', 'about', 'topic')])
# nodi paper:  5259857
# noti autori:  3976339
# nodi topic:  320616
# edge writes 16855406
# edge cites 32719868
# edge about 36000986


#Save dataset to file
def save_graph_to_file(data):

    file_path = '/home/gaia/progetto/dblp_v14/graph_data.pt'
    torch.save(data, file_path)

save_graph_to_file(data)