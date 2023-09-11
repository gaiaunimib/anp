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

def create_author_folds(data, k):
    if 'author' not in data.node_types:
        print("Author node type not found in data.")
        return None

    # conta autori per fold
    author_data = data['author']
    ids_authors = author_data['a_id']
    num_authors = len(ids_authors)
    num_authors_in_fold = num_authors // k
    remaining = num_authors % k

    rand = np.random.default_rng(seed=None)  # inizializza random generator
    shuffled_authors = rand.permutation(ids_authors)

    author_folds = []  # usa indici per identificare che autori vanno nei fold
    start = 0
    for i in range(k):
        if remaining != 0:
            fold_size = 1
            remaining -= 1
        else:
            fold_size = 0

        fold_size += num_authors_in_fold
        end = start + fold_size
        fold = shuffled_authors[start:end]
        author_folds.append(fold)
        start = end

    return author_folds

def filter_data_by_year(dataset, year) -> HeteroData:
    papers = dataset['paper']
    mask = papers.year < year

    # Extract subgraph with filtered papers and their authors and topics and the papers they cited
    subgraph = dataset.subgraph({'paper': mask})
    return subgraph

# example
filtered = filter_data_by_year(data, year = 1973)

def extract_subgraph(data, author_fold, year):
    # Step 1: Filter data by year
    filtered_data = filter_data_by_year(data, year)

    # Initialize an empty subgraph
    subgraph = HeteroData()

    # Step 2: Extract authors and their neighboring nodes
    author_nodes = []
    paper_nodes = []
    topic_nodes = []
    edge_types = [('author', 'writes', 'paper'), ('paper', 'about', 'topic')]

    for author_id in author_fold:
        # Extract author nodes
        author_nodes.append(author_id)

        # Extract neighboring paper nodes
        for edge_type in edge_types:
            source_type, _, dest_type = edge_type
            edge_index = filtered_data.edge_index_dict[edge_type]

            if source_type == 'author':
                relevant_edges = edge_index[:, filtered_data['author'].a_id[author_id]]
            elif dest_type == 'author':
                relevant_edges = edge_index[:, filtered_data['author'].a_id[author_id]]
            else:
                continue

            paper_nodes.extend(relevant_edges[1].tolist())  # Collect paper nodes

    # Step 3: Extract neighboring topic nodes
    for edge_type in edge_types:
        if edge_type[1] == 'about':
            source_type, _, dest_type = edge_type
            edge_index = filtered_data.edge_index_dict[edge_type]

            for paper_node in paper_nodes:
                relevant_edges = edge_index[:, paper_node]
                topic_nodes.extend(relevant_edges[1].tolist())  # Collect topic nodes

    # Step 4: Add nodes to the subgraph
    subgraph['author'] = filtered_data['author'][author_nodes]
    subgraph['paper'] = filtered_data['paper'][paper_nodes]
    subgraph['topic'] = filtered_data['topic'][topic_nodes]

    # Define the edges between nodes (you may need to adjust this based on your edge types)
    subgraph[('author', 'writes', 'paper')] = filtered_data[('author', 'writes', 'paper')][0][:, paper_nodes]
    subgraph[('paper', 'about', 'topic')] = filtered_data[('paper', 'about', 'topic')][0][:, topic_nodes]

    return subgraph

# Example usage:
author_fold = create_author_folds(data, k)[2]  # Choose a fold (e.g., fold 2)
year = 1973  # Choose a year
subgraph = extract_subgraph(data, author_fold, year)


# Access nodes and edges in the subgraph as needed:
author_nodes = subgraph['author']
paper_nodes = subgraph['paper']
topic_nodes = subgraph['topic']
writes_edges = subgraph['author', 'writes', 'paper']
about_edges = subgraph['paper', 'about', 'topic']

#
# #FILTRA X ANNO VECCHIO
# def filter_data(dataset, year) -> HeteroData:
#     #crea maschera indici papers
#     mask_papers = (dataset['paper']['year'] <= year).nonzero(as_tuple=False).view(-1)
#     #dove nonzero e view trasformano il tensore in lista a una dimensione
#
#     #filtra in base a maschera
#     filtered_dataset = dataset.subgraph({'paper': mask_papers})
#
#     return filtered_dataset

# # prova
# dataset_ridotto = filter_data(data, 1972)
# print('Dataset 1972: ', dataset_ridotto)


# # FUNZIONE CHE DIVIDE IL GRAFO IN BASE A ANNO E A FOLD
# # E IN TEORIA MANTIENE AUTORI COLLEGATI INSIEME VECCHIA
# def split_graph(data, fold, year):
#     filtered_dataset = HeteroData()
#     filtered_dataset = filter_data(data, year)
#     print(type(filtered_dataset))
#
#     print('Dataset filtrato:')
#     print(filtered_dataset)
#     print('metadata: ', filtered_dataset.metadata())
#     print('Nodes in "paper" type:', filtered_dataset['paper']['p_id'].shape[0])
#     print('Nodes in "author" type:', filtered_dataset['author']['a_id'].shape[0])
#     print('Nodes in "topic" type:', filtered_dataset['topic']['t_id'].shape[0])
#     num_edges_cites = filtered_dataset.num_edges_dict[('paper', 'cites', 'paper')]
#     num_edges_writes = filtered_dataset.num_edges_dict[('author', 'writes', 'paper')]
#     num_edges_about = filtered_dataset.num_edges_dict[('paper', 'about', 'topic')]
#     print("edge writes", num_edges_writes)
#     print("edge cites", num_edges_cites)
#     print("edge about", num_edges_about)
#
#     if filtered_dataset is None:
#         print("Filtered dataset is None. Skipping further processing.")
#         return None
#
# # cose che facevo ma che ora non capisco cosa intendevo fare
#     # data_size = len(filtered_dataset) #devo dividere tutto il dataset...
#     # come trovo size di tutto oeifnaoeikfnaggg
#
# # fine cose che facevo ma che ora non capisco
#
#     folds = create_author_folds(filtered_dataset, k)
#
#     authors_extracted = folds[fold]
#     subgraph = HeteroData()
#
#     for edge in data.edge_index_dict: # dict che contiene indici per gli edges dei vari tipi
#         # keys sono tuple (tipo_edge_e_nodi_relativi) e values sono i tensori dell'edge.
#         # La tupla in keys ad esempio è (author, writes, paper)
#         # Di ogni edge salviamo il tipo di nodo di partenza (source) e finale (destination)
#         # Non ci interessa di comprendere l'edge type se è writes cites etc
#         # ma dobbiamo capire con che nodi source e dest lavoriamo.
#         # Degli edges salviamo solo l'index.
#
#         # Quindi di ogni edge estraiamo source type e destination e controlliamo
#         # se è author; in tal caso estrae gli edge attorno se gli author rientrano nella
#         # lista di authors del fold. Se l'edge ha authors che rientrano, allora è
#         # rilevante e viene memorizzato in edges_rilevanti. Se l'edge ha source node e dest
#         # di cui nessuno è author, si skippa tranquillamente
#
#         # a partire da edges_rilevanti si rifà il nuovo subgraph filtrato
#
#         source_type, _, dest_type = edge #dell'edge type non ci interessa
#         edge_index = data.edge_index_dict[edge]
#
#         relevant_edges = []
#         if source_type== 'author':
#             relevant_edges = edge_index[:, data['author'].a_id[authors_extracted]]
#         # dove data['author].a_id è tensore che contiene tutti gli id di tutti gli autori
#         # per ottenere solo gli id degli autori dell'n-esimo fold allora passo indexing [authors_extracted]
#         # quindi in totale data[...] ha tutti gli a_id degli autori estratti
#
#         # poi edge_index contiene 2 dimensioni, la prima è source e la seconda è dest
#         # di tutti gli edges del tipo scelto sopra. Quindi con edge_index[:, ...]
#         # seleziono tutti gli edges che come dest hanno un autore che abbia l'a_id tra i prescelti
#
#         elif dest_type == 'author':
#             relevant_edges = edge_index[:, data['author'].a_id[authors_extracted]]
#         # così ho fatto lo stesso per dest che sono nodi autore
#         else: # source e dest non sono autori quindi ignoro e proseguo
#             continue
#
#         subgraph[edge].edge_index = relevant_edges
#         # cioè al tipo di edge selezionato all'inizio prima degli if assegnamo come
#         # nuovo edge index il nostro relevant edges.
#         # iterando in questo for faccio girare tutti i tipi diversi di edges
#         # come writes, cites, etc e sostituisco la lista completa originale con
#         # il nuovo relevant edges creato
#
#     return subgraph


#subgraph = extract_subgraph(data, fold=2, year=1973)

print("Nodes in 'author' type:", subgraph['author'].num_nodes)
print("Edges in 'author', 'writes', 'paper' type:", subgraph[('author', 'writes', 'paper')].edge_index.shape[1])
