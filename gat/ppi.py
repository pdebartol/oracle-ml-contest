from itertools import product
import os.path as osp
import json

import torch
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops

'''
HERE WE BUILD A DATASET THAT WILL HELP US IMPLEMENT GAT.

WE KNOW THAT OUR GRAPH IS COMPOSED OF 24 SUBGRAPH.

WE BUILD A TRAINING SET OF 18 SUBGRAPHS, A VALIDATION OF 2 SUBGRAPHS 

WHILE  OUR TEST SET IS MADE OF 4 SUBGRAPHS.

'''



class PPI(InMemoryDataset):

    def __init__(self,
                 root,
                 split='train'):

        assert split in ['train', 'valid', 'test']

        super(PPI, self).__init__(root)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[2])


    @property
    def raw_file_names(self):
        split = ['train', 'test1', 'test2']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        file_names = ['{}_{}'.format(s, f) for s, f in product(split, files)]
        file_names.remove('test1_labels.npy')
        file_names.remove('test2_labels.npy')
        return file_names


    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def download(self):
            return 0

    def process(self):
        for s, split in enumerate(['train']):
            path = osp.join(self.raw_dir, '{}_graph.json').format(split)
            with open(path, 'r') as f:
                G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

            x = np.load(osp.join(self.raw_dir, '{}_feats.npy').format(split))
            x = torch.from_numpy(x).to(torch.float)
            y = np.load(osp.join(self.raw_dir, '{}_labels.npy').format(split))
            y = torch.from_numpy(y).to(torch.float)

            data_list = []
            path = osp.join(self.raw_dir, '{}_graph_id.npy').format(split)
            idx = torch.from_numpy(np.load(path)).to(torch.long)
            idx = idx - idx.min()

            for i in range(idx.max().item() + 1):
                mask = idx == i
                G_s = G.subgraph(mask.nonzero().view(-1).tolist())
                edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
                edge_index = edge_index - edge_index.min()
                edge_index, _ = remove_self_loops(edge_index)
                data = Data(edge_index=edge_index, x=x[mask], y=y[mask])
                data_list.append(data)
            torch.save(self.collate(data_list[:-2]), self.processed_paths[0])
            torch.save(self.collate(data_list[-2:]), self.processed_paths[1])

            data_list = []
            for s, split in enumerate(['test1','test2']):
                path = osp.join(self.raw_dir, '{}_graph.json').format(split)
                with open(path, 'r') as f:
                    G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

                x = np.load(osp.join(self.raw_dir, '{}_feats.npy').format(split))
                x = torch.from_numpy(x).to(torch.float)

                path = osp.join(self.raw_dir, '{}_graph_id.npy').format(split)
                idx = torch.from_numpy(np.load(path)).to(torch.long)
                idx = idx - idx.min()

                for i in range(idx.max().item() + 1):
                    mask = idx == i
                    G_s = G.subgraph(mask.nonzero().view(-1).tolist())
                    edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
                    edge_index = edge_index - edge_index.min()
                    edge_index, _ = remove_self_loops(edge_index)
                    data = Data(edge_index=edge_index, x=x[mask])
                    data_list.append(data)
            torch.save(self.collate(data_list), self.processed_paths[2])