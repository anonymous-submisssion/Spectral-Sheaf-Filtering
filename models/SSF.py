import torch
import pickle
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch_geometric.data import Data
from argparse import Namespace

from models.FFT_Hypergraph import FFT_Hypergraph
from models.TS_Hypergraph import TS_Hypergraph
from models.Wavelet_Hypergraph import Wavelet_Hypergraph
from sheaf_models.sheaf_models import *



def transform_adjacency_matrix():

    sensor_ids, sensor_id_to_ind, adj_matrix = load_graph_data("./Datasets/sensor_graph/adj_mx.pkl")

    adj_array = np.array(adj_matrix)
    binary_adj = (adj_array != 0).astype(int)
    # Create edge_index
    # Find non-zero elements (edges)
    rows, cols = np.where(binary_adj == 1)
    
    # Create edge_index tensor
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    return edge_index

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.device = configs.device

        # self.feat = configs.new_len  # Length    PS: batch_size x seq_len should be > features

        self.feat = self.configs.num_sensors * self.configs.data_dim * self.configs.horizon
        self.num_class = 12
        self.top_k = 3
        self.length = 8
        self.fft_hypergraph = FFT_Hypergraph(self.top_k)
        self.ts_hypergraph = TS_Hypergraph(window_size=self.length)
        self.wavelet_hypergraph = Wavelet_Hypergraph(band_size=self.length)

        args_dict = {
                'horizon': self.configs.horizon,
                'num_features': self.feat,     # number of node features
                'num_classes': self.num_class,       # number of classes
                'All_num_layers': 8,    # number of layers
                'dropout': 0.3,         # dropout rate
                'MLP_hidden': 256,      # dimension of hidden state (for most of the layers)
                'AllSet_input_norm': True,  # normalising the input at each layer
                'residual_HCHA': True, # using or not a residual connectoon per sheaf layer

                'heads': 6,             # dimension of reduction map (d)
                'init_hedge': 'avg',    # how to compute hedge features when needed. options: 'avg'or 'rand'
                'sheaf_normtype': 'sym_degree_norm',  # the type of normalisation for the sheaf Laplacian. options: 'degree_norm', 'block_norm', 'sym_degree_norm', 'sym_block_norm'
                'sheaf_act': 'tanh',    # non-linear activation used on tpop of the d x d restriction maps. options: 'sigmoid', 'tanh', 'none'
                'sheaf_left_proj': False,   # multiply to the left with IxW or not
                'dynamic_sheaf': False, # infering a differetn sheaf at each layer or use ta shared one

                'sheaf_pred_block': 'cp_decomp', # indicated the type of model used to predict the restriction maps. options: 'MLP_var1', 'MLP_var3' or 'cp_decomp'
                'sheaf_dropout': True, # use dropout in the sheaf layer or not
                'sheaf_special_head': False,    # if True, add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
                'rank': 2,              # only for LowRank type of sheaf. mention the rank of the reduction matrix

                'HyperGCN_mediators': True, #only for the Non-Linear sheaf laplacian. Indicates if mediators are used when computing the non-linear Laplacian (same as in HyperGCN)
                'cuda': 0

            }

        self.sheaf_args = Namespace(**args_dict)
        # self.sheaf_model = SheafHyperGNN(self.sheaf_args, sheaf_type='SheafHyperGNNDiag').to(self.device)
        num_nodes = self.configs.batch_size*self.length
        self.sheaf_model = SheafHyperGCN(V=num_nodes,
                         num_features=self.sheaf_args.num_features,
                         num_layers=self.sheaf_args.All_num_layers,
                         num_classses=self.sheaf_args.num_classes,
                         args=self.sheaf_args, sheaf_type= 'DiagSheafs'
                         ).to(self.device)
        self.sheaf_projection = nn.Linear(self.length, 1)

    def forecast(self, x):
        # print("Data size: ", x.size())
        features = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

        edge_index = transform_adjacency_matrix()
        #print("Edge index size: ", edge_index.size())
        # Sheaf Hypergraph neural network
        hypergraph_data = Data(x = features, edge_index = edge_index).to(self.device)
        
        output = self.sheaf_model(hypergraph_data)
        #print("model output size: ", output.size())

        return output 

    def classification(self, x):
        
        print("Data size: ", x.size())
        features = x.reshape(x.shape[0]*x.shape[1], x.shape[2]*x.shape[3])

        edge_index = transform_adjacency_matrix()
        print("Edge index size: ", edge_index.size())
        # Sheaf Hypergraph neural network
        hypergraph_data = Data(x = features, edge_index = edge_index).to(self.device)
        
        output = self.sheaf_model(hypergraph_data)
        print("model output size: ", output.size())
        output = output.mean(dim=1)

        return output

    def forward(self, x):

        if self.task_name == 'forecast':
            out = self.forecast(x)
            return out 

        if self.task_name == 'classification':
            out = self.classification(x)
            return out 
        return None