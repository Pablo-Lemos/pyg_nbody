import torch
from torch_geometric.nn import MessagePassing, MetaLayer


def get_deltas(x):
    diff = x[:, None, :] - x[None, :, :]
    tril_indices = torch.tril_indices(diff.shape[0], diff.shape[1], offset=-1)
    return diff[tril_indices[0], tril_indices[1]]


def dummy_node_model(x, edge_index, edge_attr, u, batch):
    return x


def dummy_edge_model(x, edge_index, edge_attr, u, batch):
    return edge_attr


def force_newton(x_i, x_j, edge_attr, u, batch):
    return - x_i[:, :1] * x_j[:, :1] * edge_attr / torch.linalg.norm(
        edge_attr, axis=-1, keepdims=True)
    
    
def sum_forces(x, edge_index, edge_attr, u, batch):    
    senders = edge_index[0]
    receivers = edge_index[1]
    
    sender_mask = torch.zeros((x.shape[0], edge_attr.shape[0]), dtype=torch.float64)
    receiver_mask = torch.zeros((x.shape[0], edge_attr.shape[0]), dtype=torch.float64)
    sender_mask[senders, torch.arange(edge_attr.shape[0])] = 1.
    receiver_mask[receivers, torch.arange(edge_attr.shape[0])] = 1.
    
    return torch.matmul(sender_mask, edge_attr) - torch.matmul(receiver_mask, edge_attr)

    
class MessagePassingLayer(MessagePassing):
    def __init__(self, edge_model, node_model):
        super(MessagePassingLayer, self).__init__(aggr='add')
        self.edge_model = edge_model
        self.node_model = node_model
        self.meta = MetaLayer(self.edge_model, self.node_model)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        out = self.meta(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)        
        return out

    def message(self, x_j, x_i, edge_attr):
        return self.edge_model(x_i, x_j, edge_attr)