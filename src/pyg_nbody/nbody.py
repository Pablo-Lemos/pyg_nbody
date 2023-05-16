'''
N-body simulation using pytorch geometric.
'''

import torch
from torch_geometric.data import Data

from .utils import get_deltas, dummy_node_model, dummy_edge_model, force_newton, sum_forces, MessagePassingLayer


class NBodySimulator:
    def __init__(self, masses, x0, v0, force=force_newton, device='cpu'):
        # Initial position and velocity
        self.nParticles = x0.shape[0]
        # Defensive programming
        assert(device in ['cpu', 'cuda']), "device must be 'cpu' or 'cuda'"
        assert(x0.shape == (self.nParticles, 3)), "x0 must be a (nParticles, 3) array"
        assert(v0.shape == (self.nParticles, 3)), "v0 must be a (nParticles, 3) array"
        assert(masses.shape == (self.nParticles, 1)), "masses must be a (nParticles, 1) array"
        assert(callable(force)), "force must be a function"
        
        self._device = device
        
        x0 = torch.as_tensor(x0, dtype=torch.float64, device=self._device)
        v0 = torch.as_tensor(v0, dtype=torch.float64, device=self._device)
        mass = torch.as_tensor(masses, dtype=torch.float64, device=self._device)
        
        nodes = torch.cat([mass, x0, v0], dim=-1)
        edge_index = torch.tril_indices(self.nParticles, self.nParticles, offset=-1)
        edge_attr = get_deltas(x0)
        
        self.graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)
        
        # Define the layers
        # The force layer will compute the force between all the nodes
        self.force_layer = MessagePassingLayer(force, dummy_node_model).to(self._device)
        # The update layer will update the nodes from the forces
        self.update_layer = MessagePassingLayer(dummy_edge_model, sum_forces).to(self._device)

    def _get_forces(self):
        out = self.force_layer(self.graph.x, self.graph.edge_index, self.graph.edge_attr)
        self.graph = Data(x=out[0], edge_index=self.graph.edge_index, edge_attr=out[1])
    
    def _update_nodes(self, dt):
        """ This function can simply aggregate the edges of the graph into the nodes"""
        sumForces, _, _ = self.update_layer(self.graph.x, self.graph.edge_index, self.graph.edge_attr)
        a = sumForces / self.graph.x[:, :1]
        dv = a * dt
        nodes = self.graph.x.clone()
        nodes[:, 4:] += dv
        nodes[:, 1:4] += nodes[:, 4:] * dt
        self.graph = Data(x=nodes, edge_index=self.graph.edge_index, edge_attr=self.graph.edge_attr)
    
    def _get_distances(self):
        """ This function can simply aggregate the edges of the graph into the nodes"""
        x = self.graph.x[:, 1:4]
        edge_attr = get_deltas(x)
        self.graph = Data(x=self.graph.x, edge_index=self.graph.edge_index, edge_attr=edge_attr)
    
    def _step(self, dt):
        self._get_forces()
        self._update_nodes(dt)
        self._get_distances()    
        
    def run(self, dt, nSteps):
        X = torch.zeros((nSteps + 1, self.nParticles, 3), dtype=torch.float64, device=self._device)
        V = torch.zeros((nSteps + 1, self.nParticles, 3), dtype=torch.float64, device=self._device)
        X[0] = self.graph.x[:, 1:4]
        V[0] = self.graph.x[:, 4:]
        for t in range(nSteps):
            self._step(dt)
            X[t + 1] = self.graph.x[:, 1:4]
            V[t + 1] = self.graph.x[:, 4:]
        
        return X, V
            
        
if __name__ == "__main__":
    x0 = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    v0 = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float64)
    mass = torch.tensor([[1], [1], [1], [1]], dtype=torch.float64)

    s = NBodySimulator(mass, x0, v0, device='cpu')
    
    X, V = s.run(0.01, 10)