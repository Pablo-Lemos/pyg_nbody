from typing import ClassVar, Literal, Tuple, cast

import torch
from einops import rearrange
from gymnasium.spaces import MultiDiscrete
from torchtyping import TensorType

from pyg_nbody import NBodySimulator

from gfn.containers.states import States
from gfn.envs.env import Env
from gfn.envs.preprocessors import (
    IdentityPreprocessor,
    KHotPreprocessor,
    OneHotPreprocessor,
)

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]


preprocessors_dict = {
    "KHot": KHotPreprocessor,
    "OneHot": OneHotPreprocessor,
    "Identity": IdentityPreprocessor,
}


class NbodyEnv(Env):
    def __init__(
        self,
        x0,
        xInv,
        masses,
        massesInv,
        ndim = 2, 
        max_particles = 5,
        height = 10,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["KHot", "OneHot", "Identity"] = "KHot"
        ):
        
            assert ndim in [2, 3], "ndim must be 2 or 3"
            assert isinstance(x0, torch.Tensor), "x0 must be a torch tensor"
            assert isinstance(xInv, torch.Tensor), "xInv must be a torch tensor"
            assert isinstance(masses, torch.Tensor), "masses must be a torch tensor"
            assert isinstance(massesInv, torch.Tensor), "massesInv must be a torch tensor"
            assert x0.shape[0] == masses.shape[0], "x0 and masses must have the same number of particles"
            assert xInv.shape[0] == massesInv.shape[0], "xInv and massesInv must have the same number of particles"
            assert x0.shape[1] == ndim, "x0 must have ndim columns"
            assert xInv.shape[1] == ndim, "xInv must have ndim columns"
            assert masses.shape[1] == 1, "masses must have 1 column"
            assert massesInv.shape[1] == 1, "massesInv must have 1 column"
            assert len(x0.shape) == 2, "x0 must be a 2D tensor"
            assert len(xInv.shape) == 2, "xInv must be a 2D tensor"
            assert len(masses.shape) == 2, "masses must be a 2D tensor"
            assert len(massesInv.shape) == 2, "massesInv must be a 2D tensor"
            
            self.x0 = x0            
            self.xInv = xInv
            if ndim == 2:
                self.x0 = torch.cat([self.x0, torch.zeros_like(self.x0[:, 0:1])], dim=1)
                self.xInv = torch.cat([self.xInv, torch.zeros_like(self.xInv[:, 0:1])], dim=1)
                
            self.v0 = torch.zeros_like(self.x0)
            self.vInv = torch.zeros_like(self.xInv)
            self.masses = masses
            self.massesInv = massesInv
            
            self._simulate_true()
            
            self.ndim = ndim
            self.height = height
            self.max_particles = max_particles
            
            s0 = torch.zeros(ndim * max_particles, dtype=torch.long, device=torch.device(device_str))
            sf = torch.full(
                (ndim * max_particles,), fill_value=-1, dtype=torch.long, device=torch.device(device_str)
            )
            
            action_space = MultiDiscrete([height] * ndim + [2])
        
            if preprocessor_name == "Identity":
                preprocessor = IdentityPreprocessor(output_shape=(ndim,))
            elif preprocessor_name == "KHot":
                preprocessor = KHotPreprocessor(
                    height=height, ndim=ndim, get_states_indices=self.get_states_indices
                )
            elif preprocessor_name == "OneHot":
                preprocessor = OneHotPreprocessor(
                    n_states=self.n_states,
                    get_states_indices=self.get_states_indices,
                )
            else:
                raise ValueError(f"Unknown preprocessor {preprocessor_name}")

            super().__init__(
                action_space=action_space,
                s0=s0,
                sf=sf,
                device_str=device_str,
                preprocessor=preprocessor,
            )    

    def make_States_class(self) -> type[States]:
        "Creates a States class for this environment"
        env = self

        class HyperGridStates(States):

            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf
            
    def _simulate_true(self):
        x0 = torch.cat([self.x0, self.xInv], dim=0)
        v0 = torch.cat([self.v0, self.vInv], dim=0)
        masses = torch.cat([self.masses, self.massesInv], dim=0)
        
        sim = NBodySimulator(masses, x0, v0, device='cpu')
        
        # Simulate
        self.x_true, self.v_true = sim.run(0.1, 100)
            
    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        return actions[-1] == 1
        
    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        first_index = torch.where(states == -1)[0][0]
        states[first_index:first_index + self.ndim] = actions
        
    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        first_index = torch.where(states == -1)[0][0]
        states[first_index - self.ndim:first_index] = -1
        
    def log_reward(self, final_states: States) -> TensorFloat:
        final_states_raw = final_states.states_tensor
        log_rewards = torch.zeros(final_states_raw.shape[0])
        
        for i in range(final_states_raw.shape[0]):
            first_index = torch.where(final_states_raw[i] == -1)[0][0]
            
            if self.ndim == 2:
                xInv = torch.cat([final_states_raw[i, :first_index].view(-1, 2), torch.zeros(1, 1)], dim=0)
            else:
                xInv = final_states_raw[i, :first_index]
            
            #xInv = torch.cat([final_states_raw[i], torch.zeros(1)], dim=0)
            vInv = torch.zeros_like(xInv)
            massesInv = massesInv = torch.tensor([50], dtype=torch.float64)

            x0 = torch.cat([self.x0, xInv.unsqueeze(0)], dim=0)
            v0 = torch.cat([self.v0, vInv.unsqueeze(0)], dim=0)
            masses = torch.cat([self.masses, massesInv.unsqueeze(0)], dim=0)
            
            sim = NBodySimulator(masses, x0, v0, device='cpu')
            
            # Simulate
            x, v = sim.run(0.1, 100)
            
            log_rewards[i] = - 0.5 * (x[-1] - self.x_true[-1]).pow(2).mean(-1).mean()
        
        return log_rewards
        