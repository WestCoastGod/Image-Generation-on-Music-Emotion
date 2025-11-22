import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average for model parameters.
    Maintains a smoothed version of model weights for better generation quality.
    """
    def __init__(self, model, decay=0.999):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (higher = slower updates, smoother weights)
        """
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA parameters with current model parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        """Replace model parameters with EMA parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        """Restore original model parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.original
                param.data = self.original[name]
        self.original = {}
    
    def state_dict(self):
        """Get state dict for saving"""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
