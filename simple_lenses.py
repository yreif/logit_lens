"""
Simplified lens implementations without TunedLens dependency.
Only includes LogitLens and ReverseLogitLens functionality.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel


class LogitLens(nn.Module):
    """Simple logit lens that applies the unembedding matrix to hidden states."""
    
    def __init__(self, unembedding_matrix, final_norm = None):
        super().__init__()
        self.unembedding = unembedding_matrix
        self.final_norm = final_norm if final_norm is not None else lambda x: x
        
    @classmethod
    def from_model(cls, model: PreTrainedModel):
        """Create a LogitLens from a HuggingFace model."""
        # Get the unembedding matrix (lm_head or embed_out)
        if hasattr(model, 'lm_head'):
            unembedding = model.lm_head
        elif hasattr(model, 'embed_out'):
            unembedding = model.embed_out
        else:
            raise ValueError("Could not find unembedding matrix in model")

        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            final_norm = model.model.norm
        else:
            raise ValueError("Could not find final norm in model")
            
        return cls(unembedding, final_norm)
    
    def forward(self, hidden_states, apply_final_norm=False):
        """Apply lens to hidden states."""
        if apply_final_norm:
            hidden_states = self.final_norm(hidden_states)
        return self.unembedding(hidden_states)


class ReverseLogitLens(nn.Module):
    """Reverse logit lens that applies embedding matrix to hidden states."""
    
    def __init__(self, embedding_matrix):
        super().__init__()
        self.embedding = embedding_matrix
        
    @classmethod  
    def from_model(cls, model: PreTrainedModel):
        """Create a ReverseLogitLens from a HuggingFace model."""
        # Get the embedding matrix
        if hasattr(model, 'get_input_embeddings'):
            embedding = model.get_input_embeddings()
        else:
            raise ValueError("Could not find input embeddings in model")
            
        return cls(embedding)
    
    def forward(self, hidden_states):
        """Apply reverse lens to hidden states."""
        # For reverse lens, we compute similarity with embedding vectors
        # This is a simplified version
        embedding_weights = self.embedding.weight  # [vocab_size, hidden_dim]
        
        # Normalize both for cosine similarity
        hidden_norm = torch.nn.functional.normalize(hidden_states, dim=-1)
        embed_norm = torch.nn.functional.normalize(embedding_weights, dim=-1)
        
        # Compute similarities [batch, seq, vocab]
        similarities = torch.matmul(hidden_norm, embed_norm.T)
        
        # Scale to make it more like logits
        return similarities * 10.0