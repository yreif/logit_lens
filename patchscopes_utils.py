"""
Patchscopes utilities with batched computation and clean visualization.
"""
import torch
import numpy as np
import plotly.graph_objects as go
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional
from trajectory_analysis import decode_tokens_properly

def compute_rms_norm(x, eps=1e-8):
    return torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

class EmbeddingRescaler:
    """Efficiently computes and caches target RMS norms for rescaling."""
    
    def __init__(self, model: PreTrainedModel, enable_rescaling: bool = False, amplifying_factor: float = 1.0):
        self.model = model
        self._target_rms_norm = None
        self.enable_rescaling = enable_rescaling
        self.amplifying_factor = amplifying_factor
        
    def _compute_target_rms_norm(self) -> float:
        """Compute the average RMS norm of input embeddings."""
        if self._target_rms_norm is not None:
            return self._target_rms_norm
            
        # Get embedding matrix
        embedding_layer = self.model.get_input_embeddings()
        embedding_weights = embedding_layer.weight  # [vocab_size, hidden_dim]
        
        # Compute RMS norm for each embedding
        rms_norms = compute_rms_norm(embedding_weights)  # [vocab_size]
        
        # Take average RMS norm
        self._target_rms_norm = rms_norms.mean().item()
        return self._target_rms_norm
    
    def rescale_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Rescale hidden state to match embedding space RMS norm.
        
        Args:
            hidden_state: Hidden state tensor of shape [..., hidden_dim]
            
        Returns:
            Rescaled hidden state with same shape
        """
        rescaled = hidden_state

        if self.enable_rescaling:
            # Compute RMS norm of input hidden state
            current_rms = compute_rms_norm(hidden_state)

            # Get target RMS norm
            target_rms = self._compute_target_rms_norm()

            # Avoid division by zero
            current_rms = torch.clamp(current_rms, min=1e-8)

            # Rescale: normalize by current RMS, then scale by target RMS
            rescaled = hidden_state * (target_rms / current_rms)

        if self.amplifying_factor != 1.0:
            rescaled = rescaled * self.amplifying_factor
        
        return rescaled


def compute_patchscopes_trajectory(model: PreTrainedModel, 
                                 tokenizer: PreTrainedTokenizer,
                                 input_ids: List[int], 
                                 start_pos: int,
                                 patchscope_prompt: str = "X,X,X,X,",
                                 replace_token: str = "X",
                                 num_new_tokens: int = 5,
                                 enable_rescaling: bool = False,
                                 amplifying_factor: float = 1.0,
    ) -> List[List[str]]:
    """
    Compute patchscopes trajectory with batched computation for efficiency.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        input_ids: Input token sequence
        start_pos: Starting position to analyze
        patchscope_prompt: Prompt to use for patchscopes
        replace_token: Token to replace with hidden states
        num_new_tokens: Number of tokens to generate
        enable_rescaling: Whether to rescale hidden states to match embedding norms
        
    Returns:
        List of lists where patchscope_results[token_idx][layer_idx] = generated_text
    """
    # Determine number of layers
    try:
        n_layers = len(model.transformer.h) if hasattr(model, 'transformer') else len(model.model.layers)
    except:
        # Fallback: run a forward pass to count layers
        with torch.no_grad():
            dummy_input = torch.tensor([[1]]).to(model.device)
            outputs = model(dummy_input, output_hidden_states=True)
            n_layers = len(outputs.hidden_states) - 1  # Exclude embedding layer
    
    # Initialize rescaler if needed
    rescaler = EmbeddingRescaler(model, enable_rescaling, amplifying_factor)
    
    tokens_to_analyze = input_ids[start_pos:]
    patchscope_results = []

    # Hook model to get the last hidden state without layer norm
    def pre_norm_capture_hook(module, input):
        module.captured_input = input[0].clone()
        return input

    # Register pre-hook
    hook = model.model.norm.register_forward_pre_hook(pre_norm_capture_hook)

    with torch.no_grad():
        # Get hidden states for all layers
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        outputs = model(input_ids=model_inputs, output_hidden_states=True)

        # Cleanup
        last_hidden_state_wo_norm = model.model.norm.captured_input.clone()
        hook.remove()
        
        # Create patchscope prompt and find replacement positions
        patchscope_input_ids = tokenizer.encode(patchscope_prompt)
        patchscope_tensor = torch.tensor(patchscope_input_ids).to(model.device)
        
        # Find which token ID corresponds to the replace_token
        # We need to be more careful about tokenization
        replace_token_ids = tokenizer.encode(replace_token, add_special_tokens=False)
        if len(replace_token_ids) == 1:
            replace_token_id = replace_token_ids[0]
        else:
            # For multi-token replacements, just use the last token for now
            replace_token_id = replace_token_ids[-1]
        
        # Create mask for positions to replace - handle different variations of the replace token
        # Check for exact match first
        replace_mask = patchscope_tensor == replace_token_id
        
        # Also check for space-prefixed version (like ' X')
        space_replace_token = ' ' + replace_token
        space_replace_ids = tokenizer.encode(space_replace_token, add_special_tokens=False)
        if len(space_replace_ids) == 1:
            space_replace_id = space_replace_ids[0]
            replace_mask = replace_mask | (patchscope_tensor == space_replace_id)
        
        if not replace_mask.any():
            print(f"Warning: No replacement positions found for token '{replace_token}' in prompt '{patchscope_prompt}'")
        
        # Convert to embeddings
        base_inputs_embeds = model.get_input_embeddings()(patchscope_tensor).unsqueeze(0)  # [1, seq_len, hidden_dim]
        
        # For each token position, batch across all layers
        for token_idx, token_id in enumerate(tokens_to_analyze):
            actual_pos = start_pos + token_idx
            
            # Collect embeddings from all layers for this token position
            layer_embeddings = []
            for layer_idx in range(1, len(outputs.hidden_states)):  # Skip embedding layer
                if layer_idx < len(outputs.hidden_states) - 1:
                    layer_embedding = outputs.hidden_states[layer_idx][:, actual_pos]  # [batch=1, hidden_dim]
                else:
                    # replace last layer hidden states with unnormalized version
                    layer_embedding = last_hidden_state_wo_norm[:, actual_pos]
                layer_embedding = layer_embedding.squeeze(0)  # Remove batch dimension -> [hidden_dim]
                
                # Apply rescaling if enabled
                if rescaler is not None:
                    layer_embedding = rescaler.rescale_hidden_state(layer_embedding)
                    
                layer_embeddings.append(layer_embedding)
            
            # Stack embeddings for batched processing
            layer_embeddings = torch.stack(layer_embeddings, dim=0)  # [n_layers, hidden_dim]
            
            # Create batched inputs - one for each layer
            batch_size = len(layer_embeddings)
            batched_inputs_embeds = base_inputs_embeds.repeat(batch_size, 1, 1)  # [batch_size, seq_len, hidden_dim]
            
            # Replace ALL positions where replace_token appears with the layer embeddings
            if replace_mask.any():
                # Get positions to replace
                replace_positions = replace_mask.nonzero(as_tuple=False).flatten()
                for batch_idx in range(batch_size):
                    for pos in replace_positions:
                        batched_inputs_embeds[batch_idx, pos] = layer_embeddings[batch_idx]
            
            # Generate with patchscopes in batch
            try:
                patchscope_outputs = model.generate(
                    inputs_embeds=batched_inputs_embeds,
                    max_new_tokens=num_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    output_scores=False
                )
                
                # Decode all outputs
                token_results = []
                for i, output in enumerate(patchscope_outputs):
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                    token_results.append(generated_text.strip())

            except Exception as e:
                import pdb; pdb.set_trace()
                for i, output in enumerate(patchscope_outputs):
                    token_results.append(f"Error: {str(e)[:20]}...")
            
            patchscope_results.append(token_results)
    
    return patchscope_results


def create_patchscopes_plot(input_ids: List[int], 
                          tokenizer: PreTrainedTokenizer,
                          patchscope_results: List[List[str]], 
                          start_pos: int) -> go.Figure:
    """Create a patchscopes visualization matching the logit lens style."""
    
    tokens_to_analyze = input_ids[start_pos:]
    decoded_tokens = decode_tokens_properly(tokenizer, tokens_to_analyze)
    n_layers = len(patchscope_results[0]) if patchscope_results else 0
    
    # Follow the same structure as logit lens:
    # X-axis: Input tokens (what model sees)
    # Y-axis: Model layers (0 at bottom, increasing upward)
    # Each cell: Shows patchscope result for that layer/position
    
    # X-axis labels: Use position indices for data, clean labels for display
    x_positions = list(range(len(decoded_tokens)))
    x_display_labels = [f"'{token}'" for token in decoded_tokens]

    # Y-axis labels: Model layers (0 to n-1)  
    y_labels = [f"Layer {i}" for i in range(n_layers)]
    
    # Reorganize data: patchscope_results[token][layer] -> data[layer][token]
    # This matches the logit lens structure where data is [layers x tokens]
    layer_data = []
    cell_text = []
    hover_text = []
    
    for layer_idx in range(n_layers):
        layer_row_data = []
        layer_row_text = []
        layer_row_hover = []
        
        for token_idx in range(len(patchscope_results)):
            patchscope_result = patchscope_results[token_idx][layer_idx]
            
            # Use length as the heatmap value (like we did before)
            layer_row_data.append(len(patchscope_result))
            
            # Cell text: Truncate long results for display
            display_text = patchscope_result[:8] + "..." if len(patchscope_result) > 8 else patchscope_result
            layer_row_text.append(display_text)
            
            # Hover text: Full information
            hover_lines = []
            hover_lines.append(f"<b>Input: '{decoded_tokens[token_idx]}'</b>")
            hover_lines.append(f"<b>Layer {layer_idx}</b>") 
            hover_lines.append("")
            hover_lines.append("Patchscope result:")
            hover_lines.append(f"'{patchscope_result}'")
            
            layer_row_hover.append("<br>".join(hover_lines))
        
        layer_data.append(layer_row_data)
        cell_text.append(layer_row_text)
        hover_text.append(layer_row_hover)
    
    # Create heatmap with same structure as logit lens
    fig = go.Figure(data=go.Heatmap(
        z=layer_data,  # [layers x tokens] - values for coloring
        x=x_positions,    # Input tokens
        y=y_labels,    # Model layers
        text=cell_text,  # Text shown in cells
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
        colorscale='plasma',  # Different colorscale to distinguish from logit lens
        showscale=True
    ))
    
    fig.update_layout(
        title="Patchscopes Trajectory - Hover to see full text",
        xaxis_title="Input Tokens",
        yaxis_title="Model Layers", 
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            font_size=12,
            font_family="monospace",
            font_color="black",
            bordercolor="rgba(0, 0, 0, 0.5)"
        ),
        height=max(600, n_layers * 40),  # Same taller layout as logit lens
        margin=dict(l=80, r=40, t=80, b=100),
        xaxis=dict(
            tickmode='array',
            side='bottom',
            tickvals=x_positions,
            ticktext=x_display_labels,  # Clean display labels
            tickangle=45 if len(tokens_to_analyze) > 8 else 0,
            automargin=True
        ),
        yaxis=dict(
            tickmode='linear',
            # Layer 0 at bottom, increasing upward (same as logit lens)
            automargin=True
        )
    )
    
    return fig