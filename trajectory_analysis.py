"""
Trajectory analysis functionality extracted and simplified from tuned_lens.
Includes proper token handling and visualization.
"""
import torch
import numpy as np
from transformers import PreTrainedTokenizer
from typing import List, Optional, Union, Dict, Any
import plotly.graph_objects as go
import plotly.express as px


def decode_tokens_properly(tokenizer: PreTrainedTokenizer, token_ids: List[int]) -> List[str]:
    """
    Properly decode tokens handling UTF-8 and special tokenizer formats.
    """
    decoded_tokens = []
    
    for i, token_id in enumerate(token_ids):
        try:
            # Method 1: Direct decoding (best for most cases)
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Method 2: If empty or whitespace only, try convert_ids_to_tokens
            if not token_str.strip():
                raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
                
                # Handle special tokenizer formats
                if raw_token.startswith('▁'):  # SentencePiece
                    token_str = raw_token.replace('▁', ' ')
                elif raw_token.startswith('Ġ'):  # GPT-style
                    token_str = raw_token.replace('Ġ', ' ')
                elif raw_token.startswith('##'):  # BERT-style
                    token_str = raw_token.replace('##', '')
                else:
                    token_str = raw_token
            
            # Ensure proper UTF-8 handling
            if token_str:
                try:
                    token_str = token_str.encode('utf-8').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass  # Keep original if UTF-8 handling fails
                        
            decoded_tokens.append(token_str)
            
        except Exception:
            # Ultimate fallback
            try:
                raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
                decoded_tokens.append(raw_token)
            except:
                decoded_tokens.append(f"[ID:{token_id}]")
                
    return decoded_tokens


class PredictionTrajectory:
    """
    Simplified prediction trajectory analysis.
    """
    
    def __init__(self, 
                 logits_by_layer: torch.Tensor,  # [n_layers, seq_len, vocab_size]
                 targets: List[int],
                 tokenizer: PreTrainedTokenizer,
                 input_ids: List[int]):
        """
        Initialize trajectory with logits from each layer.
        
        Args:
            logits_by_layer: Logits from each layer [n_layers, seq_len, vocab_size]
            targets: Target token IDs for each position
            tokenizer: Tokenizer for decoding
            input_ids: Input token IDs
        """
        self.logits_by_layer = logits_by_layer
        self.targets = targets
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.num_layers, self.num_tokens, self.vocab_size = logits_by_layer.shape
        
        # The x-axis shows INPUT tokens, each column shows predictions at that position
        # Early layers often echo the input, later layers predict the next token
        # So column "world" shows what the model predicts at the "world" position across layers
        self.decoded_tokens = decode_tokens_properly(tokenizer, input_ids)  # X-axis labels
        self.decoded_targets = decode_tokens_properly(tokenizer, targets)   # What we're actually predicting
    
    @classmethod
    def from_lens_and_model(cls, lens, model, input_ids: List[int], tokenizer, targets: List[int], **kwargs):
        """Create trajectory by running lens on all layers of the model."""
        
        # Convert to tensor if needed
        if isinstance(input_ids, list):
            input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        else:
            input_ids_tensor = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        
        # Get hidden states from all layers
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # [layer, batch, seq, hidden_dim]
            
            # Apply lens to each layer's hidden states
            num_layers = len(hidden_states)
            logits_by_layer = []
            for layer_i, layer_hidden in enumerate(hidden_states[1:]):  # Skip embedding layer
                layer_logits = lens(layer_hidden, layer_i < num_layers-1).squeeze(0)  # Remove batch dim
                logits_by_layer.append(layer_logits)
            
            logits_by_layer = torch.stack(logits_by_layer)  # [n_layers, seq_len, vocab_size]
        
        return cls(logits_by_layer, targets, tokenizer, input_ids if isinstance(input_ids, list) else input_ids)
    
    def slice_sequence(self, slice_obj):
        """Slice the trajectory to analyze only part of the sequence."""
        sliced_logits = self.logits_by_layer[:, slice_obj, :]
        sliced_input_ids = self.input_ids[slice_obj] if isinstance(self.input_ids, list) else self.input_ids[slice_obj].tolist()
        sliced_targets = self.targets[slice_obj] if isinstance(self.targets, list) else self.targets[slice_obj].tolist()
        
        return PredictionTrajectory(sliced_logits, sliced_targets, self.tokenizer, sliced_input_ids)
    
    def rank(self, **kwargs):
        """Compute rank statistics."""
        return RankAnalysis(self)
    
    def entropy(self, **kwargs):
        """Compute entropy statistics."""
        return EntropyAnalysis(self)
    
    def cross_entropy(self, **kwargs):
        """Compute cross entropy statistics."""  
        return CrossEntropyAnalysis(self)
    
    def forward_kl(self, **kwargs):
        """Compute forward KL divergence."""
        return ForwardKLAnalysis(self)


class TrajectoryAnalysis:
    """Base class for trajectory analysis results."""
    
    def __init__(self, trajectory: PredictionTrajectory):
        self.trajectory = trajectory
        self.compute_statistics()
        self.compute_top_predictions()
    
    def compute_statistics(self):
        """Override in subclasses to compute specific statistics."""
        pass
        
    def compute_top_predictions(self, topk: int = 5):
        """Compute top-k predictions for each layer and token position."""
        # Get top predictions for each layer and position
        self.top_predictions = []  # [layer][token][rank] = (token_text, prob%)
        self.top1_text = []       # [layer][token] = top1_token_text
        
        for layer_idx in range(self.trajectory.num_layers):
            layer_top1 = []
            layer_topk = []
            
            for token_pos in range(self.trajectory.num_tokens):
                # Get logits for this layer and position
                layer_logits = self.trajectory.logits_by_layer[layer_idx, token_pos, :]
                
                # Get top-k predictions
                probs = torch.softmax(layer_logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, topk)
                
                # Convert to text with probabilities
                topk_predictions = []
                for i, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
                    token_text = decode_tokens_properly(self.trajectory.tokenizer, [token_id.item()])[0]
                    prob_percent = prob.item() * 100
                    topk_predictions.append((token_text, prob_percent))
                
                layer_topk.append(topk_predictions)
                layer_top1.append(topk_predictions[0][0])  # Top-1 token text
            
            self.top_predictions.append(layer_topk)
            self.top1_text.append(layer_top1)
    
    def figure(self, title: str = "", topk: int = 5, **kwargs):
        """Create plotly figure with correct trajectory visualization."""
        
        # Recompute with requested topk
        self.compute_top_predictions(topk)
        
        # Prepare data for heatmap
        # X-axis: Input tokens (positions we're predicting for)
        # Y-axis: Model layers  
        # Z-values: Statistics (rank, entropy, etc.)

        # X-axis labels: Use position indices for data, clean labels for display
        x_positions = list(range(self.trajectory.num_tokens))
        x_display_labels = [f"'{token}'" for token in self.trajectory.decoded_tokens]
        
        y_labels = [f"Layer {i}" for i in range(self.trajectory.num_layers)]
        
        # Cell text: Top-1 predictions  
        cell_text = []
        for layer_idx in range(self.trajectory.num_layers):
            layer_text = []
            for token_pos in range(self.trajectory.num_tokens):
                top1_token = self.top1_text[layer_idx][token_pos]
                # Truncate long tokens for display
                display_token = top1_token[:8] + "..." if len(top1_token) > 8 else top1_token
                layer_text.append(display_token)
            cell_text.append(layer_text)
        
        # Hover text: Top-k predictions with probabilities
        hover_text = []
        for layer_idx in range(self.trajectory.num_layers):
            layer_hover = []
            for token_pos in range(self.trajectory.num_tokens):
                # Build hover text with top-k predictions
                hover_lines = []
                hover_lines.append(f"<b>Position: '{self.trajectory.decoded_tokens[token_pos]}'</b>")
                hover_lines.append(f"<b>Target: '{self.trajectory.decoded_targets[token_pos]}'</b>")
                hover_lines.append(f"<b>Layer {layer_idx}</b>")
                hover_lines.append(f"<b>{self.stat_name}: {self.data[layer_idx][token_pos]:.3f}</b>")
                hover_lines.append("")
                hover_lines.append("Layer's top predictions:")
                
                for rank, (token_text, prob_pct) in enumerate(self.top_predictions[layer_idx][token_pos]):
                    hover_lines.append(f"{rank+1}. '{token_text}' ({prob_pct:.1f}%)")
                
                layer_hover.append("<br>".join(hover_lines))
            hover_text.append(layer_hover)
        # Create custom colorscale that emphasizes low values
        # Find data range to create appropriate breakpoints
        data_array = np.array(self.data)
        min_val = data_array.min()
        max_val = data_array.max()
        
        # Create colorscale with more contrast for low values
        if max_val > 100:  # For data with large range (like 0-250k)
            # Use more colors for low values (0-100) and compress high values
            custom_colorscale = [
                [0.0, '#440154'],    # Very low - dark purple
                [0.02, '#481567'],   # Low - purple
                [0.05, '#482677'],   # Low-medium - dark blue
                [0.1, '#404387'],    # Medium-low - blue
                [0.15, '#33638D'],   # Medium - teal-blue
                [0.3, '#287D8E'],    # Medium-high - teal
                [0.5, '#29AF7F'],    # High - green
                [0.7, '#73D055'],    # Higher - light green
                [0.85, '#BFE046'],   # Very high - yellow-green
                [1.0, '#FDE725']     # Maximum - bright yellow
            ]
        else:  # For smaller ranges
            custom_colorscale = 'viridis'
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=self.data,  # Statistics values
            x=x_positions,   # Input tokens
            y=y_labels,   # Model layers
            text=cell_text,  # Top-1 predictions shown in cells
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            colorscale=custom_colorscale,
            showscale=True
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Input Tokens (predicting next token)",
            yaxis_title="Model Layers", 
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.95)",
                font_size=12,
                font_family="monospace",
                font_color="black",
                bordercolor="rgba(0, 0, 0, 0.5)"
            ),
            height=max(600, self.trajectory.num_layers * 40),  # Make taller for readability
            margin=dict(l=80, r=40, t=80, b=100),
            xaxis=dict(
                tickmode='array',
                side='bottom',
                tickvals=x_positions,
                ticktext=x_display_labels,  # Clean display labels
                tickangle=45 if self.trajectory.num_tokens > 8 else 0,
                automargin=True
            ),
            yaxis=dict(
                tickmode='linear',
                # Remove autorange='reversed' so layer 0 is at bottom (normal order)
                automargin=True
            )
        )

        return fig


class RankAnalysis(TrajectoryAnalysis):
    """Analysis of token rank in predictions."""
    
    def compute_statistics(self):
        self.stat_name = "Rank"
        # Data should be [layers x tokens] for correct heatmap orientation
        ranks = []
        
        for layer_idx in range(self.trajectory.num_layers):
            layer_ranks = []
            for token_idx in range(self.trajectory.num_tokens):
                target_id = self.trajectory.targets[token_idx]
                layer_logits = self.trajectory.logits_by_layer[layer_idx, token_idx, :]
                
                # Get rank of target token (higher logit = lower rank number)
                sorted_indices = torch.argsort(layer_logits, descending=True)
                rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1
                layer_ranks.append(rank)
            
            ranks.append(layer_ranks)
        
        self.data = ranks


class EntropyAnalysis(TrajectoryAnalysis):
    """Analysis of prediction entropy."""
    
    def compute_statistics(self):
        self.stat_name = "Entropy"
        # Data should be [layers x tokens] for correct heatmap orientation
        entropies = []
        
        for layer_idx in range(self.trajectory.num_layers):
            layer_entropies = []
            for token_idx in range(self.trajectory.num_tokens):
                layer_logits = self.trajectory.logits_by_layer[layer_idx, token_idx, :]
                probs = torch.softmax(layer_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                layer_entropies.append(entropy)
            
            entropies.append(layer_entropies)
        
        self.data = entropies


class CrossEntropyAnalysis(TrajectoryAnalysis):
    """Analysis of cross entropy with targets."""
    
    def compute_statistics(self):
        self.stat_name = "Cross Entropy"
        # Data should be [layers x tokens] for correct heatmap orientation  
        cross_entropies = []
        
        for layer_idx in range(self.trajectory.num_layers):
            layer_ce = []
            for token_idx in range(self.trajectory.num_tokens):
                target_id = self.trajectory.targets[token_idx]
                layer_logits = self.trajectory.logits_by_layer[layer_idx, token_idx, :]
                log_probs = torch.log_softmax(layer_logits, dim=-1)
                ce = -log_probs[target_id].item()
                layer_ce.append(ce)
            
            cross_entropies.append(layer_ce)
        
        self.data = cross_entropies


class ForwardKLAnalysis(TrajectoryAnalysis):
    """Analysis of forward KL divergence."""
    
    def compute_statistics(self):
        self.stat_name = "Forward KL"
        # Data should be [layers x tokens] for correct heatmap orientation
        kl_divs = []
        
        # Use final layer as reference
        final_layer_logits = self.trajectory.logits_by_layer[-1]  # [seq_len, vocab_size]
        
        for layer_idx in range(self.trajectory.num_layers):
            layer_kls = []
            for token_idx in range(self.trajectory.num_tokens):
                final_probs = torch.softmax(final_layer_logits[token_idx], dim=-1)
                layer_logits = self.trajectory.logits_by_layer[layer_idx, token_idx, :]
                layer_probs = torch.softmax(layer_logits, dim=-1)
                
                # KL(layer || final)
                kl = torch.sum(layer_probs * torch.log((layer_probs + 1e-12) / (final_probs + 1e-12))).item()
                layer_kls.append(kl)
            
            kl_divs.append(layer_kls)
        
        self.data = kl_divs