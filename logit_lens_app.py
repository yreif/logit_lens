"""
Logit lens application with patchscopes support.
No TunedLens dependency - uses our simplified implementations.
"""
import torch
import gradio as gr
import plotly.graph_objects as go
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_lenses import LogitLens, ReverseLogitLens  
from trajectory_analysis import PredictionTrajectory
from patchscopes_utils import compute_patchscopes_trajectory, create_patchscopes_plot

# Configuration
LENS_OPTIONS = ["Logit Lens", "Reverse Logit Lens"]
SAMPLING_OPTIONS = ["Greedy", "Top-p"]
STATISTIC_OPTIONS = {
    "Entropy": "entropy",
    "Cross Entropy": "cross_entropy", 
    "Rank": "rank",
    "Forward KL": "forward_kl",
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Logit Lens Analysis Tool")
    parser.add_argument("--model_name", type=str, default="CohereLabs/aya-expanse-8b",
                       help="HuggingFace model identifier")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token")
    parser.add_argument("--fp16", action="store_true", default=False,
                       help="Use FP16 precision")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use BF16 precision")
    parser.add_argument("--disable_patchscopes", action="store_true", default=False, 
                       help="Disable patchscopes computation for faster analysis")
    return parser.parse_args()


class ModelManager:
    """Manages model loading and caching."""
    
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model_name = None
        self.model = None
        self.tokenizer = None
        self.lenses = {}
        
        print(f"Using device: {self.device}")
    
    def load_model(self, model_name: str):
        """Load model and lenses if not already loaded."""
        if model_name == self.current_model_name:
            return  # Already loaded
            
        print(f"Loading model: {model_name}")
        
        # Determine dtype
        dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=self.args.hf_token,
            torch_dtype=dtype
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.args.hf_token)
        
        # Set pad_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create lenses
        self.lenses = {
            "Logit Lens": LogitLens.from_model(self.model).to(self.device).to(dtype),
            "Reverse Logit Lens": ReverseLogitLens.from_model(self.model).to(self.device).to(dtype),
        }
        
        self.current_model_name = model_name
        print(f"Model loaded successfully: {model_name}")


def generate_text(model_manager: ModelManager, 
                 prompt: str, 
                 num_tokens: int,
                 sampling_method: str, 
                 temperature: float, 
                 top_p: float) -> tuple[str, str]:
    """Generate text completion."""
    
    # Encode the prompt
    input_ids = model_manager.tokenizer.encode(prompt, return_tensors="pt").to(model_manager.device)
    
    # Generate completion
    with torch.no_grad():
        if sampling_method == "Greedy":
            generated = model_manager.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=model_manager.tokenizer.eos_token_id,
                eos_token_id=model_manager.tokenizer.eos_token_id,
            )
        else:  # Top-p sampling
            generated = model_manager.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=model_manager.tokenizer.eos_token_id,
                eos_token_id=model_manager.tokenizer.eos_token_id,
            )

    # Decode results
    full_text = model_manager.tokenizer.decode(generated[0], skip_special_tokens=True)
    generated_text = model_manager.tokenizer.decode(generated[0][len(input_ids[0]):], skip_special_tokens=True)
    
    return full_text, generated_text


def analyze_trajectory(model_manager: ModelManager,
                      prompt: str,
                      model_choice: str,
                      num_tokens: int,
                      sampling_method: str,
                      temperature: float,
                      top_p: float,
                      lens_name: str,
                      statistic: str,
                      token_cutoff: int,
                      enable_patchscopes: bool,
                      enable_rescaling: bool,
                      amplifying_factor: float,
                      patchscope_prompt: str,
                      replace_token: str,
                      patchscope_tokens: int) -> tuple[go.Figure, go.Figure, str, str]:
    """Main analysis function."""
    
    # Load model if needed
    model_manager.load_model(model_choice)
    
    # Validate inputs
    if not prompt.strip():
        empty_fig = go.Figure(layout=dict(title="Please enter a prompt."))
        return empty_fig, empty_fig, "", ""
    
    try:
        # Generate text
        full_text, generated_text = generate_text(
            model_manager, prompt, num_tokens, sampling_method, temperature, top_p
        )
        
        # Prepare input for analysis
        input_ids = model_manager.tokenizer.encode(full_text)
        
        # Check if BOS token is needed and not already present
        if (model_manager.tokenizer.bos_token_id is not None and 
            len(input_ids) > 0 and 
            input_ids[0] != model_manager.tokenizer.bos_token_id):
            input_ids = [model_manager.tokenizer.bos_token_id] + input_ids
            
        targets = input_ids[1:] + [model_manager.tokenizer.eos_token_id]
        
        if len(input_ids) <= 1:
            empty_fig = go.Figure(layout=dict(title="Generated sequence too short for analysis."))
            return empty_fig, empty_fig, full_text, generated_text
            
        if token_cutoff < 1:
            empty_fig = go.Figure(layout=dict(title="Please provide valid token cutoff."))
            return empty_fig, empty_fig, full_text, generated_text

        # Determine analysis window
        start_pos = max(len(input_ids) - token_cutoff, 0)
        
        # Create trajectory analysis
        lens = model_manager.lenses[lens_name]
        trajectory = PredictionTrajectory.from_lens_and_model(
            lens=lens,
            model=model_manager.model,
            input_ids=input_ids,
            tokenizer=model_manager.tokenizer,
            targets=targets
        )

        sliced_trajectory = trajectory.slice_sequence(slice(start_pos, len(trajectory.input_ids)))

        # Create logit lens plot
        analysis_result = getattr(sliced_trajectory, STATISTIC_OPTIONS[statistic])()
        logit_lens_plot = analysis_result.figure(
            title=f"{lens_name} - {statistic} Analysis"
        )

        # Create patchscopes plot if enabled
        if enable_patchscopes and not model_manager.args.disable_patchscopes:
            try:
                patchscope_results = compute_patchscopes_trajectory(
                    model_manager.model,
                    model_manager.tokenizer,
                    input_ids,
                    start_pos,
                    patchscope_prompt=patchscope_prompt,
                    replace_token=replace_token,
                    num_new_tokens=patchscope_tokens,
                    enable_rescaling=enable_rescaling,
                    amplifying_factor=amplifying_factor,
                )
                patchscopes_plot = create_patchscopes_plot(
                    input_ids, model_manager.tokenizer, patchscope_results, start_pos
                )
            except Exception as e:
                patchscopes_plot = go.Figure(layout=dict(title=f"Patchscopes failed: {str(e)}"))
        else:
            reason = "disabled by user" if not enable_patchscopes else "disabled by --disable_patchscopes flag"
            patchscopes_plot = go.Figure(layout=dict(title=f"Patchscopes {reason}"))
        
        return logit_lens_plot, patchscopes_plot, full_text, generated_text
        
    except Exception as e:
        error_fig = go.Figure(layout=dict(title=f"Analysis failed: {str(e)}"))
        patchscopes_fig = go.Figure(layout=dict(title="Patchscopes unavailable"))
        return error_fig, patchscopes_fig, "", ""


def create_interface(model_manager: ModelManager) -> gr.Blocks:
    """Create the Gradio interface."""
    
    preamble = f"""
# Logit Lens + Patchscopes Analysis 🔎📝

**Features:**
- **Logit Lens**: Analyze how predictions evolve across model layers
- **Patchscopes**: {'**Enabled** - See what each layer thinks each token represents' if not model_manager.args.disable_patchscopes else '**Disabled** - Use without --disable_patchscopes flag'}

**How it works:**
1. Enter a prompt → Model generates completion
2. Analyze the full sequence (prompt + completion) 
3. View results with proper token display and full hover text
"""
    
    with gr.Blocks(title="Logit Lens Analysis") as demo:
        gr.Markdown(preamble)
        
        with gr.Column():
            # Model and prompt inputs
            model_choice = gr.Textbox(
                value=model_manager.args.model_name,
                label="Model Name",
                info="HuggingFace model identifier"
            )
            
            prompt = gr.Textbox(
                value="I went to the pharmacy to",
                label="Input Prompt", 
                lines=3,
                info="Text for the model to complete"
            )
            
            # Generation parameters
            with gr.Row():
                num_tokens = gr.Slider(1, 100, value=8, step=1, label="Generation Length")
                sampling_method = gr.Dropdown(SAMPLING_OPTIONS, value="Greedy", label="Sampling Method")
            
            with gr.Row():
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            
            # Analysis parameters
            with gr.Row():
                lens_choice = gr.Dropdown(LENS_OPTIONS, value="Logit Lens", label="Lens Type")
                statistic = gr.Dropdown(list(STATISTIC_OPTIONS.keys()), value="Rank", label="Statistic")
                token_cutoff = gr.Slider(2, 50, value=20, step=1, label="Analyze Last N Tokens")

            # Patchscopes controls
            with gr.Row():
                enable_patchscopes = gr.Checkbox(value=True, label="Enable Patchscopes")
                enable_rescaling = gr.Checkbox(value=False, label="Rescale to Embedding RMS Norm")
                amplifying_factor = gr.Slider(0, 50, value=1, step=0.1, label="Amplifying Factor")
                patchscope_tokens = gr.Slider(1, 10, value=5, step=1, label="Patchscopes Generation Length")

            with gr.Row():
                patchscope_prompt = gr.Textbox(
                    value="Repeat this:X,X,X,X,X,",
                    label="Patchscopes Prompt", 
                    info="Use X as placeholder for hidden states"
                )
                replace_token = gr.Textbox(
                    value="X",
                    label="Replacement Token",
                    info="Token to replace with hidden states"
                )
            
            # Action button
            analyze_btn = gr.Button("Generate & Analyze", variant="primary", size="lg")
            
            # Outputs
            with gr.Column():
                gr.Markdown("## Generated Text")
                full_text_output = gr.Textbox(label="Full Sequence", lines=4, interactive=False)
                generated_text_output = gr.Textbox(label="Generated Completion", lines=2, interactive=False)
                
                gr.Markdown("## Analysis Results")
                logit_lens_plot = gr.Plot(label="Logit Lens Trajectory")
                patchscopes_plot = gr.Plot(label="Patchscopes Trajectory")
        
        # Event handlers
        inputs = [prompt, model_choice, num_tokens, sampling_method, temperature, top_p,
                 lens_choice, statistic, token_cutoff, enable_patchscopes, enable_rescaling, amplifying_factor,
                 patchscope_prompt, replace_token, patchscope_tokens]
        outputs = [logit_lens_plot, patchscopes_plot, full_text_output, generated_text_output]
        
        analyze_btn.click(fn=lambda *args: analyze_trajectory(model_manager, *args), 
                         inputs=inputs, outputs=outputs)
        
        demo.load(fn=lambda *args: analyze_trajectory(model_manager, *args),
                 inputs=inputs, outputs=outputs)
    
    return demo


def main():
    """Main application entry point."""
    args = parse_args()
    model_manager = ModelManager(args)
    
    # Create and launch interface
    demo = create_interface(model_manager)
    demo.launch(share=True)


if __name__ == "__main__":
    main()