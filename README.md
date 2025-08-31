# Logit Lens

A tool for analyzing transformer model predictions using logit lens and patchscopes techniques.

## Features

- **Logit Lens**: Visualize how model predictions evolve across layers
- **Reverse Logit Lens**: Analyze predictions from the reverse direction
- **Patchscopes**: Advanced trajectory analysis with patching techniques
- **Interactive Web Interface**: Built with Gradio for easy experimentation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the interactive application:

```bash
python logit_lens_app.py
```

The application will launch a web interface where you can:
- Input text prompts
- Select different lens types and analysis methods
- Visualize prediction trajectories across model layers

## Files

- `logit_lens_app.py`: Main Gradio application
- `simple_lenses.py`: Core logit lens implementations
- `trajectory_analysis.py`: Prediction trajectory analysis tools
- `patchscopes_utils.py`: Patchscopes utilities and plotting functions