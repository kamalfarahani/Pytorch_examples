import torch
import torch.nn as nn

from pathlib import Path
from colored import Fore, Back, Style


def save_model(
    model: nn.Module,
    path: str,
    model_name: str
) -> None:
    """
    Saves the model to the given path with the given name.
    Args:
        model (nn.Module): The neural network model.
        path (str): The path to save the model.
        model_name (str): The name of the model.
    """
    target_dir = Path(path)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = target_dir / model_name
    
    # Color for showing saving message
    color: str = f'{Style.BOLD}{Back.GREEN}'
    print(f'{color}Saving model...{Style.reset}')
    
    torch.save(model.state_dict(), model_path)