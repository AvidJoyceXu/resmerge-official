import re
import pathlib
import torch
import rich
import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.console import Console

def start_epoch_index_from_checkpoint_path(ckpt_path) -> Optional[int]:
    """Parse epoch index from Top-K / default ckpt names (e.g. epoch=0200-train_loss=...). Returns None for e.g. latest.ckpt."""
    name = pathlib.Path(ckpt_path).name
    m = re.search(r'epoch[=_](\d+)', name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def log_action_mse(step_log, category, pred_action, gt_action):
    B, T, _ = pred_action.shape
    pred_action = pred_action.view(B, T, -1)
    gt_action = gt_action.view(B, T, -1)
    error = torch.nn.functional.mse_loss(pred_action, gt_action)
    step_log[f'{category}_action_mse_error'] = error
    print(f"{category} action mse error: {error}")

def _print_config(config) -> None:
    """Print the configuration as a nicely formatted table."""
    from rich.table import Table

    table = Table(title="⚙️ Training Configuration", show_header=True, header_style="bold green")
    table.add_column("Parameter", style="bold white")
    table.add_column("Value", style="bold cyan")

    def flatten_config(cfg, prefix: str = "") -> list[tuple[str, str]]:
        rows = []
        for field, value in cfg.items():
            full_field = f"{prefix}.{field}" if prefix else field
            if isinstance(value, (list, tuple, set)):
                # Format list/tuple/set values
                rows.append((full_field, ", ".join(str(item) for item in value)))
            else:
                # Add simple values
                rows.append((full_field, str(value)))
        return rows

    for param, value in flatten_config(config):
        table.add_row(param, value)

    rich.print(table)

class CustomHandler(logging.Handler):
    def __init__(self, console):
        super().__init__()
        self.console = console

    def emit(self, record):
        
        custom_level = "PsiPolicy: "
        
        message = record.getMessage()
        
        self.console.print(f"[bold blue]{custom_level}[/] {message}")

def create_logger():
    """
    Create a logger that writes to a log file and stdout.
    """
    console = Console()
    handler = CustomHandler(console=console)

    logging.basicConfig(
        level="INFO",
        format="%(message)s",  
        handlers=[
            handler
        ],
    )
    logger = logging.getLogger(__name__)
    return logger