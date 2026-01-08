import torch
import torch.nn as nn
import torch_pruning as tp
from typing import Dict, List, Union, Optional

class PhysicalPruner:
    """
    Handles physical structural pruning using torch-pruning library.
    Ensures dependency consistency (e.g., for ResNet skip connections).
    """
    def __init__(self, model: nn.Module, example_input: torch.Tensor = None):
        """
        Args:
            model: PyTorch model to prune
            example_input: Dummy input for dependency graph construction
                           (default: random tensor [1, 3, 224, 224])
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        if example_input is None:
            example_input = torch.randn(1, 3, 224, 224).to(self.device)
        else:
            example_input = example_input.to(self.device)
            
        # Build dependency graph
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(model, example_inputs=example_input)
        
    def prune(self, pruning_plan: Dict[str, List[int]]) -> nn.Module:
        """
        Execute physical pruning based on the provided plan.
        
        Args:
            pruning_plan: Dictionary mapping layer names to list of filter indices to prune.
                          e.g., {'layer1.0.conv1': [0, 2, 5], ...}
                          
        Returns:
            Physically pruned model (in-place modification)
        """
        print(f"\n=== Executing Physical Pruning ===")
        print(f"Plan contains {len(pruning_plan)} layers to prune.")
        
        # Map layer names to modules
        modules = dict(self.model.named_modules())
        
        pruned_layers = 0
        total_idxs = 0
        
        for name, idxs in pruning_plan.items():
            if name not in modules:
                print(f"Warning: Layer {name} not found in model, skipping.")
                continue
                
            module = modules[name]
            
            # Ensure indices are unique and sorted
            idxs = sorted(list(set(idxs)))
            if not idxs:
                continue
                
            # Get pruning plan from DependencyGraph
            # We use the appropriate pruning function based on layer type
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Get dependency-aware pruning plan
                # prune_out_channels is standard for structured pruning of filters
                try:
                    pruning_group = self.DG.get_pruning_group(
                        module, 
                        tp.prune_conv_out_channels if isinstance(module, nn.Conv2d) else tp.prune_linear_out_channels,
                        idxs=idxs
                    )
                    
                    # Execute pruning
                    if self.DG.check_pruning_group(pruning_group, idxs):
                        pruning_group.exec()
                        pruned_layers += 1
                        total_idxs += len(idxs)
                    else:
                        print(f"Warning: Pruning group check failed for {name}")
                        
                except Exception as e:
                    print(f"Error pruning {name}: {str(e)}")
                    
        print(f"Physical pruning complete. Pruned {total_idxs} filters across {pruned_layers} layers.")
        print(f"Model is now physically smaller.")
        
        return self.model
