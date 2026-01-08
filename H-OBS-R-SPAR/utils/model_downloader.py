"""
Pretrained Model Downloader

Downloads and manages pretrained models for experiments.
"""

import torch
import torchvision.models as models
from pathlib import Path
from typing import Optional
import os


MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'efficientnet_b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth',
}


class PretrainedModelDownloader:
    """Downloads and caches pretrained models."""
    
    def __init__(self, cache_dir: str = './pretrained_models'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_model(self, model_name: str, force_download: bool = False) -> str:
        """
        Download pretrained model.
        
        Args:
            model_name: Name of model (resnet18, resnet50, etc.)
            force_download: Force re-download even if cached
        
        Returns:
            Path to downloaded model weights
        """
        if model_name not in MODEL_URLS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")
        
        model_path = self.cache_dir / f"{model_name}.pth"
        
        if model_path.exists() and not force_download:
            print(f"Using cached model: {model_path}")
            return str(model_path)
        
        print(f"Downloading {model_name}...")
        
        # Use torchvision's download mechanism
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model_name == 'mobilenetv2':
            model = models.mobilenet_v2(pretrained=True)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
        
        # Save model weights
        torch.save(model.state_dict(), model_path)
        print(f"Saved to: {model_path}")
        
        return str(model_path)
    
    def download_all(self):
        """Download all available models."""
        for model_name in MODEL_URLS.keys():
            try:
                self.download_model(model_name)
            except Exception as e:
                print(f"Failed to download {model_name}: {e}")
    
    def list_cached_models(self):
        """List all cached models."""
        cached = list(self.cache_dir.glob("*.pth"))
        print(f"\nCached models ({len(cached)}):")
        for model_path in cached:
            size_mb = model_path.stat().st_size / (1024**2)
            print(f"  - {model_path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("=== Pretrained Model Downloader ===\n")
    
    downloader = PretrainedModelDownloader()
    
    # Download specific model
    downloader.download_model('resnet18')
    
    # List cached models
    downloader.list_cached_models()
    
    print("\nDownloader test completed!")
