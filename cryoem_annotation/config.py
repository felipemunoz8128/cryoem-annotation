"""Configuration management."""

from pathlib import Path
from typing import Optional, Dict, Any
import os

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class Config:
    """Configuration manager."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML config file (optional)
        """
        self.config = self._load_defaults()
        
        if config_file and config_file.exists():
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'micrograph_folder': None,
            'sam_model': {
                'type': 'vit_b',
                'checkpoint_path': 'sam_vit_b_01ec64.pth',
            },
            'output': {
                'folder': 'annotation_results',
                'create_overview': True,
                'save_masks': True,
            },
            'image': {
                'extensions': ['.mrc', '.tif', '.tiff', '.png', '.jpg', '.jpeg'],
                'normalization': {
                    'method': 'percentile',
                    'percentile_range': [1, 99],
                },
            },
            'labeling': {
                'categories': [
                    {'name': 'mature', 'key': '1'},
                    {'name': 'immature', 'key': '2'},
                    {'name': 'indeterminate', 'key': '3'},
                    {'name': 'other', 'key': '4'},
                    {'name': 'empty', 'key': '5'},
                ],
            },
        }
    
    def load_from_file(self, config_file: Path) -> None:
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            print(f"Warning: PyYAML not installed. Cannot load config file {config_file}")
            print("Install with: pip install pyyaml")
            return
        
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(self.config, file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _merge_config(self, base: Dict, override: Dict) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'CRYOEM_MICROGRAPH_FOLDER': ('micrograph_folder',),
            'CRYOEM_SAM_TYPE': ('sam_model', 'type'),
            'CRYOEM_SAM_CHECKPOINT': ('sam_model', 'checkpoint_path'),
            'CRYOEM_OUTPUT_FOLDER': ('output', 'folder'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested(self.config, config_path, value)
    
    def _set_nested(self, config: Dict, path: tuple, value: Any) -> None:
        """Set nested configuration value."""
        for key in path[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[path[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


def load_config(config_file: Optional[Path] = None) -> Config:
    """
    Load configuration.
    
    Args:
        config_file: Path to config file (optional)
    
    Returns:
        Config instance
    """
    return Config(config_file)

