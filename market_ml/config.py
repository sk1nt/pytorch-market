import os
import importlib.util
from pathlib import Path

# Example config for API keys and paths
class Config:
    def __init__(self) -> None:
        # API keys (prefer environment variables)
        self.GEXBOT_API_KEY = os.getenv('GEXBOT_API_KEY', '')
        self.POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
        # Data paths
        self.OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'outputs/')
        self.DATA_DIR = os.getenv('DATA_DIR', 'data/')
        # Other config
        self.MAX_REQUESTS_PER_SEC = int(os.getenv('MAX_REQUESTS_PER_SEC', '20'))
        self.AGGREGATION = os.getenv('AGGREGATION', '1d')

        # Optionally merge overrides from market_ml/.config.py (gitignored)
        self._merge_dotconfig()

    def _merge_dotconfig(self) -> None:
        dotcfg_path = Path(__file__).with_name('.config.py')
        if not dotcfg_path.exists():
            return
        spec = importlib.util.spec_from_file_location('market_ml_dotconfig', str(dotcfg_path))
        if spec and spec.loader:  # type: ignore[truthy-function]
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            # Case 1: top-level UPPERCASE variables
            merged = False
            for name in dir(mod):
                if name.isupper():
                    setattr(self, name, getattr(mod, name))
                    merged = True
            # Case 2: module exposes 'config' object
            if hasattr(mod, 'config'):
                cfg_obj = getattr(mod, 'config')
                for name in dir(cfg_obj):
                    if name.isupper():
                        setattr(self, name, getattr(cfg_obj, name))
                        merged = True
            # Case 3: module exposes Config class
            if not merged and hasattr(mod, 'Config'):
                try:
                    cfg_cls = getattr(mod, 'Config')
                    cfg_obj = cfg_cls()
                    for name in dir(cfg_obj):
                        if name.isupper():
                            setattr(self, name, getattr(cfg_obj, name))
                            merged = True
                except Exception:
                    pass

# Module-level instance for convenience imports
config = Config()
