import pickle
from pathlib import Path

def check_pkl(path):
    path = Path(path)
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception:
        # 尝试兼容性加载
        import numpy as np
        class RenamingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("numpy._core"):
                    module = module.replace("numpy._core", "numpy.core", 1)
                return super().find_class(module, name)
        with path.open("rb") as f:
            data = RenamingUnpickler(f).load()

    print(f"Type: {type(data)}")
    if isinstance(data, list):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print(f"First item: {data[0]}")
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k in data.keys():
            if hasattr(data[k], 'shape'):
                print(f"  {k} shape: {data[k].shape}")
            elif isinstance(data[k], list):
                print(f"  {k} length: {len(data[k])}")

if __name__ == "__main__":
    check_pkl("/workspace/liushuo/cupid/square_d1_core/score.pkl")
