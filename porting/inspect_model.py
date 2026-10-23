"""Print the module tree and state_dict of a timm model, the starting point of a port.

    uv run inspect_model.py <timm architecture> [key=value ...]   e.g. resnet18 layers=[1,1,1,1]
"""
import json
import sys

from common import build


def main():
    arch, *pairs = sys.argv[1:]
    kwargs = {k: json.loads(v) for k, v in (p.split("=", 1) for p in pairs)}
    model = build({"architecture": arch, "kwargs": kwargs})
    print(model)
    print("\nstate_dict:")
    for k, v in model.state_dict().items():
        print(f"  {k:60s} {tuple(v.shape)} {str(v.dtype).removeprefix('torch.')}")
    print(f"\n{sum(p.numel() for p in model.parameters())} parameters")


if __name__ == "__main__":
    main()
