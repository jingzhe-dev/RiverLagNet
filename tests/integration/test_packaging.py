from importlib import import_module
from importlib.resources import files
from pathlib import Path


def test_packaged_hydra_configs_exist_and_match_root_configs() -> None:
    assert import_module("RiverLagNet.configs") is not None
    root_configs = Path(__file__).resolve().parents[2] / "configs"
    package_configs = files("RiverLagNet").joinpath("configs")
    for source in root_configs.rglob("*.yaml"):
        relative = source.relative_to(root_configs).as_posix()
        packaged = package_configs.joinpath(relative)
        assert packaged.is_file(), f"missing packaged Hydra config: {relative}"
        assert packaged.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
