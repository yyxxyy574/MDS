import pathlib
import yaml

ROOT = pathlib.Path(__file__).parent
for key, value in yaml.safe_load((ROOT / 'character.yaml').read_text()).items():
    globals()[key] = value

for key, value in yaml.safe_load((ROOT / 'dilemma.yaml').read_text()).items():
    globals()[key] = value