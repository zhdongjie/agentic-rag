from pathlib import Path

import yaml


def load_yaml(
        path: str | Path,
        encoding: str = "utf-8"
) -> dict:
    """
    Load and parse a YAML file into a Python dictionary.

    This function reads a YAML file from the given path and parses its
    contents using `yaml.safe_load`, which ensures that only standard
    YAML constructs are processed (preventing execution of arbitrary code).

    Args:
        path (str | Path): Path to the YAML file.
        encoding (str): File encoding used when reading the file.
            Defaults to "utf-8".

    Returns:
        dict: Parsed YAML content as a Python dictionary.
            - Returns None if the YAML file is empty.
    """
    with open(Path(path), "r", encoding=encoding) as f:
        return yaml.safe_load(f)
