import os
import yaml
from src.datascience import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a yaml file and return a ConfigBox object

    Args:
    path_to_yaml: Path to the yaml file

    Returns:
    ConfigBox object

    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Reading yaml file from {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Yaml file is empty: {path_to_yaml}")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create directories if they don't exist

    Args:
    path_to_directories: List of paths to directories
    verbose: If True, print message

    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a json file and return a ConfigBox object

    Args:
    path_to_json: Path to the json file

    Returns:
    ConfigBox object

    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"Json file loaded from {path}")
    return ConfigBox(content)

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save a dictionary to a json file

    Args:
    path: Path to the json file
    data: Dictionary to save

    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Json file saved to {path}")



@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save a binary file

    Args:
    data: Data to save
    path: Path to save the binary file

    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved to {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a binary file

    Args:
    path: Path to the binary file

    Returns:
    Data loaded from the binary file

    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from {path}")
    return data

