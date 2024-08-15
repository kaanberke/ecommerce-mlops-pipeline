import math

import numpy as np


def convert_numpy_types(obj):
    """
    Converts numpy types to native python types
    Args:
        obj: object to convert
    Returns:
        object with numpy types converted to native python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) and not np.isinf(obj) else None
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, float):
        return obj if not math.isnan(obj) and not math.isinf(obj) else None
    return obj


def remove_none_values(obj):
    """
    Removes None values from a dictionary or list
    Args:
        obj: object to remove None values from
    Returns:
        object with None values removed
    """
    if isinstance(obj, dict):
        return {k: remove_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_none_values(v) for v in obj if v is not None]
    return obj
