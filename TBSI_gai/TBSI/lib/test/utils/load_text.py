import numpy as np
import pandas as pd


def load_text_numpy(path, delimiter, dtype):
    # Use manual file I/O to bypass numpy._datasource bug
    # that causes FileNotFoundError in multiprocessing on existing files
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                with open(path, 'r') as f:
                    ground_truth_rect = np.loadtxt(f, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        with open(path, 'r') as f:
            ground_truth_rect = np.loadtxt(f, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str
