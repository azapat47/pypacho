import os

def get_path():
    return os.path.join(os.path.dirname(__file__), 'kernel.cl')

def get_dir():
    return os.path.dirname(__file__)