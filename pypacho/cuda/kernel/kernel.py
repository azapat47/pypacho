import os

def get_path():
    return os.path.join(os.path.dirname(__file__), 'kernel.cu')

def get_path_d():
    return os.path.join(os.path.dirname(__file__), 'kernel_d.cu')
