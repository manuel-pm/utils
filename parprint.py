"""
Print functions for parallel execution with MPI
"""
from __future__ import print_function

from mpi4py import MPI

from utils.bcolors import print_bold, print_success, print_highlight, print_warning, print_error


def parprint(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)

def parprint_bold(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print_bold(*args, **kwargs)

def parprint_success(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print_success(*args, **kwargs)

def parprint_highlight(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print_highlight(*args, **kwargs)

def parprint_warning(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print_warning(*args, **kwargs)

def parprint_error(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print_error(*args, **kwargs)

def parprint_error_and_exit(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print_error(*args, **kwargs)
    MPI.COMM_WORLD.Barrier()
    sys.exit(1)

