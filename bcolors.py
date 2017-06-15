from __future__ import print_function

import sys


"""
Convenience class to provide ANSI scape characters for different situations
If more advance formatting control is required, use packages termcolor or
blessings
"""
class bcolors:
    # contexts
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''
        self.BOLD = ''
        self.UNDERLINE = ''

    def enable(self):
        self.HEADER = '\033[95m'
        self.OKBLUE = '\033[94m'
        self.OKGREEN = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'


"""
Convenience functions to provide specific formatted printing
"""
def print_color(color, message, **kwargs):
    color_code = ''
    if color in ['red', 'r']:
        color_code = bcolors.FAIL
    elif color in ['green', 'g']:
        color_code = bcolors.OKGREEN
    elif color in ['blue', 'b']:
        color_code = bcolors.OKBLUE
    elif color in ['yellow', 'y']:
        color_code = bcolors.WARNING
    print(color_code + message + bcolors.ENDC, **kwargs)

def print_bold(bold_message):
    print(bcolors.BOLD + bold_message + bcolors.ENDC)

def print_success(success_message):
    print(bcolors.OKGREEN + success_message + bcolors.ENDC)

def print_highlight(highlighted_message):
    print(bcolors.BOLD + bcolors.FAIL + highlighted_message + bcolors.ENDC)

def print_warning(warning_message):
    print(bcolors.WARNING + "WARNING: " + warning_message + bcolors.ENDC)

def print_error(error_message):
    print(bcolors.FAIL + "ERROR: " + error_message + bcolors.ENDC)

def print_error_and_exit(error_message):
    print(bcolors.FAIL + "ERROR: " + error_message + bcolors.ENDC)
    sys.exit(1)

"""
Version for parallel execution with MPI
"""
from mpi4py import MPI


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
        print_error_and_exit(*args, **kwargs)
    # MPI.COMM_WORLD.Barrier()
    # sys.exit(1)

