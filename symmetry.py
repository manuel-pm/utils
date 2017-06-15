from __future__ import print_function

from utils.bcolors import print_error


def crystal_system(space_group_isgn):
    """
    Returns the crystal system given the international space group number.
    This can be calculated, for example, using spglib's
    get_symmetry_dataset(atoms)['number'], where atoms is an ASE Atoms object

    Notes
    -----

    Correspondence obtained from:
    https://en.wikipedia.org/wiki/List_of_space_groups

    """

    sg = int(space_group_isgn)
    if 1 <= sg <= 2:
        return 'triclinic'
    elif 3 <= sg <= 15:
        return 'monoclinic'
    elif 16 <= sg <= 74:
        return 'orthorhombic'
    elif 75 <= sg <= 142:
        return 'tetragonal'
    elif 143 <= sg <= 167:
        return 'trigonal_low'
    elif 168 <= sg <= 194:
        return 'hexagonal'
    elif 195 <= sg <= 230:
        return 'cubic'
    else:
        print_error("Space group number outside 1-270 range")
        return ''

