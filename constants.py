from __future__ import print_function

try:
    from scipy import constants as cts
    scipy_available = True
except ImportError:
    scipy_available = False

if scipy_available:
    hbar = cts.hbar
    hbar_eVs = cts.physical_constants['Planck constant over 2 pi in eV s'][0]
    e = cts.e
    m_e = cts.m_e
    kbol_eVK = cts.physical_constants['Boltzmann constant in eV/K'][0]
    physical_constants = cts.physical_constants
else:
    hbar = 1.054571726e-34
    hbar_eVs = 6.58211928e-16
    e = 1.602176565e-19
    m_e = 9.10938291e-31
    kbol = 1.3806488e-23
    kbol_eVK = 8.6173324e-05

