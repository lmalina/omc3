""" Common Functions not attached to any class in this module.
"""
import numpy as np
import pandas as pd
import itertools
from utils import logging_tools
from utils.contexts import timeit

LOG = logging_tools.get_logger(__name__)


# General Helpers ##############################################################


def upper(list_of_strings):
    """ Set all items of list to uppercase """
    return [item.upper() for item in list_of_strings]


def lower(list_of_strings):
    """ Set all items of list to lowercase """
    return [item.lower() for item in list_of_strings]


# Twiss Helpers ################################################################


def get_all_rdts(n):
    """ Returns list of all valid RDTs of order 2 to n """
    if n <= 1:
        raise ValueError("'n' must be greater 1 for resonance driving terms.")
    permut = [x for x in itertools.product(range(n + 1), repeat=4)
           if 1 < sum(x) <= n and not (x[0] == x[1] and x[2] == x[3])]
    return ['F{:d}{:d}{:d}{:d}'.format(j, k, l, m) for j, k, l, m in sorted(permut, key=sum)]



# Phase Advance Functions ######################################################


def get_phase_advances(twiss_df):
    """
    Calculate phase advances between all elements

    Returns:
        Matrices similar to DPhi(i,j) = Phi(j) - Phi(i)
    """
    LOG.debug("Calculating Phase Advances:")
    phase_advance_dict = dict.fromkeys(['X', 'Y'])
    with timeit(lambda t:
                LOG.debug("  Phase Advances calculated in {:f}s".format(t))):
        for plane in ['X', 'Y']:
            colmn_phase = "MU" + plane

            phases_mdl = twiss_df.loc[twiss_df.index, colmn_phase]
            # Same convention as in [1]: DAdv(i,j) = Phi(j) - Phi(i)
            phase_advances = pd.DataFrame((phases_mdl[None, :] - phases_mdl[:, None]),
                                          index=twiss_df.index,
                                          columns=twiss_df.index)
            # Do not calculate dphi and tau here.
            # only slices of phase_advances as otherwise super slow
            phase_advance_dict[plane] = phase_advances
    return phase_advance_dict


def dphi(data, q):
    """ Return dphi from phase advances in data, see Eq. 8 in [#FranchiAnalyticformulasrapid2017]_
    """
    return data + np.where(data <= 0, q, 0)  # '<=' seems to be what MAD-X does


def tau(data, q):
    """ Return tau from phase advances in data, see Eq. 16 in [#FranchiAnalyticformulasrapid2017]_
    """
    return data + np.where(data <= 0, q / 2, -q / 2)  # '<=' seems to be what MAD-X does
