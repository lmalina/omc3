"""
Provides Classes to calculate optics from twiss parameters.

The calculation is based on formulas in [#AibaFirstbetabeatingmeasurement2009]_ and
[#FranchiAnalyticformulasrapid2017]_.

Only works properly for on-orbit twiss files.

 - Resonance Driving Terms: Eq. A8 in [#FranchiAnalyticformulasrapid2017]_
 - Linear Dispersion: Eq. 24 in [#FranchiAnalyticformulasrapid2017]_
 - Linear Chromaticity: Eq. 31 in [#FranchiAnalyticformulasrapid2017]_
 - Chromatic Beating: Eq. 36 in [#FranchiAnalyticformulasrapid2017]_

.. rubric:: References

.. [#AibaFirstbetabeatingmeasurement2009]
    M. Aiba et al.,
    First simultaneous measurement of sextupolar and octupolar resonance driving
    terms in a circular accelerator from turn-by-turn beam position monitor data,
    Phys. Rev. ST Accel. Beams 17, 074001 (2014).
    https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.17.074001

.. [#FranchiAnalyticformulasrapid2017]
    A. Franchi et al.,
    Analytic formulas for the rapid evaluation of the orbit response matrix
    and chromatic functions from lattice parameters in circular accelerators
    https://arxiv.org/abs/1711.06589

.. [#CalagaBetatroncouplingMerging2005]
    R. Calaga et al.,
    'Betatron coupling: Merging Hamiltonian and matrix approaches'
    Phys. Rev. ST Accel. Beams, vol. 8, no. 3, p. 034001, Mar. 2005.

"""

from math import factorial

import numpy as np


import tfs
from twiss_optics.twiss_functions import get_phase_advances, dphi, get_all_rdts
from utils import logging_tools
from utils.contexts import timeit
from utils.dict_tools import DotDict


LOG = logging_tools.get_logger(__name__)

PLOT_DEFAULTS = {
        "style": 'standard',
        "manual": {u'lines.linestyle': '-',
                   u'lines.marker': '',
                   }
}


################################
#        TwissOptics
################################


class TwissOptics(object):
    """ Class for calculating optics parameters from twiss-model.

    Args:
        model_path_or_df: Path to twissfile of model or DataFrame of model.
        quick_init: Initializes without calculating phase advances. Default: False
    """

    ################################
    #       init Functions
    ################################

    def __init__(self, model_path_or_df, quick_init=True):
        self.twiss_df = self._get_model_df(model_path_or_df)
        self._ip_pos = self._find_ip_positions()

        self._results_df = self._make_results_dataframe()

        self._phase_advance = None
        if not quick_init:
            self._phase_advance = self.get_phase_adv()

        self._plot_options = DotDict(PLOT_DEFAULTS)

    @staticmethod
    def _get_model_df(model_path_or_tfs):
        """ Check if DataFrame given, if not load model from file  """
        if isinstance(model_path_or_tfs, str):
            LOG.debug("Creating TwissOptics from '{:s}'".format(model_path_or_tfs))
            df = tfs.read(model_path_or_tfs, index="NAME")
        else:
            LOG.debug("Creating TwissOptics from input DataFrame")
            df = model_path_or_tfs
            if (len(df.index.values) == 0) or not isinstance(df.index.values[0], str):
                raise IndexError("Index of DataFrame needs to be the element names."
                                 "This does not seem to be the case.")
        return df

    def _find_ip_positions(self):
        """ Returns IP positions from Dataframe.
        Only needed for plotting, so if not present it will just skip it.
        Nice for LHC though.

        Load model into twiss_df first!
        """
        tw = self.twiss_df
        return tw.loc[tw.index.str.match(r"IP\d$", case=False), 'S']

    def _make_results_dataframe(self):
        """ Creating a dataframe used for storing results. """
        LOG.debug("Creating Results Dataframes.")
        results_df = tfs.TfsDataFrame(index=self.twiss_df.index)
        results_df["S"] = self.twiss_df["S"]
        return results_df


    ################################
    #         Properties
    ################################

    def get_phase_adv(self):
        """ Wrapper for returning the matrix of phase-advances. """
        if self._phase_advance is None:
            self._phase_advance = get_phase_advances(self.twiss_df)
        return self._phase_advance

    def get_coupling(self, method='rdt'):
        """ Returns the coupling term.


        .. warning::
            This function changes sign of the real part of the RDTs compared to
            [#FranchiAnalyticformulasrapid2017]_ to be consistent with the RDT
            calculations from [#CalagaBetatroncouplingMerging2005]_ .

        Args:
            method: 'rdt' - Returns the values calculated by calc_rdts()
                    'cmatrix' - Returns the values calculated by calc_cmatrix()
        """
        if method == 'rdt':
            if "F1001" not in self._results_df or "F1010" not in self._results_df:
                self.calc_rdts(['F1001', 'F1010'])
            res_df = self._results_df.loc[:, ['S', 'F1001', 'F1010']]
            res_df.loc[:, "F1001"].real *= -1
            res_df.loc[:, "F1010"].real *= -1
            return res_df
        elif method == 'cmatrix':
            if "F1001_C" not in self._results_df:
                self.calc_cmatrix()
            res_df = self._results_df.loc[:, ['S', 'F1001_C', 'F1010_C']]
            return res_df.rename(columns=lambda x: x.replace("_C", ""))
        else:
            raise ValueError("method '{:s}' not recognized.".format(method))

    ################################
    #          C Matrix
    ################################

    def calc_cmatrix(self):
        """ Calculates C matrix and Coupling and Gamma from it.
        See [#CalagaBetatroncouplingMerging2005]_
        """
        tw = self.twiss_df
        res = self._results_df

        LOG.debug("Calculating CMatrix.")
        with timeit(lambda t:
                    LOG.debug("  CMatrix calculated in {:f}s".format(t))):

            j = np.array([[0., 1.],
                          [-1., 0.]])
            rs = np.reshape(tw.as_matrix(columns=["R11", "R12",
                                                  "R21", "R22"]),
                            (len(tw), 2, 2))
            cs = np.einsum("ij,kjn,no->kio",
                           -j, np.transpose(rs, axes=(0, 2, 1)), j)
            cs = np.einsum("k,kij->kij", (1 / np.sqrt(1 + np.linalg.det(rs))), cs)

            g11a = 1 / np.sqrt(tw.loc[:, "BETX"])
            g12a = np.zeros(len(tw))
            g21a = tw.loc[:, "ALFX"] / np.sqrt(tw.loc[:, "BETX"])
            g22a = np.sqrt(tw.loc[:, "BETX"])
            gas = np.reshape(np.array([g11a, g12a,
                                       g21a, g22a]).T,
                             (len(tw), 2, 2))

            ig11b = np.sqrt(tw.loc[:, "BETY"])
            ig12b = np.zeros(len(tw))
            ig21b = -tw.loc[:, "ALFY"] / np.sqrt(tw.loc[:, "BETY"])
            ig22b = 1. / np.sqrt(tw.loc[:, "BETY"])
            igbs = np.reshape(np.array([ig11b, ig12b,
                                        ig21b, ig22b]).T,
                              (len(tw), 2, 2))
            cs = np.einsum("kij,kjl,kln->kin", gas, cs, igbs)
            gammas = np.sqrt(1 - np.linalg.det(cs))

            res.loc[:, "GAMMA_C"] = gammas

            res.loc[:, "F1001_C"] = ((cs[:, 0, 0] + cs[:, 1, 1]) * 1j +
                                     (cs[:, 0, 1] - cs[:, 1, 0])) / 4 / gammas
            res.loc[:, "F1010_C"] = ((cs[:, 0, 0] - cs[:, 1, 1]) * 1j +
                                     (-cs[:, 0, 1]) - cs[:, 1, 0]) / 4 / gammas

            res.loc[:, "C11"] = cs[:, 0, 0]
            res.loc[:, "C12"] = cs[:, 0, 1]
            res.loc[:, "C21"] = cs[:, 1, 0]
            res.loc[:, "C22"] = cs[:, 1, 1]

            LOG.debug("  Average coupling amplitude |F1001|: {:g}".format(np.mean(
                np.abs(res.loc[:, "F1001_C"]))))
            LOG.debug("  Average coupling amplitude |F1010|: {:g}".format(np.mean(
                np.abs(res.loc[:, "F1010_C"]))))
            LOG.debug("  Average gamma: {:g}".format(np.mean(
                np.abs(res.loc[:, "GAMMA_C"]))))

        self._log_added('GAMMA_C', 'F1001_C', 'F1010_C', 'C11', 'C12', 'C21', 'C22')


    ################################
    #   Resonance Driving Terms
    ################################

    def calc_rdts(self, order_or_rdts):
        """ Calculates the Resonance Driving Terms.
        
        Eq. A8 in [#FranchiAnalyticformulasrapid2017]_

        Args:
            order_or_rdts: int, string or list of strings
                If an int is given all Resonance Driving Terms up to this order
                will be calculated.
                The strings are assumed to be the desired driving term names, e.g. "F1001"
        """
        if isinstance(order_or_rdts, int):
            rdt_list = get_all_rdts(order_or_rdts)
        elif not isinstance(order_or_rdts, list):
            rdt_list = [order_or_rdts]
        else:
            rdt_list = order_or_rdts

        LOG.debug("Calculating RDTs: {:s}.".format(str(rdt_list)[1:-1]))
        with timeit(lambda t:
                    LOG.debug("  RDTs calculated in {:f}s".format(t))):

            i2pi = 2j * np.pi
            tw = self.twiss_df
            phs_adv = self.get_phase_adv()
            res = self._results_df

            for rdt in rdt_list:
                if not len(rdt) == 5 and rdt[0].upper() == 'F':
                    ValueError(f"'{rdt}' does not seem to be a valid RDT name.")

                conj_rdt = ''.join(['F', rdt[2], rdt[1], rdt[4], rdt[3]])

                if conj_rdt in self._results_df:
                    res[rdt.upper()] = np.conjugate(self._results_df[conj_rdt])
                else:
                    j, k, l, m = int(rdt[1]), int(rdt[2]), int(rdt[3]), int(rdt[4])
                    n = j + k + l + m

                    if n < 2:
                        ValueError(f"The RDT-order has to be >1 but was {n:d} for {rdt}")

                    denom1 = 1./(factorial(j) * factorial(k) * factorial(l) * factorial(m) * 2**n)
                    denom2 = 1./(1. - np.exp(i2pi * ((j-k) * tw.Q1 + (l-m) * tw.Q2)))

                    if (l + m) % 2 == 0:
                        src = 'K' + str(n-1) + 'L'
                        sign = -(1j ** (l+m))
                    else:
                        src = 'K' + str(n-1) + 'SL'
                        sign = -(1j ** (l+m+1))

                    try:
                        mask_in = tw[src] != 0
                        if sum(mask_in) == 0:
                            raise KeyError
                    except KeyError:
                        # either src is not in tw or all k's are zero.
                        LOG.warning("  All {:s} == 0. RDT '{:s}' will be zero.".format(src, rdt))
                        res.loc[:, rdt.upper()] = 0
                    else:
                        # the next three lines determine the main order of speed, hence
                        # - mask as much as possible
                        # - additions are faster than multiplications (-> applymap last)
                        phx = dphi(phs_adv['X'].loc[mask_in, :], tw.Q1)
                        phy = dphi(phs_adv['Y'].loc[mask_in, :], tw.Q2)
                        phase_term = ((j-k) * phx + (l-m) * phy).applymap(lambda p: np.exp(i2pi*p))

                        beta_term = (tw.loc[mask_in, src] *
                                     tw.loc[mask_in, 'BETX'] ** ((j+k) / 2.) *
                                     tw.loc[mask_in, 'BETY'] ** ((l+m) / 2.)
                                     )

                        res.loc[:, rdt.upper().replace('F', 'H')] = sign * phase_term.multiply(
                            beta_term, axis="index").sum(axis=0).transpose() * denom1

                        res.loc[:, rdt.upper()] = sign * phase_term.multiply(
                            beta_term, axis="index").sum(axis=0).transpose() * denom1 * denom2

                        LOG.debug("  Average RDT amplitude |{:s}|: {:g}".format(rdt, np.mean(
                            np.abs(res.loc[:, rdt.upper()]))))

        self._log_added(*rdt_list)

    @staticmethod
    def _log_added(*args):
        """ Logging Helper to log which fields were added """
        if len(args) > 0:
            LOG.debug(f"  Added fields to results: '{', '.join(args)}'")
