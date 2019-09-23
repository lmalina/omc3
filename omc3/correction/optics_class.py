import numpy as np
import tfs
from utils import logging_tools
from utils.contexts import timeit

LOG = logging_tools.get_logger(__name__)


def get_coupling(tw):
    """ Returns the coupling term.

            .. warning::
                This function changes sign of the real part of the RDTs compared to
                [#FranchiAnalyticformulasrapid2017]_ to be consistent with the RDT
                calculations from [#CalagaBetatroncouplingMerging2005]_ .


                Calculates C matrix and Coupling and Gamma from it.
        See [#CalagaBetatroncouplingMerging2005]_

            Args:
                model:  Model to be used by cmatrix calculation
            """
    res = tfs.TfsDataFrame(index=tw.index)
    res["S"] = tw["S"]

    with timeit(lambda t: LOG.debug(f"  CMatrix calculated in {t} s")):
        j = np.array([[0., 1.],
                      [-1., 0.]])
        rs = np.reshape(tw.as_matrix(columns=["R11", "R12", "R21", "R22"]), (len(tw), 2, 2))
        cs = np.einsum("ij,kjn,no->kio",
                       -j, np.transpose(rs, axes=(0, 2, 1)), j)
        cs = np.einsum("k,kij->kij", (1 / np.sqrt(1 + np.linalg.det(rs))), cs)

        g11a = 1 / np.sqrt(tw.loc[:, "BETX"])
        g12a = np.zeros(len(tw))
        g21a = tw.loc[:, "ALFX"] / np.sqrt(tw.loc[:, "BETX"])
        g22a = np.sqrt(tw.loc[:, "BETX"])
        gas = np.reshape(np.array([g11a, g12a, g21a, g22a]).T, (len(tw), 2, 2))

        ig11b = np.sqrt(tw.loc[:, "BETY"])
        ig12b = np.zeros(len(tw))
        ig21b = -tw.loc[:, "ALFY"] / np.sqrt(tw.loc[:, "BETY"])
        ig22b = 1. / np.sqrt(tw.loc[:, "BETY"])
        igbs = np.reshape(np.array([ig11b, ig12b, ig21b, ig22b]).T, (len(tw), 2, 2))
        cs = np.einsum("kij,kjl,kln->kin", gas, cs, igbs)
        gammas = np.sqrt(1 - np.linalg.det(cs))

        res.loc[:, "GAMMA_C"] = gammas
        res.loc[:, "F1001_C"] = ((cs[:, 0, 0] + cs[:, 1, 1]) * 1j +
                                 (cs[:, 0, 1] - cs[:, 1, 0])) / 4 / gammas
        res.loc[:, "F1010_C"] = ((cs[:, 0, 0] - cs[:, 1, 1]) * 1j +
                                 (-cs[:, 0, 1]) - cs[:, 1, 0]) / 4 / gammas

        LOG.debug(f"  Average coupling amplitude |F1001|: {np.mean(np.abs(res.loc[:, 'F1001_C']))}")
        LOG.debug(f"  Average coupling amplitude |F1010|: {np.mean(np.abs(res.loc[:, 'F1010_C']))}")
        LOG.debug(f"  Average gamma: {np.mean(np.abs(res.loc[:, 'GAMMA_C']))}")

    res_df = res.loc[:, ['S', 'F1001_C', 'F1010_C']]
    return res_df.rename(columns=lambda x: x.replace("_C", ""))
