"""
:module: correction.getdiff

Created on 24/02/18

:author: Lukas Malina

Calculates the difference between GetLLM output and correction plugged in the model.
Provide as first argument the path to the output files of GetLLM.

model inputs:
    twiss_cor.dat and twiss_no.dat

outputs in measurement directory:
    phasex.out and phasey.out
    bbx.out and bby.out
    dx.out, dy.out and ndx.out
    couple.out and chromatic_coupling.out

TODOs and Notes:
    OpticsMeasurement: possibly extend and use with measurement filters from global correction
                        to be used in sbs, new corrections, getdiff, plot_export

    Expected values after correction to be put in, little tricky with phase column names
    No coupling in twiss_no.dat? not used

Some hints:
    MEA, MODEL, EXPECT are usually the names for the differences between the values and the model.
    Apart from phase, where these differences are called DIFF and DIFF_MDL (EXPECT is still the
    same) while MEA and MODEL are the actual measurement and model values respectively.

    Don't look into the coupling and chromatic coupling namings.
"""
import sys
from os.path import join, isdir
import numpy as np
import pandas as pd

from optics_measurements.io_filehandler import OpticsMeasurement
from correction import optics_class
import tfs
from utils import logging_tools #, beta_star_from_twiss as bsft
from optics_measurements.constants import DELTA, ERR
from optics_measurements.toolbox import df_rel_diff, df_diff
LOG = logging_tools.get_logger(__name__)

TWISS_CORRECTED = "twiss_cor.dat"
TWISS_NOT_CORRECTED = "twiss_no.dat"
TWISS_CORRECTED_PLUS = "twiss_cor_dpp.dat"  # positive dpp
TWISS_CORRECTED_MINUS = "twiss_cor_dpm.dat"  # negative dpp

PLANES = ("X", "Y")

MEA = "MEA"
ERROR = "ERROR"
MODEL = "MODEL"
EXPECT = "EXPECT"

CORR = "_c"
NO = "_n"
NAME = "NAME"
S= "S"
# Main invocation ############################################################


def get_diff_filename(id):
    return f"diff_{id}.out"


def getdiff(meas_path=None, beta_file_name="beta_phase_"):
    """ Calculates the differences between measurement, corrected and uncorrected model.

    After running madx and creating the model with (twiss_cor) and without (twiss_no)
    corrections, run this functions to get tfs-files with the differences between measurements
    and models.

    Args:
        meas_path (str): Path to the measurement folder.
        Needs to contain twiss_cor.dat and twiss_no.dat.
    """
    if meas_path is None:
        meas_path = sys.argv[1]

    LOG.debug(f"Started getdiff for measurment directory: {meas_path}")

    if not isdir(meas_path):
        raise IOError(f"No valid measurement directory: {meas_path}")
    corrected_model_path = join(meas_path, TWISS_CORRECTED)
    uncorrected_model_path = join(meas_path, TWISS_NOT_CORRECTED)

    meas = OpticsMeasurement(meas_path)
    twiss_cor = tfs.read(corrected_model_path).set_index('NAME', drop=False)
    twiss_no = tfs.read(uncorrected_model_path).set_index('NAME', drop=False)
    coup_cor = optics_class.coupling_from_r_matrix(twiss_cor)
    coup_no = optics_class.coupling_from_r_matrix(twiss_no)
    model = pd.merge(twiss_cor, twiss_no, how='inner', on='NAME', suffixes=(CORR, NO))
    coupling_model = pd.merge(coup_cor, coup_no, how='inner', left_index=True, right_index=True,
                              suffixes=(CORR, NO))
    coupling_model['NAME'] = coupling_model.index.values

    for plane in PLANES:
        _write_betabeat_diff_file(meas_path, meas, model, plane, beta_file_name)
        _write_phase_diff_file(meas_path, meas, model, plane)
        _write_disp_diff_file(meas_path, meas, model, plane)
        _write_closed_orbit_diff_file(meas_path, meas, model, plane)
    _write_coupling_diff_file(meas_path, meas, coupling_model)
    _write_norm_disp_diff_file(meas_path, meas, model)
    #_write_chromatic_coupling_files(meas_path, corrected_model_path)
    #_write_betastar_diff_file(meas_path, meas, twiss_cor, twiss_no)
    LOG.debug("Finished 'getdiff'.")


# Writing Functions ##########################################################


def _write_betabeat_diff_file(meas_path, meas, model, plane, betafile):
    LOG.debug("Calculating beta diff.")
    if betafile == "getbeta":
        meas_beta = meas.beta[plane]
    elif betafile == "getampbeta":
        meas_beta = meas.amp_beta[plane]
    elif betafile == "getkmodbeta":
        meas_beta = meas.kmod_beta[plane]
    else:
        raise KeyError("Unknown beta file name '{}'.".format(betafile))

    tw = pd.merge(meas_beta, model, how='inner', on='NAME')
    tw[MEA] = tw.loc[:, f"{DELTA}BET{plane}"].values
    tw[ERROR] = tw.loc[:, f"{ERR}{DELTA}BET{plane}"].values
    tw[MODEL] = df_rel_diff(tw, f"BET{plane}_c", f"BET{plane}_n")
    tw[EXPECT] = df_diff(tw, MEA, MODEL)
    tfs.write(join(meas_path, get_diff_filename('bb' + plane.lower())),
              tw.loc[:, [NAME, S, MEA, ERROR, MODEL, EXPECT]])


def _write_phase_diff_file(meas_path, meas, model, plane):
    LOG.debug("Calculating phase diff.")
    tw = pd.merge(meas.phase[plane], model, how='inner', on='NAME')
    tw[MEA] = tw.loc[:, f"{DELTA}PHASE{plane}"].values
    tw[ERROR] = tw.loc[:, f"{ERR}{DELTA}PHASE{plane}"].values
    tw[MODEL] = np.concatenate((np.diff(df_diff(tw, f"MU{plane}{CORR}", f"MU{plane}{NO}")), np.array([0.0])))
    tw[EXPECT] = df_diff(tw, MEA, MODEL)
    tfs.write(join(meas_path, get_diff_filename('phase' + plane.lower())),
              tw.loc[tw.index[:-1], [NAME, S, MEA, ERROR, MODEL, EXPECT]])


def _write_disp_diff_file(meas_path, meas, model, plane):
    LOG.debug("Calculating dispersion diff.")
    try:
        tw = pd.merge(meas.disp[plane], model, how='inner', on='NAME')
    except IOError:
        LOG.debug("Dispersion measurements not found. Skipped.")
    else:
        tw[MEA] = tw.loc[:, f"{DELTA}D{plane}"].values
        tw[ERROR] = tw.loc[:, f"{ERR}{DELTA}D{plane}"].values
        tw[MODEL] = df_diff(tw, f"D{plane}{CORR}", f"D{plane}{NO}")
        tw[EXPECT] = df_diff(tw, MEA, MODEL)
        tfs.write(join(meas_path, get_diff_filename('d' + plane.lower())),
                  tw.loc[:, [NAME, S, MEA, ERROR, MODEL, EXPECT]])


def _write_closed_orbit_diff_file(meas_path, meas, model, plane):
    LOG.debug("Calculating orbit diff.")
    up = plane.upper()
    try:
        tw = pd.merge(meas.orbit[plane], model, how='inner', on='NAME')
    except IOError:
        LOG.debug("Orbit measurements not found. Skipped.")
    else:
        tw[MEA] = tw.loc[:, f"{DELTA}{plane}"].values
        tw[ERROR] = tw.loc[:, f"{ERR}{DELTA}{plane}"].values
        tw[MODEL] = df_diff(tw, f"{plane}{CORR}", f"{plane}{NO}")
        tw[EXPECT] = df_diff(tw, MEA, MODEL)
        tfs.write(join(meas_path, get_diff_filename('co' + plane.lower())),
                  tw.loc[:, [NAME, S, MEA, ERROR, MODEL, EXPECT]])


def _write_norm_disp_diff_file(meas_path, meas, model, plane="X"):
    LOG.debug("Calculating normalized dispersion diff.")
    try:
        tw = pd.merge(meas.norm_disp, model, how='inner', on='NAME')
    except IOError:
        LOG.debug("Normalized dispersion measurements not found. Skipped.")
    else:
        tw[MEA] = tw.loc[:, f"{DELTA}ND{plane}"].values
        tw[ERROR] = tw.loc[:, f"{ERR}{DELTA}ND{plane}"].values
        tw[MODEL] = (tw.loc[:, f"D{plane}{CORR}"].values / np.sqrt(tw.loc[:, f"BET{plane}{CORR}"].values)
                     - tw.loc[:, f"D{plane}{NO}"].values / np.sqrt(tw.loc[:, f"BET{plane}{NO}"].values))
        tw[EXPECT] = df_diff(tw, MEA, MODEL)
        tfs.write(join(meas_path, get_diff_filename('ndx')),
                  tw.loc[:, [NAME, S, MEA, ERROR, MODEL, EXPECT]])


def _write_coupling_diff_file(meas_path, meas, model):
    LOG.debug("Calculating coupling diff.")
    tw = pd.merge(meas.coupling, model, how='inner', on='NAME')
    out_columns = ['NAME', 'S']
    for idx, rdt in enumerate(['F1001', 'F1010']):
        tw[rdt + 're'] = tw.loc[:, rdt + 'R']
        tw[rdt + 'im'] = tw.loc[:, rdt + 'I']
        tw[rdt + 'e'] = tw.loc[:, 'FWSTD{:d}'.format(idx + 1)]
        tw[rdt + 're_m'] = np.real(tw.loc[:, rdt + '_c'])
        tw[rdt + 'im_m'] = np.imag(tw.loc[:, rdt + '_c'])
        tw[rdt + 're_prediction'] = tw.loc[:, rdt + 're'] - tw.loc[:, rdt + 're_m']
        tw[rdt + 'im_prediction'] = tw.loc[:, rdt + 'im'] - tw.loc[:, rdt + 'im_m']
        tw[rdt + 'W_prediction'] = np.sqrt(np.square(tw[rdt + 're_prediction'])
                                           + np.square(tw[rdt + 'im_prediction']))

        out_columns += [rdt + 're', rdt + 'im', rdt + 'e',
                        rdt + 're_m', rdt + 'im_m',
                        rdt + 'W', rdt + 'W_prediction',
                        rdt + 're_prediction', rdt + 'im_prediction']

    tw['in_use'] = 1
    out_columns += ['in_use']
    tfs.write(join(meas_path, get_diff_filename('couple')), tw.loc[:, out_columns])


# def _write_chromatic_coupling_files(meas_path, cor_path):
#     LOG.debug("Calculating chromatic coupling diff.")
#     # TODO: Add Cf1010
#     try:
#         twiss_plus = tfs.read(join(split(cor_path)[0], TWISS_CORRECTED_PLUS), index='NAME')
#         twiss_min = tfs.read(join(split(cor_path)[0], TWISS_CORRECTED_MINUS), index='NAME')
#     except IOError:
#         LOG.debug("Chromatic coupling measurements not found. Skipped.")
#     else:
#         deltap = np.abs(twiss_plus.DELTAP - twiss_min.DELTAP)
#         plus = optics_class.get_coupling(twiss_plus)
#         minus = optics_class.get_coupling(twiss_min)
#         model = pd.merge(plus, minus, how='inner', left_index=True, right_index=True,
#                          suffixes=('_p', '_m'))
#         model['NAME'] = model.index.values
#         if exists(join(meas_path, "chromcoupling_free.out")):
#             meas = tfs.read(join(meas_path, "chromcoupling_free.out"))
#         else:
#             meas = tfs.read(join(meas_path, "chromcoupling.out"))
#         tw = pd.merge(meas, model, how='inner', on='NAME')
#         cf1001 = (tw.loc[:, 'F1001_p'] - tw.loc[:, 'F1001_m']) / deltap
#         tw['Cf1001r_model'] = np.real(cf1001)
#         tw['Cf1001i_model'] = np.imag(cf1001)
#         tw['Cf1001r_prediction'] = tw.loc[:, 'Cf1001r'] - tw.loc[:, 'Cf1001r_model']
#         tw['Cf1001i_prediction'] = tw.loc[:, 'Cf1001i'] - tw.loc[:, 'Cf1001i_model']
#         tfs.write(join(meas_path, get_diff_filename('chromatic_coupling')),
#                   tw.loc[:, ['NAME', 'S',
#                              'Cf1001r', 'Cf1001rERR',
#                              'Cf1001i', 'Cf1001iERR',
#                              'Cf1001r_model', 'Cf1001i_model',
#                              'Cf1001r_prediction', 'Cf1001i_prediction']])


# def _write_betastar_diff_file(meas_path, meas, twiss_cor, twiss_no):
#     LOG.debug("Calculating betastar diff at the IPs.")
#     try:
#         meas = meas.kmod_betastar.set_index(bsft.RES_COLUMNS[0])
#     except IOError:
#         LOG.debug("Beta* measurements not found. Skipped.")
#     else:
#         # get all IPs
#         ip_map = {}
#         beam = ''
#         for label in meas.index.values:
#             ip, beam = re.findall(r'\d', label)[-2:]  # beam should be the same for all
#             if ip not in "1258":
#                 raise NotImplementedError(
#                     "Beta-Star comparison is not yet implemented for measurements in IP" + ip)
#             ip_label = "IP" + ip
#             ip_map[label] = ip_label
#
#         beam = int(beam)
#         all_ips = set(ip_map.values())
#         try:
#             # calculate waist and so on
#             model = bsft.get_beta_star_and_waist_from_ip(twiss_cor, beam, all_ips)
#             design = bsft.get_beta_star_and_waist_from_ip(twiss_no, beam, all_ips)
#         except KeyError:
#             LOG.warn("Can't find all IPs in twiss files. Skipped beta* calculations.")
#         else:
#             # extract data
#             tw = pd.DataFrame()
#             for label in meas.index:
#                 plane = label[-1]
#                 ip_name = bsft.get_full_label(ip_map[label], beam, plane)
#                 tw.loc[label, "S"] = model.loc[ip_name, "S"]
#
#                 # calculate alpha* but with s-oriented waist definition
#                 meas["ALPHASTAR"] = meas["WAIST"] / meas["BETAWAIST"]
#                 meas["ALPHASTAR_ERR"] = ((meas["WAIST_ERR"] / meas["WAIST"] +
#                                           meas["BETAWAIST_ERR"] / meas["BETAWAIST"]) *
#                                          meas["ALPHASTAR"]
#                                          )
#                 for attr in bsft.RES_COLUMNS[2:]:
#                     # default diff parameter
#                     tw.loc[label, attr + "_MEA"] = (meas.loc[label, attr]
#                                                     - design.loc[ip_name, attr])
#                     tw.loc[label, attr + "_ERROR"] = meas.loc[label, attr + "_ERR"]
#                     tw.loc[label, attr + "_MODEL"] = (model.loc[ip_name, attr]
#                                                       - design.loc[ip_name, attr])
#
#                     # additional for checks (e.g. for betastar* panel)
#                     tw.loc[label, attr + "_MEAVAL"] = meas.loc[label, attr]
#                     tw.loc[label, attr + "_DESIGNVAL"] = design.loc[ip_name, attr]
#                     tw.loc[label, attr + "_MODELVAL"] = model.loc[ip_name, attr]
#
#                     # and the beatings
#                     tw.loc[label, "B{}_MEA".format(attr)] = (tw.loc[label, attr + "_MEA"]
#                                                              / design.loc[ip_name, attr])
#                     tw.loc[label, "B{}_MODEL".format(attr)] = (tw.loc[label, attr + "_MODEL"]
#                                                                / design.loc[ip_name, attr])
#
#                     # special handling for the expectation values, as waist and betawaist
#                     # should be derived directly from alpha* and beta*
#                     if attr in bsft.RES_COLUMNS[2:4]:
#                         # beta* and alpha*: as usual
#                         tw.loc[label, attr + "_EXPECT"] = (tw.loc[label, attr + "_MEA"]
#                                                            - tw.loc[label, attr + "_MODEL"])
#                         tw.loc[label, attr + "_EXPECTVAL"] = (design.loc[ip_name, attr]
#                                                               + tw.loc[label, attr + "_EXPECT"])
#                         tw.loc[label, "B{}_EXPECT".format(attr)] = (
#                                 tw.loc[label, "B{}_MEA".format(attr)]
#                                 - tw.loc[label, "B{}_MODEL".format(attr)])
#
#                     else:
#                         # waist and betawaist: calculate expected value directly and go from there
#                         tw.loc[label, attr + "_EXPECTVAL"] = (
#                             bsft.get_waist_wrapper(attr,
#                                                    tw.loc[label, "BETASTAR_EXPECTVAL"],
#                                                    tw.loc[label, "ALPHASTAR_EXPECTVAL"],
#                                                    )
#                         )
#
#                         tw.loc[label, attr + "_EXPECT"] = (
#                                 tw.loc[label, attr + "_EXPECTVAL"] - design.loc[ip_name, attr])
#
#                         tw.loc[label, "B{}_EXPECT".format(attr)] = (
#                                 tw.loc[label, attr + "_EXPECTVAL"] / design.loc[ip_name, attr])
#
#             tfs.write(join(meas_path, get_diff_filename('betastar')), tw,
#                       save_index=bsft.RES_COLUMNS[0])


if __name__ == "__main__":
    getdiff()
