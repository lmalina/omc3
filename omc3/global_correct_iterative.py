r"""
Iterative Correction Scheme.

The response matrices :math:`R_{O}` for the observables :math:`O` (e.g. BBX, MUX, ...)
are loaded from a file and then the equation

.. math:: R_{O} \cdot \delta var = O_{meas} - O_{model}
    :label: eq1

is being solved for :math:`\delta var` via a chosen method (at the moment only numpys pinv,
which creates a pseudo-inverse via svd is used).

The response matrices are hereby merged into one matrix for all observables to solve vor all
:math:`\delta var` at the same time.

To normalize the observables to another ``weigths`` (W) can be applied.

Furthermore, an ``errorcut``, specifying the maximum errorbar for a BPM to be used, and
``modelcut``, specifying the maximum distance between measurement and model for a BPM to be used,
can be defined. Data from BPMs outside of those cut-values will be discarded.
These cuts are defined for each observable separately.

After each iteration the model variables are changed by :math:`-\delta var` and the
observables are recalculated by Mad-X.
:eq:`eq1` is then solved again.


:author: Lukas Malina, Joschua Dilly


Possible problems and notes:
 * error-based weights default? likely - but be carefull with low tune errors vs
svd cut in pseudoinverse
 * manual creation of pd.DataFrame varslist, deltas? maybe
tunes in tfs_pandas single value or a column?
 * There should be some summation/renaming for iterations
 * For two beam correction
 * The two beams can be treated separately until the calcultation of correction
 * Values missing in the response (i.e. correctors of the other beam) shall be
treated as zeros
 * Missing a part that treats the output from LSA

"""
import os
import datetime
import time
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import OrthogonalMatchingPursuit

import tfs
import madx_wrapper
from model import manager
from optics_measurements.constants import (PHASE_NAME, DISPERSION_NAME, NORM_DISP_NAME, EXT)
from correction.fullresponse import response_twiss
from twiss_optics.optics_class import TwissOptics
from utils import iotools, logging_tools, stats
from generic_parser.entrypoint import entrypoint, EntryPointParameters
LOG = logging_tools.get_logger(__name__)

ERR = "ERR"
MDL = "MDL"
DELTA = "DELTA"
VALUE = "VALUE"
WEIGHT = "WEIGHT"
ERROR = "ERROR"
MODEL = "MODEL"
DIFF = "DIFF"

# Configuration ##################################################################

OPTICS_PARAMS_CHOICES = ('MUX', 'MUY', 'BETX', 'BETY', 'DX', 'DY', 'NDX',
                         'Q', 'F1001R', 'F1001I', 'F1010R', 'F1010I')

CORRECTION_DEFAULTS = {
    "optics_file": None,
    "output_filename": "changeparameters_iter",
    "svd_cut": 0.01,
    "optics_params": ['MUX', 'MUY', 'BETX', 'BETY', 'NDX', 'Q'],
    "variables": ["MQM", "MQT", "MQTL", "MQY"],
    "beta_file_name": "beta_phase_",
    "method": "pinv",
    "max_iter": 3,
    }


# Define functions here, to new optics params
def _get_default_values():
    return {
        'modelcut': {
            'MUX': 0.05, 'MUY': 0.05,
            'BETX': 0.2, 'BETY': 0.2,
            'DX': 0.2, 'DY': 0.2,
            'NDX': 0.2, 'Q': 0.1,
            'F1001R': 0.0, 'F1001I': 0.0,
            'F1010R': 0.0, 'F1010I': 0.0,
        },
        'errorcut': {
            'MUX': 0.035, 'MUY': 0.035,
            'BETX': 0.02, 'BETY': 0.02,
            'DX': 0.02, 'DY': 0.02,
            'NDX': 0.02, 'Q': 0.027,
            'F1001R': 0.02, 'F1001I': 0.02,
            'F1010R': 0.02, 'F1010I': 0.02,
        },
        'weights': {
            'MUX': 1, 'MUY': 1,
            'BETX': 0, 'BETY': 0,
            'DX': 0, 'DY': 0,
            'NDX': 0, 'Q': 10,
            'F1001R': 0, 'F1001I': 0,
            'F1010R': 0, 'F1010I': 0,
        },
    }


def _get_measurement_filters():
    return defaultdict(lambda: _get_filtered_generic, {'Q': _get_tunes})


def _get_response_filters():
    return defaultdict(lambda:  _get_generic_response, {
        'MUX': _get_phase_response, 'MUY': _get_phase_response,
        'Q': _get_tune_response})


def _get_model_appenders():
    return defaultdict(lambda:  _get_model_generic, {
        'MUX': _get_model_phases, 'MUY': _get_model_phases,
        'BETX': _get_model_betabeat, 'BETY': _get_model_betabeat,
        'NDX': _get_model_norm_disp, 'Q': _get_model_tunes, })


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(flags="--meas_dir", name="meas_dir", required=True,
                         help="Path to the directory containing the measurement files.",)
    params.add_parameter(flags="--model_dir", name="model_dir", required=True,
                         help="Path to the model to use.",)
    params.add_parameter(flags="--output_dir", name="output_dir", required=True,
                         help="Path to the directory where to write the output files.", )
    params.add_parameter(flags="--fullresponse", name="fullresponse_path",
                         help="Path to the fullresponse binary file.If not given, "
                              "calculates the response analytically.",)
    params.add_parameter(flags="--optics_params", name="optics_params", type=str, nargs="+",
                         default=CORRECTION_DEFAULTS["optics_params"],
                         choices=('MUX', 'MUY', 'BBX', 'BBY', 'BETX', 'BETY', 'DX', 'DY', 'NDX', 'Q',
                                  'F1001R', 'F1001I', 'F1010R', 'F1010I'),
                         help="List of parameters to correct upon (e.g. BETX BETY)", )
    params.add_parameter(flags="--modifiers", name="modifiers",
                         help="Path to the optics file to use. If not present will default to "
                              "model_path/modifiers.madx, if such a file exists.", )
    params.add_parameter(flags="--output_filename", name="output_filename",
                         default=CORRECTION_DEFAULTS["output_filename"],
                         help="Identifier of the output files.", )
    params.add_parameter(flags="--min_corrector_strength", name="min_corrector_strength",
                         type=float, default=0.,
                         help="Minimum (absolute) strength of correctors.",)
    params.add_parameter(flags="--model_cut", name="modelcut", nargs="+", type=float,
                         help="Reject BPMs whose deviation to the model is higher "
                              "than the correspoding input. Input in order of optics_params.",)
    params.add_parameter(flags="--error_cut", name="errorcut", nargs="+", type=float,
                         help="Reject BPMs whose error bar is higher than the corresponding "
                              "input. Input in order of optics_params.",)
    params.add_parameter(flags="--weights", name="weights", nargs="+", type=float,
                         help="Weight to apply to each measured quantity. "
                              "Input in order of optics_params.",)
    params.add_parameter(flags="--variables", name="variable_categories", nargs="+",
                         default=CORRECTION_DEFAULTS["variables"],
                         help="List of names of the variables classes to use.", )
    params.add_parameter(flags="--beta_file_name", name="beta_file_name",
                         default=CORRECTION_DEFAULTS["beta_file_name"],
                         help="Prefix of the beta file to use. E.g.: getkmodbeta", )
    params.add_parameter(flags="--method", name="method", type=str, choices=["pinv", "omp"],
                         default=CORRECTION_DEFAULTS["method"],
                         help="Optimization method to use.", )
    params.add_parameter(flags="--svd_cut", name="svd_cut", type=float,
                         default=CORRECTION_DEFAULTS["svd_cut"],
                         help="Cutoff for small singular values of the pseudo inverse. "
                              "(Method: 'pinv')Singular values smaller than "
                              "rcond*largest_singular_value are set to zero", )
    params.add_parameter(flags="--n_correctors", name="n_correctors", type=int,
                         help="Maximum number of correctors to use. (Method: 'omp')")
    params.add_parameter(flags="--max_iter", name="max_iter", type=int,
                         default=CORRECTION_DEFAULTS["max_iter"],
                         help="Maximum number of correction re-iterations to perform. "
                              "A value of `0` means the correction is calculated once.", )
    params.add_parameter(flags="--use_errorbars", name="use_errorbars", action="store_true",
                         help="Take into account the measured errorbars in the correction.", )
    params.add_parameter(flags="--update_response", action="store_true", name="update_response",
                         help="Update the (analytical) response per iteration.", )
    params.add_parameter(flags="--virt_flag", name="virt_flag", action="store_true",
                         help="If true, it will use virtual correctors.", )
    return params


# Entry Point ##################################################################

@entrypoint(_get_params())
def global_correction(opt, accel_opt):
    """ Do the global correction. Iteratively.
        # TODO auto-generate docstring
    """
    LOG.info("Starting Iterative Global Correction.")
    # check on opt
    opt = _check_opt_add_dicts(opt)
    opt = _add_hardcoded_paths(opt)
    iotools.create_dirs(opt.output_dir)
    meth_opt = _get_method_opt(opt)

    # get accelerator class
    accel_cls = manager.get_accel_class(accel_opt)
    accel_inst = accel_cls(model_dir=opt.model_dir)
    if opt.modifiers is not None:
        accel_inst.modifiers = opt.modifiers

    # read data from files
    vars_list = _get_varlist(accel_cls, opt.variable_categories, opt.virt_flag)
    optics_params, meas_dict = _get_measurment_data(opt.optics_params, opt.meas_dir,
                                                    opt.beta_file_name, opt.weights,)

    if opt.fullresponse_path is not None:
        resp_dict = _load_fullresponse(opt.fullresponse_path, vars_list)
    else:
        resp_dict = response_twiss.create_response(accel_inst, opt.variable_categories,
                                                   optics_params)
    # the model in accel_inst is modified later, so save nominal model here to variables
    nominal_model = _maybe_add_coupling_to_model(accel_inst.get_model_tfs(), optics_params)

    # apply filters to data
    meas_dict = _filter_measurement(optics_params, meas_dict, nominal_model, opt.use_errorbars,
                                    opt.weights, opt.errorcut, opt.modelcut)
    meas_dict = _append_model_to_measurement(nominal_model, meas_dict, optics_params)
    resp_dict = _filter_response_index(resp_dict, meas_dict, optics_params)
    resp_matrix = _join_responses(resp_dict, optics_params, vars_list)
    delta = tfs.TfsDataFrame(0, index=vars_list, columns=[DELTA])

    # ######### Iteration Phase ######### #

    for iteration in range(opt.max_iter + 1):
        LOG.info(f"Correction Iteration {iteration} of {opt.max_iter}.")

        # ######### Update Model and Response ######### #
        if iteration > 0:
            LOG.debug("Updating model via MADX.")
            corr_model_path = os.path.join(opt.output_dir, f"twiss_{iteration}{EXT}")
            _create_corrected_model(corr_model_path, opt.change_params_path, accel_inst)

            corr_model_elements = tfs.read(corr_model_path, index="NAME")
            corr_model_elements = _maybe_add_coupling_to_model(corr_model_elements, optics_params)

            bpms_index_mask = accel_inst.get_element_types_mask(corr_model_elements.index,
                                                                types=["bpm"])
            corr_model = corr_model_elements.loc[bpms_index_mask, :]

            meas_dict = _append_model_to_measurement(corr_model, meas_dict, optics_params)
            if opt.update_response:
                LOG.debug("Updating response.")
                # please look away for the next two lines.
                accel_inst._model = corr_model
                accel_inst._elements = corr_model_elements
                resp_dict = response_twiss.create_response(accel_inst, opt.variable_categories,
                                                           optics_params)
                resp_dict = _filter_response_index(resp_dict, meas_dict, optics_params)
                resp_matrix = _join_responses(resp_dict, optics_params, vars_list)

        # ######### Actual optimization ######### #
        delta += _calculate_delta(resp_matrix, meas_dict, optics_params, vars_list, opt.method,
                                  meth_opt)
        delta, resp_matrix, vars_list = _filter_by_strength(delta, resp_matrix,
                                                            opt.min_corrector_strength)
        # remove unused correctors from vars_list

        writeparams(opt.change_params_path, delta)
        writeparams(opt.change_params_correct_path, -delta)
        LOG.debug(f"Cumulative delta: {np.sum(np.abs(delta.loc[:, DELTA].values)):.5e}")

    write_knob(opt.knob_path, delta)
    LOG.info("Finished Iterative Global Correction.")

# Main function helpers #######################################################


def _check_opt_add_dicts(opt):
    """ Check on options and put in missing values """
    def_dict = _get_default_values()
    opt.optics_params = [p.replace("BB", "BET") for p in opt.optics_params]
    for key in ("modelcut", "errorcut", "weights"):
        if opt[key] is None:
            opt[key] = [def_dict[key][p] for p in opt.optics_params]
        elif len(opt[key]) != len(opt.optics_params):
            raise AttributeError(f"Length of {key} is not the same as of the optical parameters!")
        opt[key] = dict(zip(opt.optics_params, opt[key]))

    return opt


def _add_hardcoded_paths(opt):
    opt.change_params_path = os.path.join(opt.output_dir, f"{opt.output_filename}.madx")
    opt.change_params_correct_path = os.path.join(opt.output_dir,
                                                  f"{opt.output_filename}_correct.madx")
    opt.knob_path = os.path.join(opt.output_dir, f"{opt.output_filename}.tfs")
    return opt


def _get_method_opt(opt):
    """ Slightly unnecessary function to separate method-options
    for easier debugging and readability """
    return opt.get_subdict(["svd_cut", "n_correctors"])


def _print_rms(meas, diff_w, r_delta_w):
    """ Prints current RMS status """
    f_str = "{:>20s} : {:.5e}"
    LOG.debug("RMS Measure - Model (before correction, w/o weigths):")
    for key in meas:
        LOG.debug(f_str.format(key, _rms(meas[key].loc[:, DIFF].values)))

    LOG.info("RMS Measure - Model (before correction, w/ weigths):")
    for key in meas:
        LOG.info(f_str.format(
            key, _rms(meas[key].loc[:, DIFF].values * meas[key].loc[:, WEIGHT].values)))

    LOG.info(f_str.format("All", _rms(diff_w)))
    LOG.debug(f_str.format("R * delta", _rms(r_delta_w)))
    LOG.debug("(Measure - Model) - (R * delta)   ")
    LOG.debug(f_str.format("", _rms(diff_w - r_delta_w)))


def _load_fullresponse(full_response_path, variables):
    """
    Full response is dictionary of optics-parameter gradients upon
    a change of a single quadrupole strength
    """
    LOG.debug("Starting loading Full Response optics")
    with open(full_response_path, "rb") as full_response_file:
        full_response_data = pickle.load(full_response_file)
    loaded_vars = [var for resp in full_response_data.values() for var in resp]
    if not any([v in loaded_vars for v in variables]):
        raise ValueError("None of the given variables found in response matrix. "
                         "Are you using the right categories?")

    LOG.debug("Loading ended")
    return full_response_data


def _get_measurment_data(keys, meas_dir, beta_file_name, w_dict):
    """ Retruns a dictionary full of get_llm data """
    measurement = {}
    filtered_keys = [k for k in keys if w_dict[k] != 0]
    for key in filtered_keys:
        if key.startswith('MU'):
            measurement[key] = read_meas(meas_dir, f"{PHASE_NAME}{key[-1].lower()}{EXT}")
        elif key.startswith('D'):
            measurement[key] = read_meas(meas_dir, f"{DISPERSION_NAME}{key[-1].lower()}{EXT}")
        elif key == "NDX":
            measurement[key] = read_meas(meas_dir, f"{NORM_DISP_NAME}{key[-1].lower()}{EXT}")
        elif key in ('F1001R', 'F1001I', 'F1010R', 'F1010I'):
            pass
        elif key == "Q":
            measurement[key] = pd.DataFrame({
                # Just fractional tunes:
                VALUE: np.remainder([read_meas(meas_dir, f"{PHASE_NAME}x{EXT}")['Q1'],
                                     read_meas(meas_dir, f"{PHASE_NAME}x{EXT}")['Q2']], [1, 1]),
                # TODO measured errors not in the file
                ERROR: np.array([0.001, 0.001])
            }, index=['Q1', 'Q2'])
        elif key.startswith('BET'):
            measurement[key] = read_meas(meas_dir, f"{beta_file_name}{key[-1].lower()}{EXT}")
    return filtered_keys, measurement


def read_meas(meas_dir, filename):
    return tfs.read(os.path.join(meas_dir, filename), index="NAME")


def _get_varlist(accel_cls, variables, virt_flag):  # TODO: Virtual?
    varlist = np.array(accel_cls.get_variables(classes=variables))
    if len(varlist) == 0:
        raise ValueError("No variables found! Make sure your categories are valid!")
    return varlist


def _maybe_add_coupling_to_model(model, keys):
    if any([key for key in keys if key.startswith("F1")]):
        tw_opt = TwissOptics(model)
        couple = tw_opt.get_coupling(method="cmatrix")
        model["F1001R"] = couple["F1001"].apply(np.real).astype(np.float64)
        model["F1001I"] = couple["F1001"].apply(np.imag).astype(np.float64)
        model["F1010R"] = couple["F1010"].apply(np.real).astype(np.float64)
        model["F1010I"] = couple["F1010"].apply(np.imag).astype(np.float64)
    return model


# Parameter filtering ########################################################


def _filter_measurement(keys, meas, model, errorbar, w_dict, e_dict, m_dict):
    """ Filters measurements and renames columns to VALUE, ERROR, WEIGHT"""
    filters = _get_measurement_filters()
    new = dict.fromkeys(keys)
    for key in keys:
        new[key] = filters[key](key, meas[key], model, errorbar, w_dict[key],
                                modelcut=m_dict[key], errorcut=e_dict[key])
    return new


def _get_filtered_generic(key, meas, model, erwg, weight, modelcut, errorcut):
    common_bpms = meas.index.intersection(model.index)
    meas = meas.loc[common_bpms, :]
    new = tfs.TfsDataFrame(index=common_bpms)
    col = key if "MU" not in key else f"PHASE{key[-1]}"
    new[VALUE] = meas.loc[:, col].values
    new[ERROR] = meas.loc[:, f"{ERR}{col}"].values
    new[WEIGHT] = _get_errorbased_weights(key, weight, meas.loc[:, f"{ERR}{DELTA}{col}"]) if erwg else weight
    # filter cuts
    error_filter = meas.loc[:, f"{ERR}{DELTA}{col}"].values < errorcut
    model_filter = np.abs(meas.loc[:, f"{DELTA}{col}"].values) < modelcut
    if False:  # TODO automated model cut
        model_filter = get_smallest_data_mask(np.abs(meas.loc[:, f"{DELTA}{col}"].values),
                                              portion=0.95)
    if "MU" in key:
        new['NAME2'] = meas.loc[:, 'NAME2'].values
        second_bpm_in = np.in1d(new.loc[:, 'NAME2'].values, new.index.values)
        good_bpms = error_filter & model_filter & second_bpm_in
        good_bpms[-1] = False
    else:
        good_bpms = error_filter & model_filter
    LOG.debug(f"Number of BPMs with {key}: {np.sum(good_bpms)}")
    return new.loc[good_bpms, :]


def _get_tunes(key, meas, model, erwg, weight, modelcut, errorcut):
    meas[WEIGHT] = weight
    if erwg:
        meas[WEIGHT] = _get_errorbased_weights(key, meas[WEIGHT], meas[ERROR])
    LOG.debug(f"Number of tune measurements: {len(meas.index.values)}")
    return meas


def _get_errorbased_weights(key, weights, errors):
    # TODO case without errors used may corrupt the correction (typical error != 1)
    w2 = stats.weights_from_errors(errors)
    if w2 is None:
        LOG.warn(f"Weights will not be based on errors for '{key}'"
                 f", zeros of NaNs were found. Maybe don't use --errorbars.")
        return weights
    return weights * np.sqrt(w2)


# Response filtering ##########################################################


def _filter_response_index(response, measurement, keys):
    not_in_response = [k for k in keys if k not in response]
    if len(not_in_response) > 0:
        raise KeyError(f"The following optical parameters are not present in current"
                       f"response matrix: {not_in_response}")

    filters = _get_response_filters()
    new_resp = {}
    for key in keys:
        new_resp[key] = filters[key](response[key], measurement[key])
    return new_resp


def _get_generic_response(resp, meas):
    return resp.loc[meas.index.values, :]


def _get_phase_response(resp, meas):
    phase1 = resp.loc[meas.index.values, :]
    phase2 = resp.loc[meas.loc[:, 'NAME2'].values, :]
    return -phase1.sub(phase2.values)  # phs2-phs1 but with idx of phs1


def _get_tune_response(resp, meas):
    return resp


# Model appending #############################################################


def _append_model_to_measurement(model, measurement, keys):
    appenders = _get_model_appenders()
    meas = {}
    for key in keys:
        meas[key] = appenders[key](model, measurement[key], key)
    return meas


def _get_model_generic(model, meas, key):
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, key].values
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_phases(model, meas, key):
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = (model.loc[meas['NAME2'].values, key].values -
                       model.loc[meas.index.values, key].values)
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_betabeat(model, meas, key):
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, key].values
        meas[DIFF] = (meas.loc[:, VALUE].values - meas.loc[:, MODEL].values) / meas.loc[:, MODEL].values
    return meas


def _get_model_norm_disp(model, meas, key):
    col = key[1:]
    beta = f"BET{key[-1]}"
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, col].values / np.sqrt(model.loc[meas.index.values, beta].values)
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_tunes(model, meas, key):
    # We want just fractional tunes
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = np.remainder([model['Q1'], model['Q2']], [1, 1])
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


# Main Calculation #############################################################


def _calculate_delta(resp_matrix, meas_dict, keys, vars_list, method, meth_opt):
    """ Get the deltas for the variables.

    Output is Dataframe with one column 'DELTA' and vars_list index. """
    weight_vector = _join_columns('WEIGHT', meas_dict, keys)
    diff_vector = _join_columns('DIFF', meas_dict, keys)

    resp_weighted = resp_matrix.mul(weight_vector, axis="index")
    diff_weighted = diff_vector * weight_vector

    delta = _get_method_fun(method)(resp_weighted, diff_weighted, meth_opt)
    delta = tfs.TfsDataFrame(delta, index=vars_list, columns=[DELTA])

    # check calculations
    update = np.dot(resp_weighted, delta[DELTA])
    _print_rms(meas_dict, diff_weighted, update)

    return delta


def _get_method_fun(method):
    funcs = {"pinv": _pseudo_inverse, "omp": _orthogonal_matching_pursuit,}
    return funcs[method]


def _pseudo_inverse(response_mat, diff_vec, opt):
    """ Calculates the pseudo-inverse of the response via svd. (numpy) """
    if opt.svd_cut is None:
        raise ValueError("svd_cut setting needed for pseudo inverse method.")

    return np.dot(np.linalg.pinv(response_mat, opt.svd_cut), diff_vec)


def _orthogonal_matching_pursuit(response_mat, diff_vec, opt):
    """ Calculated n_correctors via orthogonal matching pursuit"""
    if opt.n_correctors is None:
        raise ValueError("n_correctors setting needed for orthogonal matching pursuit.")

    # return orthogonal_mp(response_mat, diff_vec, opt.n_correctors)
    res = OrthogonalMatchingPursuit(opt.n_correctors).fit(response_mat, diff_vec)
    coef = res.coef_
    LOG.debug(f"Orthogonal Matching Pursuit Results: \n"
              f"  Chosen variables: {response_mat.columns.values[coef.nonzero()]}\n"
              f"  Score: {res.score(response_mat, diff_vec)}")
    return coef


# MADX related #################################################################


def _create_corrected_model(twiss_out, change_params, accel_inst):
    """ Use the calculated deltas in changeparameters.madx to create a corrected model """
    madx_script = accel_inst.get_update_correction_job(twiss_out, change_params)
    madx_wrapper.resolve_and_run_string(madx_script, log_file=os.devnull,)


def write_knob(knob_path, delta):
    a = datetime.datetime.fromtimestamp(time.time())
    delta_out = - delta.loc[:, [DELTA]]
    delta_out.headers["PATH"] = os.path.dirname(knob_path)
    delta_out.headers["DATE"] = str(a.ctime())
    tfs.write(knob_path, delta_out, save_index="NAME")


def writeparams(path_to_file, delta):
    with open(path_to_file, "w") as madx_script:
        for var in delta.index.values:
            value = delta.loc[var, DELTA]
            madx_script.write(f"{var} = {var} {value:+e};\n")


# Small Helpers ################################################################


def _rms(a):
    return np.sqrt(np.mean(np.square(a)))


def _join_responses(resp, keys, varslist):
    """ Returns matrix #BPMs * #Parameters x #variables """
    return pd.concat([resp[k] for k in keys],  # dataframes
                     axis="index",  # axis to join along
                     join_axes=[pd.Index(varslist)]
                     # other axes to use (pd Index obj required)
                     ).fillna(0.0)


def _join_columns(col, meas, keys):
    """ Retuns vector: N= #BPMs * #Parameters (BBX, MUX etc.) """
    return np.concatenate([meas[key].loc[:, col].values for key in keys], axis=0)


def _filter_by_strength(delta, resp_matrix, min_strength=0):
    """ Remove too small correctors """
    delta = delta.loc[delta[DELTA].abs() > min_strength]
    return delta, resp_matrix.loc[:, delta.index], delta.index.values


def get_smallest_data_mask(data, portion=0.95):
    if not 0 <= portion <= 1:
        raise ValueError("Portion of data has to be between 0 and 1")
    b = int(len(data) * portion)
    mask = np.ones_like(data, dtype=bool)
    mask[np.argpartition(data, b)[b:]] = False
    return mask

if __name__ == "__main__":
    with logging_tools.DebugMode(active=True, log_file="iterative_correction.log"):
        global_correction()
