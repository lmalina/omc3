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

from correction import handler
from utils import iotools, logging_tools
from generic_parser.entrypoint import entrypoint, EntryPointParameters
LOG = logging_tools.get_logger(__name__)


def correction_params():
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


@entrypoint(correction_params())
def global_correction_entrypoint(opt, accel_opt):
    """ Do the global correction. Iteratively.
        # TODO auto-generate docstring
    """
    LOG.info("Starting Iterative Global Correction.")
    # check on opt
    opt = _check_opt_add_dicts(opt)
    opt = _add_hardcoded_paths(opt)
    iotools.create_dirs(opt.output_dir)
    handler.correct(accel_opt, opt)


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


if __name__ == "__main__":
    with logging_tools.DebugMode(active=True, log_file="iterative_correction.log"):
        global_correction_entrypoint()
