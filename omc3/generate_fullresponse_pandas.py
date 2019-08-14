"""
Provides a response generation wrapper.
The response matrices can be either created by response_madx or analytically via TwissResponse.

:author: Joschua Dillys
"""
import os
import pickle
from model import manager
from utils import logging_tools
from generic_parser.entrypoint import EntryPointParameters, entrypoint
from correction.fullresponse import response_madx, response_twiss
from global_correct_iterative import CORRECTION_DEFAULTS

LOG = logging_tools.get_logger(__name__)

DEFAULT_PATTERNS = {
    "job_content": "%JOB_CONTENT%",  # used in lhc_model_creator, sequence_evaluation
    "twiss_columns": "%TWISS_COLUMNS%",  # used in lhc_model_creator, sequence_evaluation
    "element_pattern": "%ELEMENT_PATTERN%",  # used in lhc_model_creator, sequence_evaluation
}


def get_params():
    params = EntryPointParameters()
    params.add_parameter(flags="--creator", name="creator", type=str, choices=("madx", "twiss"),
                         default="madx", help="Create either with madx or analytically from twiss file.")
    params.add_parameter(flags="--variables", name="variable_categories", nargs="+",
                         default=CORRECTION_DEFAULTS["variables"], help="List of the variables classes to use.")
    params.add_parameter(flags=["--model_dir"], help="Path to the model directory.", name="model_dir", required=True, type=str)
    params.add_parameter(flags="--optics_file", name="modifiers",
                         help=("Path to the optics file to use. If not present will default to model_path/modifiers.madx, if such a file exists."), )
    params.add_parameter(flags=["--outfile"], help="Name of fullresponse file.", name="outfile_path", required=True, type=str)
    params.add_parameter(flags=["--deltak"], help="Delta K1L to be applied to quads for sensitivity matrix (madx-only).", default=0.00002, name="delta_k", type=float)
    params.add_parameter(flags="--optics_params", help="List of parameters to correct upon (e.g. BBX BBY; twiss-only).", name="optics_params", type=str, nargs="+",)
    params.add_parameter(flags="--debug", help="Print debug information.", name="debug", action="store_true",)
    return params


@entrypoint(get_params())
def create_response(opt, other_opt):
    """ Entry point for creating pandas-based response matrices.

    The response matrices can be either created by response_madx or TwissResponse.

    Keyword Args:
        Required
        model_dir (str): Path to the model directory.
                         **Flags**: ['-m', '--model_dir']
        outfile_path (str): Name of fullresponse file.
                            **Flags**: ['-o', '--outfile']
        Optional
        creator (str): Create either with madx or analytically from twiss file.
                       **Flags**: --creator
                       **Choices**: ('madx', 'twiss')
                       **Default**: ``madx``
        debug: Print debug information.
               **Flags**: --debug
               **Action**: ``store_true``
        delta_k (float): Delta K1L to be applied to quads for sensitivity matrix (madx-only).
                         **Flags**: ['-k', '--deltak']
                         **Default**: ``2e-05``
        optics_params (str): List of parameters to correct upon (e.g. BBX BBY; twiss-only).
                             **Flags**: --optics_params
        variable_categories: List of the variables classes to use.
                             **Flags**: --variables
                             **Default**: ``['MQM', 'MQT', 'MQTL', 'MQY']``
    """
    with logging_tools.DebugMode(active=opt.debug,
                                 log_file=os.path.join(opt.model_dir, "generate_fullresponse.log")):
        LOG.info("Creating response.")
        accel_cls = manager.get_accel_class(other_opt)
        accel_inst = accel_cls(model_dir=opt.model_dir)
        if opt.modifiers is not None:
            accel_inst.modifiers = opt.modifiers

        if opt.creator == "madx":
            fullresponse = response_madx.generate_fullresponse(
                accel_inst, opt.variable_categories, delta_k=opt.delta_k
            )

        elif opt.creator == "twiss":
            fullresponse = response_twiss.create_response(
                accel_inst, opt.variable_categories, opt.optics_params
            )

        LOG.debug(f"Saving Response into file '{opt.outfile_path}'")
        with open(opt.outfile_path, 'wb') as dump_file:
            pickle.dump(fullresponse, dump_file)#, protocol=-1)


if __name__ == "__main__":
    create_response()
