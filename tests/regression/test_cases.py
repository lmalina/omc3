from . import context
import os
import argparse
from os.path import join, abspath, dirname, pardir
import tests.regression import compare_utils, regression


ABS_ROOT = abspath(join(dirname(__file__), pardir, pardir))

REGR_DIR = join("tests", "regression")
TBTS = join("tests", "inputs", "tbt_files")
MODELS = join("tests", "inputs", "models")
OPTICS = join("tests", "inputs", "optics_files")
HARM_FILES = join("tests", "inputs", "harmonic_results")
GETLLM_FILES = join("tests", "inputs", "getllm_results")
TO_SOURCES = "omc3"

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keepfiles", help="Keep output files if test fails.", dest="keepfiles", action="store_true",)
    return parser.parse_args()


TEST_CASES_HOLE_IN_ONE = (
    regression.TestCase(
        name="hole_in_one_test_flat_3dkick",
        script=join(TO_SOURCES, "hole_in_one.py"),
        arguments=("--harpy --clean --file={file} --model={model} --output={output} clean "
                   "harpy --tunex 0.27 --tuney 0.322 --tunez 4.5e-4 "
                   "--nattunex 0.28 --nattuney 0.31 --tolerance 0.005".format(
            file=join(TBTS, "flat_beam1_3d.sdds"),
            model=join(MODELS, "flat_beam1", "twiss.dat"),
            output=join(REGR_DIR, "_out_hole_in_one_test_flat_3dkick"))),
        arguments2=("--file={file} --model={model} --output={output} clean "
                   "harpy --tunex 0.27 --tuney 0.322 --tunez 4.5e-4 "
                   "--nattunex 0.28 --nattuney 0.31 --tolerance 0.005".format(
            file=join(TBTS, "flat_beam1_3d.sdds"),
            model=join(MODELS, "flat_beam1", "twiss.dat"),
            output=join(REGR_DIR, "_out_hole_in_one_test_flat_3dkick"))),
        output=join(REGR_DIR, "_out_hole_in_one_test_flat_3dkick"),
        test_function=lambda d1, d2: compare_utils.compare_dirs(d1, d2, ignore=[r".*\.log"]),
        pre_hook=lambda dir: os.makedirs(join(dir, REGR_DIR, "_out_hole_in_one_test_flat_3dkick")),
    ),
)

TEST_CASES_MEASURE_OPTICS = (
    regression.TestCase(
        name="measure_optics_test_flat_disp",
        script=join(TO_SOURCES, "hole_in_one.py"),
        arguments=("--accel=LHCB1 "
                   "--model={model} "
                   "--files={files_dir}/on_mom_file1.sdds,{files_dir}/on_mom_file2.sdds,{files_dir}/neg_mom_file1.sdds,{files_dir}/pos_mom_file1.sdds "
                   "--output={output} "
                   )
            .format(model=join(MODELS, "flat_beam1", "twiss.dat"),
                    files_dir=join(HARM_FILES, "flat_60_15cm_b1"),
                    output=join(REGR_DIR, "_out_getllm_test_flat_disp"),
                    ),
        output=join(REGR_DIR, "_out_getllm_test_flat_disp"),
        test_function=lambda dir1, dir2:
        compare_utils.compare_dirs_ignore_words(dir1, dir2, ["Command", "Date", "CWD"]),
        pre_hook=lambda dir: None,
    ),
)

TEST_CASES_MODEL_CREATION = (
    regression.TestCase(
        name="model_creator_test_lhc",
        script=join(TO_SOURCES, "model_creator.py"),
        arguments=("--type nominal --accel lhc --lhcmode lhc_runII_2017 "
                   "--beam 1 --nattunex 0.28 --nattuney 0.31 --acd "
                   "--drvtunex 0.27 --drvtuney 0.322 --dpp 0.0 "
                   "--optics {optics} "
                   "--output {output}").format(optics=join(OPTICS, "2017", "opticsfile.19"),
                                               output=join(REGR_DIR,
                                                           "_out_model_creator_test_lhc")),
        output=join(REGR_DIR, "_out_model_creator_test_lhc"),
        test_function=lambda dir1, dir2:
        compare_utils.compare_dirs_ignore_words(dir1, dir2, ["ORIGIN", "DATE", "TIME"]),
        pre_hook=lambda dir: os.makedirs(join(dir, REGR_DIR, "_out_model_creator_test_lhc")),
    ),
)


def run_tests(opts=None):
    """Run the test cases and raise RegressionTestFailed on failure.
    """
    alltests = (
            list(TEST_CASES_HOLE_IN_ONE) +
            list(TEST_CASES_MEASURE_OPTICS) +
            list(TEST_CASES_MODEL_CREATION)
    )
    regression.launch_test_set(alltests, ABS_ROOT,
                               yaml_conf=join(ABS_ROOT, ".travis.yml"),
                               keep_fails=opts.keepfiles if opts else False)


if __name__ == "__main__":
    _options = _parse_args()
    run_tests(_options)
