import os

from tfs.collection import TfsCollection, Tfs
from optics_measurements.constants import (AMP_BETA_NAME, BETA_NAME, CHROM_BETA_NAME, PHASE_NAME,
                                           TOTAL_PHASE_NAME, DISPERSION_NAME, NORM_DISP_NAME,
                                           ORBIT_NAME, KICK_NAME, IP_NAME, EXT)



class OpticsMeasurement(TfsCollection):
    """Class to hold and load the optics measurements results.

    The class will the files, if none of them if present an IOError will be raised.
    """
    beta = Tfs(BETA_NAME)
    amp_beta = Tfs(AMP_BETA_NAME)
    kmod_beta = Tfs("kmodbeta")
    kmod_betastar = Tfs("getkmodbetastar", two_planes=False)
    phase = Tfs(PHASE_NAME)
    phasetot = Tfs(TOTAL_PHASE_NAME)
    disp = Tfs(DISPERSION_NAME)
    orbit = Tfs(ORBIT_NAME)
    coupling = Tfs("getcouple", two_planes=False)
    norm_disp = Tfs(f"{NORM_DISP_NAME}x", two_planes=False)

    def get_filename(self, prefix, plane=""):
        filename = f"{prefix}{plane}{EXT}"
        if os.path.isfile(os.path.join(self.directory, filename)):
            return filename
        raise IOError(f"No file name found for prefix {prefix} in {self.directory}.")

    def write_to(self, value, prefix, plane=""):
        data_frame, suffix = value
        filename = f"{prefix}{plane}{suffix}{EXT}"
        return filename, data_frame
