import os
import pandas as pd
import numpy as np


def dose_score(_pred: np.ndarray, _gt: np.ndarray, _dose_mask=None) -> np.ndarray:
    """
    DOSE_SCORE:
        This is modified from https://github.com/ababier/open-kbp
    """
    if _dose_mask is not None:
        _pred = _pred[_dose_mask > 0]
        _gt = _gt[_dose_mask > 0]

    return np.mean(np.abs(_pred - _gt))


def dvh_score(
    _dose: np.ndarray, _mask: np.ndarray, mode: str, spacing=None
) -> dict[str, np.ndarray]:
    """
    DVH_SCORE:
        This is modified from https://github.com/ababier/open-kbp
    """
    output = {}

    if mode.lower() == "target":
        _roi_dose = _dose[_mask > 0]
        # D1
        output["D1"] = np.percentile(_roi_dose, 99)
        # D95
        output["D95"] = np.percentile(_roi_dose, 5)
        # D99
        output["D99"] = np.percentile(_roi_dose, 1)

    elif mode.upper() == "OAR":
        if spacing is None:
            raise Exception("dvh score computation requires voxel spacing information.")

        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)
        _voxel_size = np.prod(spacing)

        # D_0.1_cc
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        if fractional_volume_to_evaluate <= 0:
            output["D_0.1_cc"] = np.asarray(0.0)
        else:
            output["D_0.1_cc"] = np.percentile(_roi_dose, fractional_volume_to_evaluate)

        # Dmean
        output["mean"] = np.mean(_roi_dose)
    else:
        raise Exception("Unknown mode!")

    return output


def generate_results(input_path: str, cases: list, list_oar_names: list, constraint: str, thresh: float, n_oar: int):
    """
    GENERATE_RESULTS:
        Following algorithm 1 in the paper, to generate classifications.
    """
    results = []
    for case in cases:
        out = []
        if case == 75:
            out = [None, None, None, None]
        else:
            data = pd.read_csv(
                os.path.join(
                    input_path,
                    "ISAS_GBM_" + str(case).zfill(3) + "_" + constraint + ".csv",
                ),
                index_col=0,
            )

            percentage_diff = (
                data[["1", "2", "3", "4"]].div(data["0"], axis=0).loc[list_oar_names]
            )

            for idx in range(1, 5):
                if np.any(
                        np.sum(percentage_diff[[str(idx)]] > thresh, axis=0, )
                        >= n_oar
                ):
                    out.append("Worse")
                elif np.any(
                        np.sum(percentage_diff[[str(idx)]] < -thresh, axis=0, )
                        >= n_oar
                ):
                    out.append("Better")
                else:
                    out.append("No Change")
        out.insert(0, str(case))
        results.append(out)
    results_table = pd.DataFrame(results)
    results_table.iloc[7, 4] = None  # Handle case 78
    results_table.iloc[10, 4] = None  # Handle case 82
    results_table = results_table.loc[:, 1:].stack().reset_index()
    return results_table