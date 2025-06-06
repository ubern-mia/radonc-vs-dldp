import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

from src.scores import generate_results

font = {"size": 12}
matplotlib.rc("font", **font)


if __name__ == "__main__":

    file_path = os.path.dirname(__file__)
    path_to_data = os.path.join(file_path, "data/radonc-vs-dldp-data.zip")
    zf = zipfile.ZipFile(path_to_data)

    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)

        expert_path = os.path.join(tempdir, "radonc-vs-dldp-data", "expert")
        data_path = os.path.join(tempdir, "radonc-vs-dldp-data", "experiments")

        jw_results = pd.read_csv(os.path.join(expert_path, "jw_impact.csv"))
        jw_results = jw_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

        ee_results = pd.read_csv(os.path.join(expert_path, "ee_impact.csv"))
        ee_results = ee_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

        er_results = pd.read_csv(os.path.join(expert_path, "er_impact.csv"))
        er_results = er_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

        list_oar_names = [
            "BrainStem",
            "Chiasm",
            "Cochlea_L",
            "Cochlea_R",
            "Eye_L",
            "Eye_R",
            "Hippocampus_L",
            "Hippocampus_R",
            "LacrimalGland_L",
            "LacrimalGland_R",
            "OpticNerve_L",
            "OpticNerve_R",
            "Pituitary",
        ]

        cases = [70, 71, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 85, 86]
        constraint_type = "max"
        n_oar_gt = 1
        thresh_gt = 0.1
        input_path = os.path.join(data_path, "reference")
        reference_results = generate_results(
            input_path, cases, list_oar_names, constraint_type, thresh_gt, n_oar_gt
        )

        alpha = 0.1
        n_oar_pred = 3
        thresh_pred = alpha * thresh_gt
        input_path = os.path.join(data_path, "predicted")
        predicted_results = generate_results(
            input_path, cases, list_oar_names, constraint_type, thresh_pred, n_oar_pred
        )

        data_dict = {
            "Prediction": predicted_results,
            "R1": jw_results,
            "R2": ee_results,
            "R3": er_results,
        }
        cat_names = ["Better", "No Change", "Worse"]
        for key in data_dict.keys():
            conf_mat = confusion_matrix(
                list(reference_results.iloc[:, 2]),
                list(data_dict[key].iloc[:, 2]),
                normalize="true",
            )
            print(f"Confusion matrix for (Reference vs {key}) is: \n{conf_mat}")
            # plt.figure(figsize=(24, 24))
            # cm = ConfusionMatrixDisplay(conf_mat, display_labels=cat_names)
            # cm.plot()
            # plt.title(f"Reference vs {key}")
            # plt.show()
            df_cm = pd.DataFrame(conf_mat, index=cat_names, columns=cat_names)
            # cm = ConfusionMatrixDisplay(conf_mat, display_labels=cat_names)
            # cm.plot()
            off_diag_mask = np.eye(*conf_mat.shape, dtype=bool)

            vmin = np.min(conf_mat)
            vmax = np.max(conf_mat)
            fig = plt.figure()
            sns.heatmap(
                df_cm,
                annot=True,
                mask=~off_diag_mask,
                cmap="Greens",
                vmin=vmin,
                vmax=vmax,
            )
            sns.heatmap(
                df_cm,
                annot=True,
                mask=off_diag_mask,
                cmap="OrRd",
                vmin=vmin,
                vmax=vmax,
                cbar_kws=dict(ticks=[]),
            )
            plt.title(f"Reference vs {key}")
            plt.savefig(f"figures/confusion_matrix_{key}.png")
            plt.close()
