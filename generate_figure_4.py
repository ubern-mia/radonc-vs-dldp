import os
import zipfile
import tempfile
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import matplotlib

from src.scores import generate_results

font = {"size": 20}
matplotlib.rc("font", **font)


if __name__ == "__main__":

    file_path = os.path.dirname(__file__)
    path_to_data = os.path.join(file_path, "data/radonc-vs-dldp-data.zip")
    zf = zipfile.ZipFile(path_to_data)

    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)

        expert_path = os.path.join(tempdir, "radonc-vs-dldp-data", "expert")
        data_path = os.path.join(tempdir, "radonc-vs-dldp-data", "experiments")
        reference_path = os.path.join(data_path, "reference")
        prediction_path = os.path.join(data_path, "predicted")

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
        type = "max"
        n_oar_gt = 1
        thresh_gt = 0.1
        input_path = os.path.join(data_path, "reference")
        reference_results = generate_results(
            input_path, cases, list_oar_names, type, thresh_gt, n_oar_gt
        )

        precision_map = {"Pred": {}, "R1": {}, "R2": {}, "R3": {}}
        recall_map = {"Pred": {}, "R1": {}, "R2": {}, "R3": {}}
        f1_map = {"Pred": {}, "R1": {}, "R2": {}, "R3": {}}
        for alpha in [
            0.005,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
            0.11,
            0.12,
            0.13,
            0.14,
            0.15,
        ]:
            for n_oar_pred in range(9, 0, -1):
                thresh_pred = alpha * thresh_gt
                predicted_path = os.path.join(data_path, "predicted")
                predicted_results = generate_results(
                    predicted_path, cases, list_oar_names, type, thresh_pred, n_oar_pred
                )

                data_dict = {
                    "Pred": predicted_results,
                    "R1": jw_results,
                    "R2": ee_results,
                    "R3": er_results,
                }
                cat_names = ["Better", "No Change", "Worse"]
                for key in data_dict.keys():
                    if alpha not in precision_map[key].keys():
                        precision_map[key][alpha] = {}
                    precision_map[key][alpha][n_oar_pred] = precision_score(
                        list(reference_results.iloc[:, 2]),
                        list(data_dict[key].iloc[:, 2]),
                        average="weighted",
                    )
                    if alpha not in recall_map[key].keys():
                        recall_map[key][alpha] = {}
                    recall_map[key][alpha][n_oar_pred] = recall_score(
                        list(reference_results.iloc[:, 2]),
                        list(data_dict[key].iloc[:, 2]),
                        average="weighted",
                    )
                    if alpha not in f1_map[key].keys():
                        f1_map[key][alpha] = {}
                    f1_map[key][alpha][n_oar_pred] = f1_score(
                        list(reference_results.iloc[:, 2]),
                        list(data_dict[key].iloc[:, 2]),
                        average="weighted",
                    )
        precision_df = pd.DataFrame.from_dict(precision_map["Pred"])
        precision_array = precision_df.to_numpy()
        plt.figure(figsize=(16, 12))
        ax = sns.heatmap(
            precision_array,
            xticklabels=precision_df.columns,
            yticklabels=precision_df.index,
            cmap="viridis",
            annot=True,
            square=True,
            cbar=False,
        )
        plt.title(r"Variation of Precision (to $\alpha$ and nOAR)")
        plt.gca().set_xlabel(r"$\alpha$")
        plt.ylabel("nOAR")
        plt.savefig("figures/precision_heatmap.png")
        plt.close()

        recall_df = pd.DataFrame.from_dict(recall_map["Pred"])
        recall_array = recall_df.to_numpy()
        plt.figure(figsize=(16, 12))
        ax = sns.heatmap(
            recall_array,
            xticklabels=recall_df.columns,
            yticklabels=recall_df.index,
            cmap="viridis",
            annot=True,
            square=True,
            vmin=0.3,
            vmax=0.75,
            cbar=False,
        )
        plt.title(r"Variation of Recall (to $\alpha$ and nOAR)")
        plt.gca().set_xlabel(r"$\alpha$")
        plt.ylabel("nOAR")
        plt.savefig("figures/recall_heatmap.png")
        plt.close()

        f1_df = pd.DataFrame.from_dict(f1_map["Pred"])
        f1_array = f1_df.to_numpy()
        plt.figure(figsize=(16, 12))
        ax = sns.heatmap(
            f1_array,
            xticklabels=f1_df.columns,
            yticklabels=f1_df.index,
            cmap="viridis",
            annot=True,
            square=True,
            vmin=0.3,
            vmax=0.75,
            cbar=False,
        )
        plt.title(r"Variation of F1 score (to $\alpha$ and nOAR)")
        plt.gca().set_xlabel(r"$\alpha$")
        plt.ylabel("nOAR")
        plt.savefig("figures/f1_heatmap.png")
        plt.close()
