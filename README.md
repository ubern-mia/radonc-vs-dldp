# Comparing the Performance of Radiation Oncologists versus a Deep Learning

![MIDL 2024](https://img.shields.io/badge/Conference-MIDL%202024-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Testing](https://github.com/amithjkamath/radonc-vs-dldp/actions/workflows/test.yaml/badge.svg) ![License](https://img.shields.io/github/license/amithjkamath/radonc-vs-dldp)


This repository contains code to reproduce the analysis from our recent MIDL 2024 paper:

**"Comparing the Performance of Radiation Oncologists versus a Deep Learning Dose Predictor to Estimate Dosimetric Impact of Segmentation Variations for Radiotherapy"**  
*Amith Kamath\*, Zahira Mercado\*, Robert Poel, Jonas Willmann, Ekin Ermis, Elena Riggenbach, Nicolaus Andratschke, Mauricio Reyes*  
(\*Contributed equally)

Read the paper [here](https://openreview.net/pdf/5f8cbcc7c1bba1e30813f02448e4d7c8be57c3b2.pdf).  

Click here for the video recording of the oral talk:

[<img src="https://i.ytimg.com/vi/Co9yUIAw6H0/maxresdefault.jpg" width="50%">](https://youtu.be/Co9yUIAw6H0?t=3587 "Comparing the Performance of Radiation Oncologists versus a Deep Learning Dose Predictor")

🔗 [Project Website](https://amithjkamath.github.io/projects/2024-midl-radonc-vs-dldp/)

---

## Overview

Geometric metrics like Dice score often fail to capture the clinical impact of segmentation errors in radiotherapy. This project introduces a quality assurance (QA) framework using a deep learning-based dose prediction model to assess the **dosimetric impact** of segmentation variations in glioblastoma treatment planning.

![figure-one.png](images/figure-one.png)

The model's performance is compared with three experienced radiation oncologists on a test set of 54 tumor segmentation variants.

---

## Key Contributions

- Evaluation of a Dose Prediction model’s ability to classify segmentation variants as:
  - **Sub-optimal**
  - **No Impact**
  - **Improved**
- Comparison with radiation oncologists in terms of precision, recall, and evaluation time.
- Analysis of sensitivity to dosimetric thresholds (α) and number of affected OARs (nOAR).

---

## Setup

### Installation

```bash
git clone https://github.com/amithjkamath/radonc-vs-dldp.git
cd radonc-vs-dldp
uv venv .venv
uv pip install -r pyproject.toml
```

Run code/generate-figure-3.py and code/generate-figure-4.py to reproduce the analysis. Data required to generate figures 3 and 4 from the paper is included in radonc-vs-dldp-data.zip.

If this is useful in your research, please consider citing:

    @inproceedings{Kamath2024comparing,
      title={Comparing the Performance of Radiation Oncologists versus a Deep Learning Dose Predictor to Estimate Dosimetric Impact of Segmentation Variations for Radiotherapy},
      author={Kamath, Amith Jagannath and der Maur, Zahira Mercado Auf and Poel, Robert and Willmann, Jonas and Ermis, Ekin and Riggenbach, Elena and Andratschke, Nicolaus and Reyes, Mauricio},
      booktitle={Medical Imaging with Deep Learning},
      year={2024},
      volume={250},
      url={https://proceedings.mlr.press/v250/kamath24a.html}
    }
