# Pose-xG

**Pose-xG** is a Python-based framework for constructing a **biomechanically informed, skill-adjusted expected goals (xG) model for football (soccer)**. The approach incorporates pose-estimation–derived features from a player’s shooting motion to enhance traditional xG modeling.

This repository contains scripts and notebooks used for data preparation, pose-based feature extraction, model training, and evaluation as part of the associated research project.

---

## 📘 Overview

Traditional xG models primarily rely on shot location and contextual variables. Pose-xG extends this paradigm by integrating biomechanical features extracted from player shooting motion, enabling a more direct representation of individual finishing skill.

The repository includes:
- Camera calibration and 3D triangulation utilities
- Pose-based feature extraction
- Training and evaluation of baseline and biomechanical xG models
- Example training and test datasets

---

## 📁 Repository Structure
pose-xG/
├── data/
│ ├── Trainingsset_Base.xlsx
│ ├── Trainingsset_Bio.xlsx
│ ├── Testset_Base.xlsx
│ └── Testset_Bio.xlsx
├── scripts/
│ ├── 04_Camera_Calibration_and_Triangulation.py
│ ├── 06_Triangulation.py
│ ├── 07_Extract_Features_from_Pose.py
│ ├── 08_Base_Model.ipynb
│ └── 09_Bio_Model.ipynb
└── README.md
└── requirements.txt


- **data/** – Example training and test datasets for baseline and biomechanical xG models  
- **scripts/** – Python scripts and Jupyter notebooks for calibration, triangulation, feature extraction, and modeling

---

## 🚀 Getting Started

### Requirements

The code is written in **Python 3**. Typical dependencies include:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- pose-estimation frameworks (e.g., MediaPipe or OpenPose, depending on the preprocessing pipeline)

It is recommended to create a virtual environment and install dependencies accordingly.

---

## 🧠 Workflow

1. **Video Alignment**
   Use a video editing software to align the videos from both camera angles. We used DaVinci Resolve.

2. **Split calibration from remaining video**
   Cut the Videos into the calibration part and the remaining measurements part.

3. **Annotate the baseline features**
   Estimate the features not extractable from pose estimation, such as distance and angle to goal, with a annotion software. We used Kinovea. Additionally, we used this step to gather the timestamp of each shot.

4. **Camera Trinagulation**
   Run `04_Camera_Calibration.py` to first calibrate each camera and then stereo calibrate to get the rotation matrix and Translation vector.

5. **OpenPose**
   Estimate Human Pose Estimations from command line.

6. **Triangulation**
   Run `06_Triangulation.py` to reconstruct 3D joint positions from multi-view pose data.

7. **Feature Extraction**
   Run `07_Extract_Features_from_Pose.py` to compute pose-derived biomechanical features (e.g., joint velocities, joint angles, and motion descriptors) for each shot.

8. **Model Training & Evaluation**  
   - `08_Base_Model.ipynb` trains a standard expected-goals (xG) model based on shot and contextual features.  
   - `09_Bio_Model.ipynb` trains the biomechanically enhanced xG model incorporating pose-based features.
   - Evaluate and compare the base and biomechanical models on the provided test sets, compute performance metrics, and analyze the influence of biomechanical features on model predictions.

---

## 📊 Data

The example datasets included in this repository are provided for demonstration and reproducibility purposes. Depending on the study setup, access to raw pose or video data may be restricted due to privacy and consent constraints.

---

## 📄 License

This repository and its contents are provided under the **[LICENSE NAME]** (e.g., MIT License).  
See the `LICENSE` file for details.

---

## 📬 Contact

For questions, feedback, or requests related to this project, please contact:

**Fabian Hirn**  
**Machine Learning and Data Analytics Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg**  
**fabian.g.hirn@fau.de**

---

## 📜 Citation

If you use this code or parts of it in your research, please cite the associated paper:

```bibtex
@article{posexg2026,
  title   = {Pose-xG: Creating a skill-adjusted xG-model by incorporating biomechanical features based on the player's shooting motion},
  author  = {Fabian Hirn, Alexander Weiss, Rebecca Lennartz, Bjoern M. Eskofier, Anne D. Koelewijn},
  journal = {npj Artificial Intelligence},
  year    = {2026}
}
