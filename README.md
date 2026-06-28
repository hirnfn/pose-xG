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

1. **Camera Calibration & Triangulation**  
   Use the calibration and triangulation scripts to reconstruct 3D joint positions from multi-view pose data.

2. **Feature Extraction**  
   Run `07_Extract_Features_from_Pose.py` to compute pose-derived biomechanical features (e.g., joint angles, velocities, and motion descriptors) for each shot.

3. **Model Training**  
   - `08_Base_Model.ipynb` trains a standard expected-goals (xG) model based on shot and contextual features.  
   - `09_Bio_Model.ipynb` trains the biomechanically enhanced xG model incorporating pose-based features.

4. **Evaluation & Analysis**  
   Evaluate and compare the base and biomechanical models on the provided test sets, compute performance metrics, and analyze the influence of biomechanical features on model predictions.

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

**[Your Name]**  
**[Your Institution]**  
**[Your Email Address]**

---

## 📜 Citation

If you use this code or parts of it in your research, please cite the associated paper:

```bibtex
@article{posexg2026,
  title   = {Your Paper Title},
  author  = {Author list},
  journal = {npj Artificial Intelligence},
  year    = {2026}
}
