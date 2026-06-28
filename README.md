# \# Pose-xG

# 

# \*\*Pose-xG\*\* is a Python framework for constructing a \*\*biomechanically-informed, skill-adjusted expected goals (xG) model for football (soccer)\*\* based on player shooting motion extracted via pose estimation. The repository includes data preparation, pose feature extraction, model training, evaluation and analysis scripts that were used in the associated research project.

# 

# \---

# 

# \## 📘 Overview

# 

# This project demonstrates how pose-based biomechanical features can be incorporated into expected-goals modeling to better account for player skill and shooting motion mechanics. Features are extracted from pose sequences and used to train both baseline and biomechanically-enhanced xG models.

# 

# \### Repository Structure

├── data/

│ ├── Trainingsset\_Base.xlsx

│ ├── Trainingsset\_Bio.xlsx

│ ├── Testset\_Base.xlsx

│ └── Testset\_Bio.xlsx

├── scripts/

│ ├── 04\_Camera\_Calibration\_and\_Triangulation.py

│ ├── 06\_Triangulation.py

│ ├── 07\_Extract\_Features\_from\_Pose.py

│ ├── 08\_Base\_Model.ipynb

│ └── 09\_Bio\_Model.ipynb

└── README.md

└─── requirements.txt





\- \*\*data/\*\* – Excel files containing training and test datasets for both the base and biomechanical models.  

\- \*\*scripts/\*\* – Python scripts and notebooks for camera calibration, 3D triangulation, feature extraction, and model training/analysis.



\---



\## 🚀 Getting Started



\### Dependencies



Before running any scripts, install required packages (example using `requirements.txt` — create your own based on the project):



```bash

pip install -r requirements.txt



\## 🧠 Workflow



1\. \*\*Video Alignment\*\*

&#x20;  Use a video editing software to align the videos from both camera angles. We used DaVinci Resolve.



2\. \*\*Split calibration from remaining video\*\*

&#x20;  Cut the Videos into the calibration part and the remaining measurements part.



3\. \*\*Annotate the baseline features\*\*

&#x20;  Estimate the features not extractable from pose estimation, such as distance and angle to goal, with a annotion software. We used Kinovea. Additionally, we used this step to gather the timestamp of each shot.



4\. \*\*Camera Calibration\*\*  

&#x20;  Run `04\_Camera\_Calibration.py` to first calibrate each camera and then stereo calibrate to get the rotation matrix and Translation vector.



5\. \*\*OpenPose\*\*

&#x20;  Estimate Human Pose Estimations from command line.



6\. \*\*Triangulation\*\*

&#x20;  Run `06\_Triangulation.py` to reconstruct 3D joint positions from multi-view pose data.



7\. \*\*Feature Extraction\*\*  

&#x20;  Run `07\_Extract\_Features\_from\_Pose.py` to compute pose-derived biomechanical features (e.g., joint velocities, joint angles, and motion descriptors) for each shot.



8\. \*\*Model Training \& Evaluation\*\*  

&#x20;  - `08\_Base\_Model.ipynb` trains a standard expected-goals (xG) model based on shot and contextual features.  

&#x20;  - `09\_Bio\_Model.ipynb` trains the biomechanically-enhanced xG model incorporating pose-based features.

&#x20;  - Evaluate and compare the base and biomechanical models on the provided test sets, compute performance metrics, and analyze the impact of biomechanical features on model predictions.



\---



\## 📄 License



This repository and its contents are provided under the \*\*\[LICENSE NAME]\*\* (e.g., MIT License). See the `LICENSE` file for details.



\---



\## 📬 Contact



For questions, feedback, or requests for additional resources, please contact:



\*\*Fabian Hirn\*\*  

\*\*Machine Learning and 

Data Analytics Lab,

Friedrich-Alexander-Universität Erlangen-Nürnberg\*\*  

\*\*fabian.g.hirn@fau.de\*\*



\---



\## 📜 Citation



If you use this code or parts of it in your research, please cite the associated paper:



```bibtex

@article{posexg2026,

&#x20; title   = {Pose-xG: Creating a skill-adjusted xG-model by incorporating biomechanical features based on the player's shooting motion},

&#x20; author  = {Fabian Hirn, Alexander Weiss, Rebecca Lennartz, Bjoern M. Eskofier, Anne D. Koelewijn},

&#x20; journal = {npj Artificial Intelligence},

&#x20; year    = {2026}

}



