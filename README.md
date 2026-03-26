ECG_projectN

Matthew Miller
Init 3/25/2026
Iteration 0.0 as of 3/25/2026

The project performs DL-based signal processing on ECG data from Physionet (ptb xl[1]). The initial analysis is unsupervised learning. Information from unsupervised learning such as key features and structures are used in supervised learning to implement classification, analysis, etc.

Currently ECG is a github repository. For now, a pull request will suffice.

Change settings in 'settings.toml', which should be in the same directory as main.py. Run main.py to execute the project.

Where N is the iteration number:

ECG_projectN/
|
|- README.md
|- docs/
|
|-settings.toml
|-main.py
|-ecg_lib/
| |-train.py
| |-dataset.py
| |-utils.py
| |-models/
|   |-models.py (template)
|   |-m_classifier_0.py (example)


Bib:
1. 
ptb xl
https://physionet.org/content/ptb-xl/1.0.3/

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., and Schaeffter, T. (2022) 'PTB-XL, a large publicly available electrocardiography dataset' (version 1.0.3), PhysioNet. RRID:SCR_007345. Available at: https://doi.org/10.13026/kfzx-aw45




