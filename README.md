ECG_projectN

Matthew Miller
Init 3/25/2026
Iteration 0.1 4/26/26
0.0 as of 3/25/2026


1. The project performs DL-based signal processing on ECG data from Physionet (ptb xl[1]). The initial analysis is supervised learning. Unsupervised learning will be done next. Information from unsupervised learning such as key features and structures are used in supervised learning to implement classification, analysis, etc.

Currently ECG is a github repository. For now, download, check directories, and run main/dirtymain.py

Change settings in 'settings.toml', which should be in the same directory as main.py. Run main.py to execute the project.

Ensure root directory is ECG_project0 not just ECG

2. Key files and features (where N is the iteration number):

ECG_projectN/
|
|- README.md
|- docs/
|
|-settings.toml
|-main.py
|-ecg_lib/
| |-models/
|   |-models.py (template)
|   |-m_classifier_0.py (example)
|	|-m_conv_1.py

3. roadmap is in todo.txt in docs/

Bib:
1. 
ptb xl
https://physionet.org/content/ptb-xl/1.0.3/

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., and Schaeffter, T. (2022) 'PTB-XL, a large publicly available electrocardiography dataset' (version 1.0.3), PhysioNet. RRID:SCR_007345. Available at: https://doi.org/10.13026/kfzx-aw45




