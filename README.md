# Sequence-aware similarity for recommender systems

__Final project for Skoltech Intro to Recommender Systems 2023 course.__

__Team:__
- Seleznyov Mikhail
- Yugay Aleksandr
- Fadeev Egor
- Lukyanov Matvey

# Idea

In recommender systems, the order in which the user interacts with items might be important. However, approaches like KNN, PureSVD and ScaledSVD do not take the order into account. One of the easiest ways to incorporate positional information is to use sequence-aware similarity measure.
In this project we augment popular baselines with this simple
modification and show that it is beneficial.

# Setup

```
git clone git@github.com:Dont-Care-Didnt-Ask/sequence-aware-similarity-for-recommendations.git
cd sequence-aware-similarity-for-recommendations

conda create -n recsys python=3.9
conda activate recsys
conda install --file requirements.txt
pip install ilupp
```

NB: launching jupyter, make sure you use jupyter from this env!
You might need to launch it like `~/miniconda3/envs/recsys/bin/jupiter-lab`
