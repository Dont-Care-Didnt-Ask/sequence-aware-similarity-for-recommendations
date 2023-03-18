# Sequence-aware similarity for recommender systems

__Final project for Skoltech Intro to Recommender Systems 2023 course.__

__Team:__
- Seleznyov Mikhail
- Yugay Aleksandr
- Fadeev Egor
- Lukyanov Matvey

# Idea

In recommender systems, the order in which the user interacts with items might be important.
However, approaches like KNN, PureSVD and ScaledSVD do not take the order into account.
One of the easiest ways to incorporate positional information is to use sequence-aware similarity measure -- for example, Weighted Jaccard Index.
So, in this project we want to augment popular baselines with this simple modification and check, if it is beneficial.
