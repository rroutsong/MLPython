#!/usr/bin/env python

"""
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
    
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
    from Fisher's paper. Note that it's the same as in R, but not as in the UCI
    Machine Learning Repository, which has two wrong data points.
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    .. topic:: References
    
       - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...

    data structure:
        flower 0
            sepal length in cm - iris['data'][0][0] - 5.1
            sepal width in cm - iris['data'][0][1] - 3.5
            petal length in cm - iris['data'][0][2] - 1.4
            petal width in cm - iris['data'][0][3] - 0.2
            class - iris['target'][0] - 0
            class label - iris['target_names'][iris['target'][0]] - 'setosa'
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def scatter(xdata, ydata):
    from matplotlib.colors import ListedColormap
    cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
    pd.plotting.scatter_matrix(xdata, c=ydata, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8, cmap=cm3)


def main():
    iris = load_iris()
    Xtr, Xte, ytr, yte = train_test_split(iris['data'], iris['target'], random_state=4488)

    # Visualize iris dataset, using pandas scatter matrix
    # plt.show(scatter(pd.DataFrame(Xtr, columns=iris['feature_names']), ytr))

    # build the KNN model
    knn = KNeighborsClassifier(n_neighbors=1)
    print(knn.fit(Xtr, ytr))

    # predict using model
    X_unseen = np.array([[5, 2.9, 1, 0.2]])
    print(f"Data unseen to model :{X_unseen}")
    Y_unseen = knn.predict(X_unseen)
    print(f"Unseen prediction: {iris['target_names'][Y_unseen]}, raw prediction: {Y_unseen}")
    print("\n")
    print("Test set predictions:\n")
    yte_predict = knn.predict(Xte)
    print(yte_predict)
    print(f'\nTest set score: {np.mean(yte_predict == yte)}')


if __name__ == '__main__':
    main()
