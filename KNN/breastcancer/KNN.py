from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import mglearn.datasets
from mglearn.plots import plot_2d_separator
from mglearn import discrete_scatter
import numpy as np
import pandas as pd

"""
    ## classification dataset ##
    Breast cancer wisconsin (diagnostic) dataset
    --------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    .. topic:: References
    
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and
         prognosis via linear programming. Operations Research, 43(4), pages 570-577,
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994)
         163-171.
         
    Dataset contains 569 samples with 30 features.
    Each labeled benign or malignant
    
    ## regression dataset ##

    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 506
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.
    
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', 
       Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International
         Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
"""


def main():
    # categorical dataset
    cancer = load_breast_cancer()

    print('Categorical dataset')
    print('Example feature data:\n')
    print(cancer['data'][0])
    print('\nFeature names:\n')
    print(cancer['feature_names'])
    print("Sample counts per class:\n{}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
    ))
    print('Data size: (' + str(cancer['data'].shape) + ')')

    # regression dataset
    boston = load_boston()
    print('Regression dataset')
    print('Example feature data:\n')
    print(boston['data'][0])
    print('\nFeature names:\n')
    print(boston['feature_names'])
    print('Data size: (' + str(boston['data'].shape) + ')')

    # we are going to take the boston housing dataset
    # and create a derived feature set from the product of each feature
    # MinMaxScaler().fit_transform: Scale all data values to between 0 to 1, fit & return transformed datasets
    #   see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    # PolynomialFeatures: create the derived feature set, equal to or less than the degree specified
    #   degree kwarg: if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].  # noqa
    #   include_bias: include a bias column in the output, column where all polynomial powers are zero, intercept for linear model  # noqa
    #   see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    Xboston = MinMaxScaler().fit_transform(boston.data)
    Xboston = PolynomialFeatures(degree=2, include_bias=False).fit_transform(Xboston)

    # forge dataset, toy set
    Xforge, yforge = mglearn.datasets.make_forge()


    ## KNN Classifier ##
    # using a voting system, specified by the neighbors kwarg                 #
    # ties in a KNN voting classifier go to the lowest class value in the tie #
    Xforge_train, Xforge_test, yforge_train, yforge_test = train_test_split(Xforge, yforge, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(Xforge_train, yforge_train)

    print('Forge dataset classification, experiment:')
    print('KNN, k = 3')
    print("Test set prediction {}".format(clf.predict(Xforge_test)))
    print("Test set accuracy {:.2f}".format(clf.score(Xforge_test, yforge_test)))

    # lets visualize the decision boundary of this KNN classifier
    # fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    #
    # for n_neighbors, ax in zip([1, 3, 9], axes):
    #     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(Xforge, yforge)
    #     plot_2d_separator(clf, Xforge, fill=True, eps=0.5, ax=ax, alpha=.4)
    #     discrete_scatter(Xforge[:, 0], Xforge[:, 1], yforge, ax=ax)
    #     ax.set_title("{} neightbor(s)".format(n_neighbors))
    #     ax.set_xlabel("feature 0")
    #     ax.set_ylabel("feature 1")
    #
    # axes[0].legend(loc=3)

    # View plot in notebook
    # plt.show(fig)

    Xcancer_train, Xcancer_test, ycancer_train, ycancer_test = train_test_split(cancer.data, cancer.target,
                                                                                stratify=cancer.target, random_state=66)
    training_accuracy, testing_accuracy = [], []

    for n_neighbors in range(1, 11):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(Xcancer_train, ycancer_train)
        training_accuracy.append(clf.score(Xcancer_train, ycancer_train))
        testing_accuracy.append(clf.score(Xcancer_test, ycancer_test))

    # View accuracy/testing plot in notebook
    # plt.plot(range(1, 11), training_accuracy, label="training accuracy")
    # plt.plot(range(1, 11), testing_accuracy, label="testing accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("n_neighbors")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
