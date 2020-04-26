# Breast-Cancer-DataMining


# **Table of Contents**

[ABSTRACT 3](#_Toc38746747)

[1.Introduction: 4](#_Toc38746748)

[_ **1.1** __ **Overview** _ 4](#_Toc38746749)

[_ **1.2** __ **Literature Review** _ 4](#_Toc38746750)

[_ **1.3** __ **Objective &amp; Contribution** _ 4](#_Toc38746751)

[2.Data Analysis 5](#_Toc38746752)

[_ **2.1** __ **Attribute Information:** _ 5](#_Toc38746753)

[_ **2.2** __ **Data Exploratory Analysis:** _ 5](#_Toc38746754)

[_ **2.2.1** __ **Data Summary** _ 5](#_Toc38746755)

[_ **2.2.2** __ **Data Visualization** _ 7](#_Toc38746756)

[_ **2.2.3** __ **Bi-variate/Multivariate Analysis** _ 8](#_Toc38746757)

[_ **2.3** __ **Dimensionality Reduction:** _ 9](#_Toc38746758)

[_ **2.3.1** __ **Principal Component Analysis** _ 9](#_Toc38746759)

[_ **2.3.2** __**Linear Discriminant Analysis (LDA)**_ 11](#_Toc38746760)

[_ **2.4** __ **Data Preparation:** _ 12](#_Toc38746761)

[3.Modeling 13](#_Toc38746762)

[_ **3.1** __ **Model Analysis** _ 13](#_Toc38746765)

[_ **3.1.1** __ **Un-Supervised methods of Classifying Breast Cancer:** _ 13](#_Toc38746766)

[**3.2.2**** Supervised methods of Classifying Breast Cancer:** 14](#_Toc38746767)

[_ **3.2.3** __ **Semi-Supervised methods of Classifying Breast Cancer:** _ 27](#_Toc38746768)

[4.Conclusion 29](#_Toc38746769)

[References 29](#_Toc38746770)

# ABSTRACT

_This course projects includes analysis on Breast Cancer Prediction using Data Mining based on Breast Cancer Wisconsin (Diagnostic) Dataset from UCI repository. The major goal of this course project is to experiment data mining techniques covered under CSC 7810 - Data Mining: Algorithms and Applications and work towards implementation of those knowledge to develop a near to perfect model in predicting Breast Cancer. The more accurate the model are, more chances of artificial systems to predict if the person is having Breast Cancer. Hence the main outline of this project lies on studying existing literature, draw comparison between Unsupervised, Supervised &amp; Semi-Supervised Learning. This course work also covers novel techniques of predicting Breast Cancer using Semi-Supervised Learning. Additionally, experiments are also performed in Neural Networks to build a robust model implementing backpropagation algorithm using R interface to Keras framework &amp; TensorFlow backend engine and comparisons are drawn using multiple hidden layers and activation functions._

1.
# Introduction:

  1.
## _ **Overview** _

Breast Cancer is a group of disease in which cells in breast tissue change and divide uncontrollably leading to lump or mass. It is the most common cancer diagnosed among women and is the one of the leading causes of death among women after lung cancer in the United States. It is the most common type of cancer which causes 411,000 annual deaths worldwide.

  1.
## _ **Literature Review** _

In multiple literatures, various machine learning models - both supervised and unsupervised models have been suggested to classify Breast Cancer. However, we find till date, most approaches suggested in the literatures differ mostly in the adopted data mining technique and how to deal with the missing attribute values and labels. An important shortcoming that most of these methods share is that they are either designed for big datasets or have not been tested enough to address the challenge of data scarcity, which is often the case for cancer datasets. We take this opportunity to build our model using different approach that will address this challenge

Below mentioned literatures are studied as part of this contribution.

![](RackMultipart20200426-4-14pxhjv_html_e8c67953865c0ab5.png)

Here is the short summary of these literatures:

**Self-Training** is a basic semi-supervised learning method where a classifier is initially trained with a small number of labeled examples and this classifier is used to predict the labels of unlabeled samples.

Few literatures already implemented SSL which includes _Self-Training_ technique, however as _per our best of knowledge following methods of semi-supervised learning for a breast cancer problem has not been explored before._

**Self-Training with editing (SETRED)** is an SSL technique in which a learner keeps on labeling unlabeled examples and retraining itself on an enlarged labeled training set. Since the self-training process may erroneously label some of the unlabeled examples, sometimes the learned hypothesis does not perform well.

**Self-training Nearest Neighbor Rule using Cut Edges (SNNRCE)** is a variant of the self-training classification method with a different addition mechanism and a fixed learning scheme (1-NN). The mislabeled examples are identified using the local information provided by the neighborhood graph. A statistical test using cut edge weight is used to modify the labels of the misclassified examples.

**Tri-training** algorithm generates **three** classifiers from the original labeled example set. These classifiers are then refined using unlabeled examples in the tri-training process. In detail, at each round of tri-training, an unlabeled example is labeled for a classifier only if the other two classifiers agree on the labeling, under certain conditions.

**Co-Training by Committee** is a SSL technique that requires two _views_ of the data. It assumes that each example is described using two different feature sets that provide different, complementary information about the instance. Ideally, the two views are conditionally independent, and each view is enough. Co-training first learns a separate classifier for each view using any labeled examples. The most confident predictions of each classifier on the unlabeled data are then used to iteratively construct additional labeled training data.

**Democratic co-learning** is the methodology in which multiple algorithms instead of multiple views enable learners to label data for each other. The technique leverages off the fact that different learning algorithms have different inductive biases and that better predictions can be made by the voted majority.

  1.
## _ **Objective &amp; Contribution** _

The major motivation of this course project is to develop an effective product using Data Mining in medical domain that will transform the data and reduce dimension to reveal patterns in the dataset and create a robust analysis.

- Our **novel contribution** lies in utilizing multiple semi supervised learning (SSL) methods to build models and demonstrate its comparable accuracy with Supervised model since SSL technique is proven better and useful for a medical domain problem.
- Additionally, we performed end to end analysis and produced comparable results for Supervised &amp; Unsupervised Learning too.
- Furthermore, as a part of Supervised Model analysis, we focused on Neural Network to compare with multiple hidden layers and activation function.

The optimal models are selected based on comparable factors like balanced accuracy, sensitivity, specificity and F1 score, among other factors.

1.
# Data Analysis

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The data used for this project was collected by the University of Wisconsin and its composed by the biopsy result of 569 patients in Wisconsin hospital. They describe characteristics of the cell nuclei present in the image. The dataset is created by Dr. Wolberg a physician at University of Wisconsin and can be found at UCI repository [[Web Link](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))].

**Number of instances: 569**

**Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)**

  1.
## _ **Attribute Information:** _

The dataset&#39;s features describe characteristics of the cell nuclei on the image.

| **wdbc.data** |
| --- |
|
1.
 | ID number | Identification # |
|
1.
 | Diagnosis | M = malignant, B = benign |
| 3-32 | Ten real-valued features are computed for each cell nucleus:
 |
|
1.
 | radius | mean of distances from center to points on the perimeter |
|
1.
 | texture  | standard deviation of gray-scale values |
|
1.
 | perimeter |
 |
|
1.
 | area | Number of pixels inside contour + ½ for pixels on perimeter |
|
1.
 | smoothness | local variation in radius lengths |
|
1.
 | compactness  | perimeter^2 / area - 1.0;this dimensionless number is at a minimum with a circular disk and increases with the irregularity of the boundary, but this measure also increases for elongated cell nuclei, which is not indicative of malignancy |
|
1.
 | concavity | severity of concave portions of the contour |
|
1.
 | concave points | number of concave portions of the contour |
|
1.
 | symmetry |
 |
|
1.
 | fractal dimension | &quot;coastline approximation&quot; – 1;a higher value corresponds a less regular contour and thus to a higher probability of malignancy |

  1.
## _ **Data Exploratory Analysis:** _

We plotted the data on a histogram to understand its distribution. This step is essential fruitful analysis and understand the data pertaining to Breast Cancer disease prediction problem. This type of analysis is also useful when we do not know the right range of value, an attribute can hold especially for a dataset like this.

    1.
### _ **Data Summary** _

We observed from data summary that the data frame consists of 32 variables

\&gt; str(cancer\_data)

&#39;data.frame&#39;: 569 obs. of 32 variables:

$ id : int 842302 842517 84300903 84348301 84358402 843786 844359 84458202 844981 84501001 ...

$ diagnosis : chr &quot;M&quot; &quot;M&quot; &quot;M&quot; &quot;M&quot; ...

$ radius\_mean : num 18 20.6 19.7 11.4 20.3 ...

$ texture\_mean : num 10.4 17.8 21.2 20.4 14.3 ...

$ perimeter\_mean : num 122.8 132.9 130 77.6 135.1 ...

$ area\_mean : num 1001 1326 1203 386 1297 ...

$ smoothness\_mean : num 0.1184 0.0847 0.1096 0.1425 0.1003 ...

$ compactness\_mean : num 0.2776 0.0786 0.1599 0.2839 0.1328 ...

$ concavity\_mean : num 0.3001 0.0869 0.1974 0.2414 0.198 ...

$ concave\_points\_mean : num 0.1471 0.0702 0.1279 0.1052 0.1043 ...

$ symmetry\_mean : num 0.242 0.181 0.207 0.26 0.181 ...

$ fractal\_dimension\_mean : num 0.0787 0.0567 0.06 0.0974 0.0588 ...

$ radius\_se : num 1.095 0.543 0.746 0.496 0.757 ...

$ texture\_se : num 0.905 0.734 0.787 1.156 0.781 ...

$ perimeter\_se : num 8.59 3.4 4.58 3.44 5.44 ...

$ area\_se : num 153.4 74.1 94 27.2 94.4 ...

$ smoothness\_se : num 0.0064 0.00522 0.00615 0.00911 0.01149 ...

$ compactness\_se : num 0.049 0.0131 0.0401 0.0746 0.0246 ...

$ concavity\_se : num 0.0537 0.0186 0.0383 0.0566 0.0569 ...

$ concave\_points\_se : num 0.0159 0.0134 0.0206 0.0187 0.0188 ...

$ symmetry\_se : num 0.03 0.0139 0.0225 0.0596 0.0176 ...

$ fractal\_dimension\_se : num 0.00619 0.00353 0.00457 0.00921 0.00511 ...

$ radius\_worst : num 25.4 25 23.6 14.9 22.5 ...

$ texture\_worst : num 17.3 23.4 25.5 26.5 16.7 ...

$ perimeter\_worst : num 184.6 158.8 152.5 98.9 152.2 ...

$ area\_worst : num 2019 1956 1709 568 1575 ...

$ smoothness\_worst : num 0.162 0.124 0.144 0.21 0.137 ...

$ compactness\_worst : num 0.666 0.187 0.424 0.866 0.205 ...

$ concavity\_worst : num 0.712 0.242 0.45 0.687 0.4 ...

$ concave\_points\_worst : num 0.265 0.186 0.243 0.258 0.163 ...

$ symmetry\_worst : num 0.46 0.275 0.361 0.664 0.236 ...

$ fractal\_dimension\_worst: num 0.1189 0.089 0.0876 0.173 0.0768 ...

![](RackMultipart20200426-4-14pxhjv_html_10da46507c30abb.png)

We observed that &#39;id&#39; column has no contribution in class prediction and hence we dropped it.

\&gt; dim(cancer\_data)

[1] 569 31

We further check if the dataset has any missing value:

| \&gt; map(cancer\_data, function(.x) sum(is.na(.x)))$diagnosis[1] 0$radius\_mean[1] 0$texture\_mean[1] 0$perimeter\_mean[1] 0$area\_mean[1] 0$smoothness\_mean[1] 0$compactness\_mean[1] 0$concavity\_mean[1] 0$concave\_points\_mean[1] 0$symmetry\_mean[1] 0$fractal\_dimension\_mean[1] 0$radius\_se[1] 0$texture\_se[1] 0$perimeter\_se[1] 0$area\_se[1] 0$smoothness\_se[1] 0 |
$compactness\_se[1] 0$concavity\_se[1] 0$concave\_points\_se[1] 0$symmetry\_se[1] 0$fractal\_dimension\_se[1] 0$radius\_worst[1] 0$texture\_worst[1] 0$perimeter\_worst[1] 0$area\_worst[1] 0$smoothness\_worst[1] 0$compactness\_worst[1] 0$concavity\_worst[1] 0$concave\_points\_worst[1] 0$symmetry\_worst[1] 0$fractal\_dimension\_worst[1] 0
 |
| --- | --- |

Hence the summarization of data is as follows:

- **Diagnosis is a categorical variable.**
- **All feature values are recoded with four significant digits.**
- **No missing attribute values**
- **Data is well organized**
- **Class distribution: 357 benign, 212 malignant**

    1.
### _ **Data Visualization** _

We perform Data Visualization to find which features are most helpful in predicting malignant or benign cancer.

| \&gt; prop.table(diagnosis.table)B M 0.6274165 0.3725835
 | ![](RackMultipart20200426-4-14pxhjv_html_5cd794e94edc37ac.png) |
| --- | --- |
|
 | M= Malignant (indicates presence of cancer cells); B= Benign (indicates absence)
 |

357 observations which account for 62.7% of all observations indicating the absence of cancer cells, 212 which account for 37.3% of all observations shows the presence of cancerous cell.

By analyzing the dataset, it is found that it is a **bit unbalanced in its proportions.**

Next plot is using histograms to show how individual factor affecting **malignancy**.

| ![](RackMultipart20200426-4-14pxhjv_html_88979687077bf9aa.png) | ![](RackMultipart20200426-4-14pxhjv_html_81b0aa55353674e4.png) |
| --- | --- |
| ![](RackMultipart20200426-4-14pxhjv_html_2e2efa6479aca5eb.png) |

By looking into the histograms:

- We find most of the features are **normally distributed**.
- Comparison of radius distribution by malignancy shows that there is no perfect separation between any of the features.
- We do have good separations for _concave\_points\_worst, concavity\_worst, perimeter\_worst, area\_mean, perimeter\_mean_.
- We do have as well tight superposition for some of the values, like _symmetry\_se, smoothness\_se_.

    1.
### _ **Bi-variate/Multivariate Analysis** _

An important step in exploratory data analysis step is to identify if there is any correlation between any of these variables. By using **Pearson&#39;s correlation** , the below plot was created. The positive correlation between two variables is demonstrated through the darkness of blue color, i.e **. **** darker the blue colored box, stronger is the positive correlation between respective variables **. Similarly, negative correlation between two variables is demonstrated through the darkness of orange color, i.e. ** darker the orange colored box, stronger is the negative correlation between respective variables ****.**

![](RackMultipart20200426-4-14pxhjv_html_3c94ddfef6e9faad.png)

Often, we have features that are highly correlated and those provide redundant information.

It is observed that quite few variables which are co-related. By **eliminating highly correlated features we can avoid a predictive bias for the information contained in these features**. This also shows us, that when we want to make assumptions about the biological/ medical significance of specific features, we need to keep in mind that just because they are suitable to predicting an outcome, **they are not necessarily causal - they could simply be correlated with causal factors.**

All features with a correlation higher than 0.9 are removed, keeping the features with the lower mean.

\&gt; print(highlyCorrelated)

[1] 7 8 23 21 3 24 1 13 14 2

We identified 10 variables are highly co-related and we removed those variables. The transformed dataset is 10

variables shorter.

\&gt; # Check column count after removing correlated variables

\&gt; ncol(cancer\_data\_wcor)

[1] 20

  1.
## _ **Dimensionality Reduction:** _

    1.
### _ **Principal Component Analysis** _

Principal component analysis (PCA) is a technique used to emphasize variation and bring out strong patterns in a dataset. It&#39;s often used to make data easy to explore and visualize. For two-dimensional data, PCA seeks to rotate these two axes so that the new axis _X&#39;_ lies along the direction of maximum variation in the data. PCA requires that the axes be perpendicular, so in two dimensions the choice of _X&#39;_ will determine _Y&#39;_.

[Source: [https://setosa.io/ev/principal-component-analysis/](https://setosa.io/ev/principal-component-analysis/)]

![](RackMultipart20200426-4-14pxhjv_html_7bf85a04468ef8be.png)

![](RackMultipart20200426-4-14pxhjv_html_f78e788e37c6544f.png) ![](RackMultipart20200426-4-14pxhjv_html_14cafd66963b2150.png)

From the above result we found first two components explains the 0.6324 of the variances.

**10 principal components explain more than 0.95 of the variances and 17 to explain more than 0.99.**

_ **PCA- Biplot:** _

A biplot simultaneously plots information on the observations and the variables in a multidimensional dataset.

![](RackMultipart20200426-4-14pxhjv_html_3456c4af2ed71447.png)

Now we perform Principal Component Analysis without the co-related variables.

![](RackMultipart20200426-4-14pxhjv_html_c8aaed9c0b64d4ed.png) ![](RackMultipart20200426-4-14pxhjv_html_beebdad8ee8a9bd4.png)

![](RackMultipart20200426-4-14pxhjv_html_f73b218aaac1db48.png)

From the above table it is observed that 95% of the variance is explained with 10 PC&#39;s in the transformed dataset.

To visualize which variables are the most contributing on the first 2 components, below is plotted:

![](RackMultipart20200426-4-14pxhjv_html_522ca1e5ed35a0b1.png)

![](RackMultipart20200426-4-14pxhjv_html_ecf962279c003f69.png)

The data of the first 2 components can be easily separated into two classes. This is caused by the fact that

the variance explained by these components is not large. The data can be easily separated.

![](RackMultipart20200426-4-14pxhjv_html_f33ce1bcf8b4634e.png)

Even we tried visualizing the first 3 components.

![](RackMultipart20200426-4-14pxhjv_html_6f59681357c7ac83.png)

As it can be seen from the above plots the first 3 principal components separate the two classes to some extent only, this is expected since the variance explained by these components is not large.

We used the caret preProcess to apply pca with a **0.99 threshold.**

    1.
### _**Linear Discriminant Analysis (LDA)**_

Linear Discriminant Analysis is a linear combination of features that characterizes or separates two or more classes of objects or events.

![](RackMultipart20200426-4-14pxhjv_html_ec6b49d0d02f058b.png)

Coefficients of linear discriminants:

LD1

radius\_mean -1.075583600

texture\_mean 0.022450225

perimeter\_mean 0.117251982

area\_mean 0.001569797

smoothness\_mean 0.418282533

compactness\_mean -20.852775912

concavity\_mean 6.904756198

concave\_points\_mean 10.578586272

symmetry\_mean 0.507284238

fractal\_dimension\_mean 0.164280222

radius\_se 2.148262164

texture\_se -0.033380325

perimeter\_se -0.111228320

area\_se -0.004559805

smoothness\_se 78.305030179

compactness\_se 0.320560148

concavity\_se -17.609967822

concave\_points\_se 52.195471457

symmetry\_se 8.383223501

fractal\_dimension\_se -35.296511336

radius\_worst 0.964016085

texture\_worst 0.035360398

perimeter\_worst -0.012026798

area\_worst -0.004994466

smoothness\_worst 2.681188528

compactness\_worst 0.331697102

concavity\_worst 1.882716394

concave\_points\_worst 2.293242388

symmetry\_worst 2.749992654

fractal\_dimension\_worst 21.255049570

![](RackMultipart20200426-4-14pxhjv_html_dd078cd1c3143784.png)

These LDA features are used later as part of Neural Network Analysis.

  1.
## _ **Data Preparation:** _

Data preparation is an important step when building models. We split the dataset into Train set (80%) and Test set (20%).

![](RackMultipart20200426-4-14pxhjv_html_b4d5f37068d8e8b6.png)

\&gt; dim(train\_data)

[1] 456 21

\&gt; dim(test\_data)

[1] 113 21

1.
# Modeling

Data mining techniques have been traditionally used to extract the hidden predictive information in many diverse contexts. 3 different learning methods we experimented in our modeling technique.

- **Supervised learning**  aims to learn a function that, given a sample of data and desired outputs, approximates a function that maps inputs to outputs.
- **Semi-supervised learning**  aims to label unlabeled data points using knowledge learned from a small number of labeled data points.
- **Unsupervised learning ** does not have (or need) any labeled outputs, so its goal is to infer the natural structure present within a set of data points.&quot;

  1.
    1.

  1.
## _ **Model Analysis** _

Usually datasets contained thousands of examples. Semi-supervised learning addresses this problem by using large amount of unlabeled data, together with the labelled data, to build better classifiers and higher accuracy, located between supervised learning with fully labelled and unsupervised learning without any labelled. Labelled instances, however, are often difficult and expensive to obtain. Meanwhile, unlabeled data may be relatively easy to collect, but there have been few ways to use them. So, we tried to compare all the models to see which suits best.

    1.
### _ **Un-Supervised methods of Classifying Breast Cancer:** _

Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. Normally when performing unsupervised learning like this, a target variable (i.e. known answer or labels) isn&#39;t available. We do have it with this dataset however, this label information is only used to verify the un-supervised outcomes with actual ones.

      1.
#### **Hierarchical Clustering:**

![](RackMultipart20200426-4-14pxhjv_html_790f7ec81e5199bd.png)

We can see if we cut the dendrogram into 4. It will give us the main clusters and then we have a couple tiny ones on the left.

\&gt; # Compare hierarchical to actual diagnosis

B M

1 1 93

2 356 107

3 0 9

4 0 3

Here we picked four clusters and see that cluster 1 largely corresponds to malignant cells whereas cluster 2 largely corresponds to benign cells. In these two clusters there are (1+107) 108 instances that are not correctly classified.

\&gt; # No of incorrect Classification in hierarchical clustering

[1] 108

      1.
#### **K-means Clustering:**

We further experimented with k-means clustering algorithm to analyze how it works with our dataset.

There are 2 clusters created corresponding to the actual number of diagnosis. We repeat the algorithm 20 times. Running multiple times, as this will help to find a well performing model.

\&gt; # Compare k-means to actual diagnosis

B M

1 356 **95**

2 **1** 117

In these two clusters there are (1+95) 96 instances that are not correctly classified.

\&gt; # No of incorrect Classification in k-means clustering

[1] 96

If we compare this result with our previous hierarchical clustering as the number of incorrect instances is greater than k-means by 12.

\&gt; # Compare k-means to hierarchical clustering

1 2

1 0 94

2 451 12

3 0 9

4 0 3

\&gt; # Difference between incorrect instance classified in two clusters

[1] 12

#### **Conclusion**

Though k-means clustering performs better than hierarchical one, but it failed to classify **16.87%**

(96/569 \*100 = 16.87%) of data. This is quite risky in terms of medical domain **. It identified 95 patients as cancer patient who are not actually cancer patient and this percentage will exponentially increase in case of real-world scenario!**

This is fatal for predicting Breast Cancer Problem and hence we discard Unsupervised Learning model and

proceed with Supervised and Semi-Supervised Methods.

    1.
### **Supervised methods of Classifying Breast Cancer:**

1.
2.

      1.
#### **Naïve Bayes Model**

A Naive Bayes classifier is a probabilistic machine learning model that&#39;s used for classification task. The crux of the classifier is based on the Bayes theorem.

![](RackMultipart20200426-4-14pxhjv_html_e0bb6c0e535d5d03.png)

\&gt; confusionmatrix\_naiveb

Confusion Matrix and Statistics

Reference

Prediction B M

B 67 5

M 4 37

Accuracy : 0.9204

95% CI : (0.8542, 0.9629)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : 9.656e-13

Kappa : 0.8286

Mcnemar&#39;s Test P-Value : 1

Sensitivity : 0.8810

Specificity : 0.9437

Pos Pred Value : 0.9024

Neg Pred Value : 0.9306

Prevalence : 0.3717

Detection Rate : 0.3274

Detection Prevalence : 0.3628

Balanced Accuracy : 0.9123

&#39;Positive&#39; Class : M

The topmost significant variables and those which contributes helping in best prediction are as follows:

![](RackMultipart20200426-4-14pxhjv_html_caf11d42bb73e483.png)

      1.
#### **Logistic Regression**

Logistic regression is the appropriate regression analysis to conduct when the dependent variable is binary.

\&gt; confusionmatrix\_logreg

Confusion Matrix and Statistics

Reference

Prediction B M

B 67 0

M 4 42

Accuracy : 0.9646

95% CI : (0.9118, 0.9903)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : \&lt;2e-16

Kappa : 0.9257

Mcnemar&#39;s Test P-Value : 0.1336

Sensitivity : 1.0000

Specificity : 0.9437

Pos Pred Value : 0.9130

Neg Pred Value : 1.0000

Prevalence : 0.3717

Detection Rate : 0.3717

Detection Prevalence : 0.4071

Balanced Accuracy : 0.9718

&#39;Positive&#39; Class : M

The topmost significant variables and those which contributes helping in best prediction are as follows:

![](RackMultipart20200426-4-14pxhjv_html_bc53adc64ce34a92.png)

Tried plotting using PCA variables and found that the accuracy is same with the model with non-PCA variables.

\&gt; confusionmatrix\_logreg\_pca

Confusion Matrix and Statistics

Reference

Prediction B M

B 67 0

M 4 42

Accuracy : 0.9646

95% CI : (0.9118, 0.9903)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : \&lt;2e-16

Kappa : 0.9257

Mcnemar&#39;s Test P-Value : 0.1336

Sensitivity : 1.0000

Specificity : 0.9437

Pos Pred Value : 0.9130

Neg Pred Value : 1.0000

Prevalence : 0.3717

Detection Rate : 0.3717

Detection Prevalence : 0.4071

Balanced Accuracy : 0.9718

&#39;Positive&#39; Class : M

The topmost significant variables and those which contributes helping in best prediction are as follows:

![](RackMultipart20200426-4-14pxhjv_html_173f53c24cd8e26f.png)

      1.
#### **SVM (with radial Kernel)**

Support vector machines (SVM) is a _ **supervised learning algorithm** _ based on the idea of finding a hyperplane that best separates the features into different domains. **Gaussian RBF (Radial Basis Function)** is popular Kernel method used in SVM models. RBF kernel is a function whose value depends on the distance from the origin or from some point.

Gaussian Kernel is of the following format:

![](RackMultipart20200426-4-14pxhjv_html_760efe9c7c57dadb.png)

||X1 — X2 || = Euclidean distance between X1 &amp; X2

\&gt; confusionmatrix\_svm

Confusion Matrix and Statistics

Reference

Prediction B M

B 69 2

M 2 40

Accuracy : 0.9646

95% CI : (0.9118, 0.9903)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : \&lt;2e-16

Kappa : 0.9242

Mcnemar&#39;s Test P-Value : 1

Sensitivity : 0.9524

Specificity : 0.9718

Pos Pred Value : 0.9524

Neg Pred Value : 0.9718

Prevalence : 0.3717

Detection Rate : 0.3540

Detection Prevalence : 0.3717

Balanced Accuracy : 0.9621

&#39;Positive&#39; Class : M

The topmost significant variables and those which contributes helping in best prediction are as follows:

![](RackMultipart20200426-4-14pxhjv_html_87182d9cbf0a14fb.png)

      1.
#### **Random Forest Classifier**

Random forest, like its name implies, consists of many individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes the model&#39;s prediction.[Source: [https://towardsdatascience.com/understanding-random-forest-58381e0602d2](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)]

\&gt; confusionmatrix\_randomforest

Confusion Matrix and Statistics

Reference

Prediction B M

B 70 3

M 1 39

Accuracy : 0.9646

95% CI : (0.9118, 0.9903)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : \&lt;2e-16

Kappa : 0.9235

Mcnemar&#39;s Test P-Value : 0.6171

Sensitivity : 0.9286

Specificity : 0.9859

Pos Pred Value : 0.9750

Neg Pred Value : 0.9589

Prevalence : 0.3717

Detection Rate : 0.3451

Detection Prevalence : 0.3540

Balanced Accuracy : 0.9572

&#39;Positive&#39; Class : M

The topmost significant variables and those which contributes helping in best prediction are as follows:

![](RackMultipart20200426-4-14pxhjv_html_901a845b7e1feea0.png)

Tried checking Random Forest algorithm with PCA variables. Results are as follows:

\&gt; confusionmatrix\_randomforest\_pca

Confusion Matrix and Statistics

Reference

Prediction B M

B 70 5

M 1 37

Accuracy : 0.9469

95% CI : (0.888, 0.9803)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : 1.866e-15

Kappa : 0.8841

Mcnemar&#39;s Test P-Value : 0.2207

Sensitivity : 0.8810

Specificity : 0.9859

Pos Pred Value : 0.9737

Neg Pred Value : 0.9333

Prevalence : 0.3717

Detection Rate : 0.3274

Detection Prevalence : 0.3363

Balanced Accuracy : 0.9334

&#39;Positive&#39; Class : M

![](RackMultipart20200426-4-14pxhjv_html_dd73c3f5f2f5881a.png)

The result does not show significant increase.

      1.
#### **K-Nearest Neighbor**

The k-nearest neighbor (KNN) algorithm classifies objects on closest training examples in the feature space.

\&gt; confusionmatrix\_knn

Confusion Matrix and Statistics

Reference

Prediction B M

B 71 6

M 0 36

Accuracy : 0.9469

95% CI : (0.888, 0.9803)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : 1.866e-15

Kappa : 0.8829

Mcnemar&#39;s Test P-Value : 0.04123

Sensitivity : 0.8571

Specificity : 1.0000

Pos Pred Value : 1.0000

Neg Pred Value : 0.9221

Prevalence : 0.3717

Detection Rate : 0.3186

Detection Prevalence : 0.3186

Balanced Accuracy : 0.9286

&#39;Positive&#39; Class : M

Topmost significant variables are as follows:

![](RackMultipart20200426-4-14pxhjv_html_e2745c1072a041d1.png)

      1.
#### **ANN: Single Layer Perceptron**

&quot;Artificial Neural Networks or ANN is an information processing paradigm that is inspired by the way the biological nervous system work, such as brain process information&quot;.

**We use &#39;nnet&#39; package to fit single-hidden-layer neural network, possibly with skip-layer connections.**

_**Neural Network (all variables)**_

\&gt; confusionmatrix\_nnet

Confusion Matrix and Statistics

Reference

Prediction B M

B 69 2

M 2 40

Accuracy : 0.9646

95% CI : (0.9118, 0.9903)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : \&lt;2e-16

Kappa : 0.9242

Mcnemar&#39;s Test P-Value : 1

Sensitivity : 0.9524

Specificity : 0.9718

Pos Pred Value : 0.9524

Neg Pred Value : 0.9718

Prevalence : 0.3717

Detection Rate : 0.3540

Detection Prevalence : 0.3717

Balanced Accuracy : 0.9621

&#39;Positive&#39; Class : M

![](RackMultipart20200426-4-14pxhjv_html_a993c19410ed820f.png)

_**Neural Network (PCA)**_

\&gt; confusionmatrix\_nnet\_pca

Confusion Matrix and Statistics

Reference

Prediction B M

B 68 1

M 3 41

Accuracy : 0.9646

95% CI : (0.9118, 0.9903)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : \&lt;2e-16

Kappa : 0.9249

Mcnemar&#39;s Test P-Value : 0.6171

Sensitivity : 0.9762

Specificity : 0.9577

Pos Pred Value : 0.9318

Neg Pred Value : 0.9855

Prevalence : 0.3717

Detection Rate : 0.3628

Detection Prevalence : 0.3894

Balanced Accuracy : 0.9670

&#39;Positive&#39; Class : M

![](RackMultipart20200426-4-14pxhjv_html_8ca6171f3edb0e86.png)

_**Neural Network (LDA)**_

\&gt; confusionmatrix\_nnet\_lda

Confusion Matrix and Statistics

Reference

Prediction B M

B 70 1

M 1 41

Accuracy : 0.9823

95% CI : (0.9375, 0.9978)

No Information Rate : 0.6283

P-Value [Acc \&gt; NIR] : \&lt;2e-16

Kappa : 0.9621

Mcnemar&#39;s Test P-Value : 1

Sensitivity : 0.9762

Specificity : 0.9859

Pos Pred Value : 0.9762

Neg Pred Value : 0.9859

Prevalence : 0.3717

Detection Rate : 0.3628

Detection Prevalence : 0.3717

Balanced Accuracy : 0.9811

&#39;Positive&#39; Class : M

We experimented Neural Network with 3 different types of variables as we see it gives great accuracy in all the

models explored. Specifically, for LDA variables it works impressively. We explored this option with LDA variable as we studied in literature, LDA actually works as a linear classifier.

#### **Comparison**

We compared all the models processed so far and listed the summary as follows:

\&gt; summary(models\_results)

Call:

summary.resamples(object = models\_results)

Models: Naive\_Bayes, Logistic\_regr, Logistic\_regr\_PCA, SVM, Random\_Forest, Random\_Forest\_PCA, KNN, Neural, Neural\_PCA, Neural\_LDA

Number of resamples: 15

ROC

Min. 1st Qu. Median Mean 3rd Qu. Max. NA&#39;s

Naive\_Bayes 0.8803828 0.9429825 0.9712919 0.9630569 0.9906699 1 0

Logistic\_regr 0.8903509 0.9617225 1.0000000 0.9767411 1.0000000 1 0

Logistic\_regr\_PCA 0.9649123 0.9880383 1.0000000 0.9930090 1.0000000 1 0

SVM 0.9665072 0.9882775 1.0000000 0.9939979 1.0000000 1 0

Random\_Forest 0.9539474 0.9880383 0.9956140 0.9913211 1.0000000 1 0

Random\_Forest\_PCA 0.9617225 0.9772727 0.9880383 0.9866560 1.0000000 1 0

KNN 0.9229167 0.9780702 0.9856459 0.9829127 0.9989035 1 0

Neural 0.9760766 0.9952153 1.0000000 0.9962254 1.0000000 1 0

Neural\_PCA 0.9330144 0.9954147 1.0000000 0.9933812 1.0000000 1 0

Neural\_LDA 0.9824561 0.9956140 1.0000000 0.9970654 1.0000000 1 0

Sens

Min. 1st Qu. Median Mean 3rd Qu. Max. NA&#39;s

Naive\_Bayes 0.8421053 0.8947368 0.9473684 0.9370175 0.975 1 0

Logistic\_regr 0.8947368 0.9473684 1.0000000 0.9687719 1.000 1 0

Logistic\_regr\_PCA 0.8947368 0.9736842 1.0000000 0.9824561 1.000 1 0

SVM 0.8947368 0.9473684 1.0000000 0.9754386 1.000 1 0

Random\_Forest 0.8947368 0.9736842 1.0000000 0.9824561 1.000 1 0

Random\_Forest\_PCA 0.8421053 0.9473684 1.0000000 0.9649123 1.000 1 0

KNN 0.9473684 0.9473684 1.0000000 0.9791228 1.000 1 0

Neural 0.8947368 0.9473684 1.0000000 0.9789474 1.000 1 0

Neural\_PCA 0.8947368 1.0000000 1.0000000 0.9859649 1.000 1 0

Neural\_LDA 0.9473684 1.0000000 1.0000000 0.9894737 1.000 1 0

Spec

Min. 1st Qu. Median Mean 3rd Qu. Max. NA&#39;s

Naive\_Bayes 0.7272727 0.8257576 0.9090909 0.8873737 0.9583333 1 0

Logistic\_regr 0.7272727 0.8750000 1.0000000 0.9297980 1.0000000 1 0

Logistic\_regr\_PCA 0.8181818 0.9090909 1.0000000 0.9469697 1.0000000 1 0

SVM 0.8333333 0.9090909 1.0000000 0.9590909 1.0000000 1 0

Random\_Forest 0.7500000 0.8712121 0.9090909 0.9141414 1.0000000 1 0

Random\_Forest\_PCA 0.8181818 0.9090909 0.9090909 0.9116162 0.9166667 1 0

KNN 0.7272727 0.7840909 0.8333333 0.8575758 0.9128788 1 0

Neural 0.9090909 0.9128788 1.0000000 0.9646465 1.0000000 1 0

Neural\_PCA 0.8333333 0.9090909 1.0000000 0.9530303 1.0000000 1 0

Neural\_LDA 0.8181818 0.9128788 1.0000000 0.9585859 1.0000000 1 0

From the following plot, one can observe two models, Naive\_bayes and Logistic\_regression have great variability, depending of the processed sample.

![](RackMultipart20200426-4-14pxhjv_html_c6b7c5293a5d8734.png)

The Neural Network LDA model achieve a great Area Under the ROC Curve with some variability.

![](RackMultipart20200426-4-14pxhjv_html_4fb7f492236f0e04.png)

Here LDA worked as a classifier and posteriorly it reduced the dimensionality of the dataset and the neural network performed the classification task. Since ANN gave highest accuracy to the Breast Cancer dataset, our further work will be with neural network to experiment how multi-layer perceptron model will impact the accuracy. Also, **high accuracy for all the models motivated us to fix the problem with unbalanced dataset as mentioned earlier**. Hence, we will be exploring Multi-Layer Perceptrons in the following section with scaled data.

      1.
#### **Multi-Layer Perceptron**

The application was developed using R interface to Keras framework and TensorFlow backend engine in order to **train a sequential model which uses the backpropagation algorithm**. The basic principle of multi-layer perceptrons is the parallelism and the mathematical background. Parallelism is based on computation of the neurons at the same layer can be independent from each other. If we treat **the input layer, the hidden layer and the output layer as three nodes in a Markov Chain model, where the computation at each layer except for the input layer is dependent on the computation at its subsequent lower layer.** Hence to applyour learning in this course we are motivated to work with **Multi-layer Perceptron**.

**Handling the data problem:** The original data analysis revealed to us that the features have different order of magnitude. In order to avoid a situation where features with higher values have greater contribution to the outcome of the model, each feature was normalized using R build-in scale() function. This function **transforms each value by subtracting the mean value of the feature and dividing it by the standard deviation.**

_ **Experiments with Activation Function:** _

##### _ **ReLU** _

Train on 410 samples, validate on 46 samples

Epoch 1/20

410/410 [==============================] - 1s 3ms/sample - loss: 1.0276 - accuracy: 0.2780 - val\_loss: 1.0005 - val\_accuracy: 0.2826

Epoch 2/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.8640 - accuracy: 0.3317 - val\_loss: 0.8440 - val\_accuracy: 0.3261

Epoch 3/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.7297 - accuracy: 0.4317 - val\_loss: 0.7151 - val\_accuracy: 0.4565

Epoch 4/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.6219 - accuracy: 0.6488 - val\_loss: 0.6137 - val\_accuracy: 0.6739

Epoch 5/20

410/410 [==============================] - 1s 1ms/sample - loss: 0.5427 - accuracy: 0.8122 - val\_loss: 0.5325 - val\_accuracy: 0.8478

Epoch 6/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.4818 - accuracy: 0.8561 - val\_loss: 0.4709 - val\_accuracy: 0.8696

Epoch 7/20

410/410 [==============================] - 1s 1ms/sample - loss: 0.4357 - accuracy: 0.8707 - val\_loss: 0.4233 - val\_accuracy: 0.9130

Epoch 8/20

410/410 [==============================] - 1s 1ms/sample - loss: 0.3985 - accuracy: 0.9000 - val\_loss: 0.3847 - val\_accuracy: 0.9565

Epoch 9/20

410/410 [==============================] - 1s 1ms/sample - loss: 0.3691 - accuracy: 0.9146 - val\_loss: 0.3541 - val\_accuracy: 0.9565

Epoch 10/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.3450 - accuracy: 0.9220 - val\_loss: 0.3280 - val\_accuracy: 0.9783

Epoch 11/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.3243 - accuracy: 0.9268 - val\_loss: 0.3058 - val\_accuracy: 0.9783

Epoch 12/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.3064 - accuracy: 0.9317 - val\_loss: 0.2865 - val\_accuracy: 0.9783

Epoch 13/20

410/410 [==============================] - 1s 1ms/sample - loss: 0.2908 - accuracy: 0.9366 - val\_loss: 0.2703 - val\_accuracy: 0.9783

Epoch 14/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.2775 - accuracy: 0.9366 - val\_loss: 0.2559 - val\_accuracy: 0.9783

Epoch 15/20

410/410 [==============================] - 1s 1ms/sample - loss: 0.2652 - accuracy: 0.9390 - val\_loss: 0.2432 - val\_accuracy: 0.9783

Epoch 16/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.2544 - accuracy: 0.9415 - val\_loss: 0.2320 - val\_accuracy: 1.0000

Epoch 17/20

410/410 [==============================] - 1s 1ms/sample - loss: 0.2450 - accuracy: 0.9415 - val\_loss: 0.2217 - val\_accuracy: 1.0000

Epoch 18/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.2363 - accuracy: 0.9415 - val\_loss: 0.2126 - val\_accuracy: 1.0000

Epoch 19/20

410/410 [==============================] - 1s 1ms/sample - loss: 0.2281 - accuracy: 0.9463 - val\_loss: 0.2048 - val\_accuracy: 1.0000

Epoch 20/20

410/410 [==============================] - 0s 1ms/sample - loss: 0.2210 - accuracy: 0.9488 - val\_loss: 0.1973 - val\_accuracy: 1.0000

![](RackMultipart20200426-4-14pxhjv_html_cd87544249021a25.png)

\&gt; test.result.32

$loss

[1] 0.23472

$accuracy

[1] 0.920354

**Observation:** We trained the model for 20 epochs. The accuracy of the model increases significantly during the first 5 epochs, after which stabilizes. After epoch 5, the loss values continue to drop, and the accuracy keeps increasing, however at a lower rate.

![](RackMultipart20200426-4-14pxhjv_html_a670750bd5e60ca9.png)

##### _ **Sigmoid** _

Train on 410 samples, validate on 46 samples

Epoch 1/20

410/410 [==============================] - 1s 3ms/sample - loss: 1.0613 - accuracy: 0.3829 - val\_loss: 1.0893 - val\_accuracy: 0.2826

Epoch 2/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.8987 - accuracy: 0.3829 - val\_loss: 0.9140 - val\_accuracy: 0.2826

Epoch 3/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.7676 - accuracy: 0.3829 - val\_loss: 0.7877 - val\_accuracy: 0.2826

Epoch 4/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.6787 - accuracy: 0.4366 - val\_loss: 0.6961 - val\_accuracy: 0.3913

Epoch 5/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.6159 - accuracy: 0.5732 - val\_loss: 0.6286 - val\_accuracy: 0.5217

Epoch 6/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.5700 - accuracy: 0.7341 - val\_loss: 0.5775 - val\_accuracy: 0.7174

Epoch 7/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.5349 - accuracy: 0.8244 - val\_loss: 0.5394 - val\_accuracy: 0.8043

Epoch 8/20

410/410 [==============================] - 1s 3ms/sample - loss: 0.5081 - accuracy: 0.8756 - val\_loss: 0.5088 - val\_accuracy: 0.8478

Epoch 9/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.4871 - accuracy: 0.8951 - val\_loss: 0.4826 - val\_accuracy: 0.8696

Epoch 10/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.4686 - accuracy: 0.8902 - val\_loss: 0.4622 - val\_accuracy: 0.9348

Epoch 11/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.4540 - accuracy: 0.8902 - val\_loss: 0.4435 - val\_accuracy: 0.9348

Epoch 12/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.4410 - accuracy: 0.8902 - val\_loss: 0.4275 - val\_accuracy: 0.9348

Epoch 13/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.4297 - accuracy: 0.8902 - val\_loss: 0.4133 - val\_accuracy: 0.9348

Epoch 14/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.4193 - accuracy: 0.9000 - val\_loss: 0.4014 - val\_accuracy: 0.9348

Epoch 15/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.4103 - accuracy: 0.9024 - val\_loss: 0.3898 - val\_accuracy: 0.9348

Epoch 16/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.4016 - accuracy: 0.9000 - val\_loss: 0.3798 - val\_accuracy: 0.9348

Epoch 17/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.3938 - accuracy: 0.9049 - val\_loss: 0.3707 - val\_accuracy: 0.9348

Epoch 18/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.3864 - accuracy: 0.9024 - val\_loss: 0.3627 - val\_accuracy: 0.9348

Epoch 19/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.3797 - accuracy: 0.9024 - val\_loss: 0.3545 - val\_accuracy: 0.9348

Epoch 20/20

410/410 [==============================] - 1s 2ms/sample - loss: 0.3730 - accuracy: 0.9000 - val\_loss: 0.3473 - val\_accuracy: 0.9348

![](RackMultipart20200426-4-14pxhjv_html_742eabfac67ab73b.png)

\&gt; evaluate(ann.sigmoid, test.x, test.y, verbose = 0)

$loss

[1] 0.39895

$accuracy

[1] 0.8584071

From below plot we can see how the loss decreases as the training progresses. It is depicted that the curves have

similar shape however, the loss corresponding to ReLU function seems to decrease slightly faster but most

importantly, it settles at a lower value than the Sigmoid curve.

![](RackMultipart20200426-4-14pxhjv_html_15a1c0231c7842bd.png)

Following the loss for each function, it was examined how the training accuracy of the model changes as the training progresses. On the below plot we can see the values for both activation functions. The sigmoid curve follows the familiar &quot;S&quot; shape and results on slightly lower accuracy. The ReLU curve has the expected shape based on its function and at the end of the training phase has achieved slightly higher accuracy.

![](RackMultipart20200426-4-14pxhjv_html_53dd04d30382f032.png)

We also examined the architecture to see how many nodes on hidden layer gives best accuracy. For that purpose, models with 4, 8, 16 and 64 nodes on the hidden layer were created and compared to the one already examined with 32 nodes on the hidden layer.

**Results**

The below bar plot shows the training accuracy of the above models (epoch = 20). The model with the 64 nodes appears to have very high training accuracy (97%).

![](RackMultipart20200426-4-14pxhjv_html_1a37b10891e40db6.png)

The bar plot below describes the findings regarding the testing accuracy for the above models. It appears that the model with the worst accuracy is the one with 8 nodes on its hidden layer whereas the models with 16 and 64 nodes have the highest accuracy.

![](RackMultipart20200426-4-14pxhjv_html_ed1cba1336dcd31b.png)

\&gt; # Confusion Matrix results of all models

\&gt; table(prediction.4, test.y)

test.y

prediction.4 0 1

0 32 0

1 10 71

\&gt; table(prediction.8, test.y)

test.y

prediction.8 0 1

0 32 2

1 10 69

\&gt; table(prediction.16, test.y)

test.y

prediction.16 0 1

0 37 2

1 5 69

\&gt; table(prediction.32, test.y)

test.y

prediction.32 0 1

0 35 2

1 7 69

\&gt; table(prediction.64, test.y)

test.y

prediction.64 0 1

0 40 2

1 2 69

Therefore, we see with **changing the activation function and number of hidden layers does impact the prediction of result**.

##### _ **Comparison** _

We compared and found &quot; **Neural Network**&quot; is the best model for Supervised Learning. While analyzing MLP, we fixed the data imbalance issue which is the biggest hindrance for working with an imbalance dataset like this.

Accuracy of the MLP model is approximately near to SLP however MLP stand out better owing to data scaling factor. However, the topic is debatable and we can try increasing accuracy of this Multi-layer Perceptron by varying learning rate, batch size, decay rate and loss ratio which is the future scope of this project. Hence to summarize for a Breast Cancer problem with FNA dataset can be proceeded with Neural Networks.

    1.
### _ **Semi-Supervised methods of Classifying Breast Cancer:** _

The crux of SSL is to learn from unlabeled as much as from labeled data to predict accurate data (Chapelle et al., 2006). In SSL, data can be separated into two sets: _L = {x __1__ , . . . , x __l__ }_ with its known labels _Y __l_ _= {y__ 1 __, . . . , y__ l __},_ and _U = {x__ l+1 __, . . . , x__ l+u__}_ for which labels are unknown.

In our SSL methods we tried to implement two different setting such as transductive and inductive learning.

- _ **Transductive learning** _ concerns the problem of predicting the labels of the unlabeled samples provided during the training phase.
- _ **Inductive learning** _ considers the labeled and unlabeled data provided as the training samples, and its objective is to predict the label of unseen data.

**SVM with the RBF kernel** function (kernel matrix using the Gaussian radial basis function (RBF)) is used as **base classifier** and benchmark supervised classifier in all comparisons. The semi-supervised methods evaluated here are: _selfTraining_, _setred_, _coBC_, _triTraining_ and _Democratic co-learning._ _ **Out of which setred, coBC, triTraining and Democratic co-learning method is our noble contribution and is compared against traditional selfTraining method. Brief algorithm is described as follows:** _

#### **Training Methods:**

Our training approach includes six labeled methods. Detail is mentioned as follows:

- **Addition mechanism** describes a scheme in which the enlarged labeled set (EL) is formed. In incremental scheme, the algorithm starts with _EL = L_ and adds, step by step. Another scheme is amending, which differs from incremental in that it can iteratively add or remove any instance that meets a certain criterion.
- **Classifiers** refers to whether it uses one or multiple classifiers during the enlarging phase of the labeled set. Multi-classifier methods combine the learned hypotheses with several classifiers to predict the class of unlabeled instances.
- **Learning** specifies whether the models are constituted by the single or multiple learning algorithms. Multi-learning approaches are closely linked with multi-classifier models; a multi-learning method is itself a multi-classifier method in which the different classifiers come from different learning methods. On the other hand, a single-learning approach can be linked to both single and multi-classifiers.
- **Teaching** is a mutual-teaching approach, the classifiers teach each other their most confident predicted examples. Each _Ci_ classifier has its own _ELi_ which uses for training at each stage. _ELi_ is increased with the most confident labeled examples obtained from remaining classifiers. By contrast, the self-teaching property refers to those classifiers that maintain a single EL.
- **Stopping criteria** This is related to the mechanism used to stop the self-labeling process. It is an important factor since it influences the size of EL and therefore the learned hypothesis. Some of the approaches for this are: (i) repeat the self-labeling process until a portion of U has been exhausted, (ii) establish a limited number of iterations, and (iii) the learned hypothesis remains stable between two consecutive stages.

![](RackMultipart20200426-4-14pxhjv_html_a68ffebd8b2cc892.png) ![](RackMultipart20200426-4-14pxhjv_html_2d9d3c960aa12e2.png)

Individual approach for each algorithm implemented are discussed are as follows:

##### **SETRED**

**SETRED** initiates the self-labeling process by training a model from the original labeled set. In each iteration, the  **learner**  function detects unlabeled examples for which it makes the most confident prediction and labels those examples according to the  **pred**  function. The identification of mislabeled examples is performed using a neighborhood graph created from the distance matrix.

##### **SNNRCE**

**SNNRCE** initiates the self-labeling process by training a 1-NN from the original labeled set. This method attempts to reduce the noise in examples by labeling those instances with no cut edges in the initial stages of self-labeling learning. These highly confident examples are added into the training set. The remaining examples follow the standard self-training process until a minimum number of examples will be labeled for each class.

##### **Tri-Training**

Tri-Traininginitiates the self-labeling process by training three models from the original labeled set, using the  **learner**  function. In each iteration, the algorithm detects unlabeled examples on which two classifiers agree with the classification and includes these instances in the enlarged set of the third classifier under certain conditions. The generation of the final hypothesis is produced via the majority voting. The iteration process ends when no changes occur in any model during a complete iteration.

##### **Co-Training by Committee**

**CoBC** is a semi-supervised learning algorithm with a co-training style.  The method trains an **ensemble of diverse classifiers**. To promote the initial diversity the classifiers are trained from the reduced set of labeled examples by **Bagging**. This algorithm trains N classifiers with the learning scheme defined in the learner argument using a reduced set of labeled examples. For each iteration, an unlabeled example is labeled for a classifier if the most confident classifications assigned by the other  **N-1**  classifiers agree on the labeling proposed. The unlabeled examples candidates are selected randomly from a pool of size u.

##### **Democratic co-learning**

**Demo** is a semi-supervised learning algorithm with a co-training style. This algorithm trains N classifiers with different learning schemes defined in list gen.learners. During the iterative process, the multiple classifiers with different inductive biases label data for each other.

#### **Training Approach**

- To support semi-supervised approach, 70 percent of training data is unlabeled and replaced with NA.
- We used k-SVM as base classifier for all our models.
- In case of democratic method more than one base classifier - KNN, SVM and decision trees(C5.0) are used
- Finally, these models are compared using a supervised classifier that is built on k-SVM as this is the base classifier for all our SSL models.

We compare the accuracy as follows:

![](RackMultipart20200426-4-14pxhjv_html_25712e3d4871c246.gif)\&gt; acc

SelfT SNNRCE SETRED **TriT** coBC Demo

0.9529412 0.9000000 0.9411765 **0.9588235** 0.9352941 0.9470588

\&gt; acc.svm

[1] 0.9529412

![](RackMultipart20200426-4-14pxhjv_html_9fb3a1cea0f89245.png)

#### **Comparison**

Here we find Tri-training obtains the most accurate results and the accuracy is more than traditional supervised learning method. Even Self-training learning method of SSL performed well and provided similar accuracy like supervised model but less than tri-training method. So, a semi-supervised learning performed better than supervised learning in our case.

#### **Additional Analysis (with different dataset)**

We performed additional analysis by implementing **only on SSL** models using a different dataset(Source:[Link](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra)) and just to see if novel Semi supervised methods are performing better than supervised method. No other metrics is taken into consideration as the dataset has much less records after outlier removal.

**Attribute Information:** There are 10 predictors, all quantitative, and a binary dependent variable like the original dataset, indicating the presence or absence of breast cancer.
 The predictors are anthropometric data and parameters which can be gathered in routine blood analysis.
 Prediction models based on these predictors, if accurate, can potentially be used as a biomarker of breast cancer.

| **Quantitative Attributes:**
 Age (years)
 BMI (kg/m2)
 Glucose (mg/dL)
 Insulin (µU/mL)
 HOMA
 Leptin (ng/mL)
 Adiponectin (µg/mL)
 Resistin (ng/mL)
 MCP-1(pg/dL) |

**Labels:**
 1=Healthy controls
 2=Patients |
 ![](RackMultipart20200426-4-14pxhjv_html_90eda1830551619.png) |
| --- | --- | --- |

![](RackMultipart20200426-4-14pxhjv_html_499755c64acbaa3.png)

**Result:** Our additional experiment is successfulas we can see Semi-Supervised Learning outperform supervised learning algorithm. We see 4 out of 6 methods SSL methods performed better than the Supervised method. We could not explore about the accuracy metric of this analysis owing to time-constraint.

1.
# Conclusion

Data mining literature offer some nice classification techniques. But when we implement well known effective classification techniques, the results are found unreliable. The efficacy of the technique comes under scrutiny. We tried finding an optimal method by solving data imbalance issue and we overcome it by data scaling. After fixing the problem, we also implemented robust way of implementing neural network using _R __** **__ interface to __Keras framework and TensorFlow_. The multi-layer perceptron model gave almost same accuracy as compared to single layer perceptron model tested before with imbalanced dataset.

Here, the proposal is about an integrated framework, which ensures the reliability of the class labels assigned to a dataset whose class labels are unknown.Heterogeneous datasets with unknown class labels but known number of classes, after being treated through all the novel SSL models would be able to find the class labels for a significant portion of the data and may be accepted with reliability. Even though our novel methods got little less accuracy than supervised models, however, to solve a typical problem like this where label data is often difficult and expensive to obtain, **we propose Semi-Supervised learning mechanisms**.

# References

1. Analysis of Semi-Supervised Learning with the Yarowsky Algorithm by GholamReza Haffari and Anoop Sarkar School of Computing Science Simon Fraser University 2007
2. Blum, A., Mitchell, T.: Combining labeled and unlabeled data with co-training. In: Proceedings of the 11th Annual Conference on Computational Learning Theory, New York, NY, pp. 92–100 (1998)[Google Scholar](https://scholar.google.com/scholar?q=Blum%2C%20A.%2C%20Mitchell%2C%20T.%3A%20Combining%20labeled%20and%20unlabeled%20data%20with%20co-training.%20In%3A%20Proceedings%20of%20the%2011th%20Annual%20Conference%20on%20Computational%20Learning%20Theory%2C%20New%20York%2C%20NY%2C%20pp.%2092%E2%80%93100%20%281998%29)
3. Li, Ming &amp; Zhou, Zhi-Hua. (2005). SETRED: Self-training with editing. 3518. 611-621. 10.1007/11430919\_71.[https://www.researchgate.net/publication/220895380\_SETRED\_Self-training\_with\_editing](https://www.researchgate.net/publication/220895380_SETRED_Self-training_with_editing)
4. Wang, Yu &amp; Xu, Xiaoyan &amp; Zhao, Haifeng &amp; Hua, Zhongsheng. (2010). Semi-supervised learning based on nearest neighbor rule and cut edges. Knowledge-Based Systems. 23. 547-554. 10.1016/j.knosys.2010.03.012.
5. Tri-Training: Exploiting Unlabeled Data Using Three Classifiers Zhi-Hua Zhou, Member, IEEE and Ming L
6. Yan Zhou and Sally Goldman._Democratic co-learning._
 In IEEE 16th International Conference on Tools with Artificial Intelligence (ICTAI), pages 594-602. IEEE, Nov 2004. doi: 10.1109/ICTAI.2004.48.
7. [https://cran.r-project.org/web/packages/ssc/vignettes/ssc.pdf](https://cran.r-project.org/web/packages/ssc/vignettes/ssc.pdf)
8. [https://towardsdatascience.com/deep-learning-in-winonsin-breast-cancer-diagnosis-6bab13838abd](https://towardsdatascience.com/deep-learning-in-winonsin-breast-cancer-diagnosis-6bab13838abd)
9. [https://www.freecodecamp.org/news/how-to-program-a-neural-network-to-predict-breast-cancer-in-only-5-minutes-23289d62a4c1/](https://www.freecodecamp.org/news/how-to-program-a-neural-network-to-predict-breast-cancer-in-only-5-minutes-23289d62a4c1/)
10. [https://towardsdatascience.com/understanding-the-different-types-of-machine-learning-models-9c47350bb68a](https://towardsdatascience.com/understanding-the-different-types-of-machine-learning-models-9c47350bb68a)
