# Breast-Cancer-DataMining


# ABSTRACT

_This course projects includes analysis on Breast Cancer Prediction using Data Mining based on Breast Cancer Wisconsin (Diagnostic) Dataset from UCI repository. The major goal of this course project is to experiment data mining techniques covered under CSC 7810 - Data Mining: Algorithms and Applications and work towards implementation of those knowledge to develop a near to perfect model in predicting Breast Cancer. The more accurate the model are, more chances of artificial systems to predict if the person is having Breast Cancer. Hence the main outline of this project lies on studying existing literature, draw comparison between Unsupervised, Supervised &amp; Semi-Supervised Learning. This course work also covers novel techniques of predicting Breast Cancer using Semi-Supervised Learning. Additionally, experiments are also performed in Neural Networks to build a robust model implementing backpropagation algorithm using R interface to Keras framework &amp; TensorFlow backend engine and comparisons are drawn using multiple hidden layers and activation functions._


## _ **Overview** _

Breast Cancer is a group of disease in which cells in breast tissue change and divide uncontrollably leading to lump or mass. It is the most common cancer diagnosed among women and is the one of the leading causes of death among women after lung cancer in the United States. It is the most common type of cancer which causes 411,000 annual deaths worldwide.

  1.
## _ **Literature Review** _

In multiple literatures, various machine learning models - both supervised and unsupervised models have been suggested to classify Breast Cancer. However, we find till date, most approaches suggested in the literatures differ mostly in the adopted data mining technique and how to deal with the missing attribute values and labels. An important shortcoming that most of these methods share is that they are either designed for big datasets or have not been tested enough to address the challenge of data scarcity, which is often the case for cancer datasets. We take this opportunity to build our model using different approach that will address this challenge


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

# Data Analysis

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The data used for this project was collected by the University of Wisconsin and its composed by the biopsy result of 569 patients in Wisconsin hospital. They describe characteristics of the cell nuclei present in the image. The dataset is created by Dr. Wolberg a physician at University of Wisconsin and can be found at UCI repository [[Web Link](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))].

**Number of instances: 569**
**Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)**

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
