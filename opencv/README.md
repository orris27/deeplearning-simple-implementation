OpenCV can be used to train machine learning algorithm on C++.

OpenCV2 code:
```cpp
Mat X_train(train_size, num_features, CV_32F, X_train_f);
Mat y_train(train_size, 1, CV_32F, y_train_f);

Mat var_type = Mat(num_features + 1, 1, CV_8U );
var_type.setTo(Scalar(CV_VAR_NUMERICAL) );
var_type.at<uchar>(num_features, 0) = CV_VAR_ORDERED; // For regression. If the task is classification, we need another option.

CvRTParams params(10, // max depth
                  2, // min sample count
                  0.0001f, // regression accuracy. Small value here for regression
                  false, // 
                  15, // max number of categories (use sub-optimal algorithm for larger numbers)
                  0, // the array of priors, the bigger p_weight, the more attention
                  false,// calc_var_importance
                  //num_features, // nactive_vars
                  4, // nactive_vars
                  40, // max number of trees in the forest
                  //0.0, // forest accuracy
                  0.01f, // forest accuracy
                  //CV_TERMCRIT_ITER // termcrit_type
                  CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termcrit_type
                  );

// Extra Trees
CvERTrees *model = new CvERTrees;

// Random Forest
//CvRTrees *model = new CvRTrees;


model->train(X_train, CV_ROW_SAMPLE, y_train, Mat(), Mat(), var_type, Mat(), params);

Mat X_test(1, num_features, CV_32F, X_test_f[0]);
y_predicted = model->predict(X_test, Mat());
y_true = y_test_f[i];
```
OpenCV3 code:
```cpp
Ptr<RTrees> model = RTrees::create();

model->train(X_train, ROW_SAMPLE, y_train);

float sum_error = 0.0;
float y_predicted_x, y_predicted_y, y_true_x, y_true_y;

Mat X_test(1, num_features, CV_32F, X_test_f[0]);

y_predicted = model->predict(X_test);
y_true = y_test_f[i];
```

Both codes share the same header:
```cpp
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;
```
For data structure, `X_train_f`, `y_train_f`, `X_test_f` and `y_test_f` are `float *`. Note that if `X` is a 2d matrix, then each element should be stored in a continuous memory, i.e., we cannot use `float **` to represent `X` since `X[0]` and `X[1]` point to 2 separate areas. `train_size` is the number of samples in training data. `num_features` is the size of feature in one sample.
