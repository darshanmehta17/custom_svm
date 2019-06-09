# Custom SVM
A custom implementation of [SVM](models/svm.py) for classification with support for Gaussian RBF kernel, Polynomial kernel and Linear kernel. Uses Fast Gradient Descent algorithm to minimize smoothed hinge loss.

Package includes a wrapper class [OneVsOneClassifier](models/multiclass.py) for performing multiclass classification on the SVM.


### Kernels
##### Gaussian RBF Kernel
The Gaussian Radial Basis Function is given by
```
$$k(x,y) = \exp\left(-\frac{1}{2\sigma^2}\|x-y\|^2\right)$$
```
where $\sigma$ is the parameter called the ```sigma``` which needs to be set. Default value of ```sigma``` is set to 0.5.

##### Polynomial Kernel
The Polynomial kernel is given by
```
$$k(x,y) = (x^{T}y + b)^{p}$$
```
where $b$ is the parameter ```bias``` and $p$ is the parameter ```power``` which needs to be set. Default value of ```bias``` is set to 1 and default value of ```power``` is set to 2.

##### Linear Kernel
The Linear  kernel is given by
```
$$k(x,y) = (x^{T}y)$$
```


### Demos
For a [demo](demo_simulated.py) of SVM on a simple simulated dataset (generated using the scikit-learn library):
```
python demo_simulated.py
```

For a [demo](demo_digits.py) of SVM on a real-world dataset (Digits dataset from the scikit-learn library):
```
python demo_digits.py
```

For a [demo](demo_compare_digits.py) of the comparison of the custom implementation of SVM vs. the scikit-learn implementation on a real-world dataset (Digits dataset from the scikit-learn library):
```
python demo_compare_digits.py
```

For a [demo](demo_vowel.py) of SVM on a real-world multiclass dataset (Vowel dataset from the book Elements of Statistical Learning) using the one vs. one multiclass classification strategy:
```
python demo_vowel.py
```


### Usage
```python
from models.svm import CustomSVM
clf = CustomSVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
```


#### Dependencies
- Python 3
- joblib
- scikit-learn
- numpy
- pandas
- matplotlib
- tqdm
- scipy