= Problem 1
== Part a
See code.
== Part b
These are the result I get:
```
Best C: 2.166666666666667  (val accuracy: 0.9882)
Test accuracy: 0.9651
Nonzero weights: 14/30
```
== Part c
#image("weights.png")
In the picture above seems like we can keep 10 features and discard 20 but we can also plot the weights in log scale to get the following image:
#image("weights_log.png")
Here we can see that we can discard 18 features and keep 12. This is using a threshold of 0.001.
== Part d
With the selected features (weight greater than 0.001) we get an accuracy of 96.5%.
== Part e

Confusion Matrix:
```
                 | Pred Malignant  Pred Benign
-----------------+-----------------------------
Actual Malignant |       52             0
Actual Benign    |       3             31
```
= Problem 2
See code.

Using all 30 features, Fisher DA achieves 96.51% test accuracy. Using only the 12 features selected by the SVM, it achieves 97.67% test accuracy. Slightly better, confirming that the sparse SVM feature selection is effective.

Note: For this problem I used AI to generate the code.
= Problem 3
== Part a
The image (1600×1200) yields 30,000 non-overlapping 8×8 patches.
== Part b
Dictionary initialized with K=256 atoms drawn randomly from the flattened patches.
== Part c
#image("ksvd_error.png")
== Part d
#image("ksvd_atoms.png")
== Part e
#image("ksvd_reconstruction.png")

Note: For this problem I used AI to generate the code.
