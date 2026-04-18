= Problem 10.45
== Part A
#image("10_45/fig_a_original.png", width: 70%)
== Part B
#image("10_45/fig_b_singular_values.png", width: 70%)
== Part C
#image("10_45/fig_c_outer_products.png", width: 70%)
== Part D
#image("10_45/fig_d_approx_1to4.png", width: 70%)
== Part E
#image("10_45/fig_e_approx_10to40.png", width: 70%)
== Part F
#image("10_45/fig_f_residual.png", width: 70%)

= Problem 10.54
== Part A
#image("10_54/fig_a_singular_values.png", width: 70%)
The decay is slow so the variance is spread across many dimensions rather than concentrated in a few.
== Part B
#image("10_54/fig_b_energy.png", width: 70%)
The energy captured by 3 dimensions is 0.3020. The dimensions required to capture 95% of the energy are 73. This matches example 10.6 in the book.
== Part C
#image("10_54/fig_c_rank3.png", width: 90%)
== Part D
#image("10_54/fig_d_pca3d.png", width: 90%)
== Part E
#image("10_54/fig_e_energy_curves.png", width: 90%)
We can observe that tissue type (lung vs spleen) matters more than strain type (Schu4 vs LVS) for the complexity. The lung groups cluster together and the spleen groups are separated from them. Faster-rising energy curve means its gene expression is more low-rank.

= Problem 12.6
== Part A
The canonical correlation is $z^* approx 0.98$.
== Part B
The minimum principal angle is $theta_min approx 0.18$.
== Part C
#image("12_6/fig_c_scatter.png", width: 70%)
This shows a strong correlation, which agrees with $z^* approx 0.98$ obtained in part A.
== Part D
#image("12_6/fig_d_images.png", width: 70%)
Both pictures look similar which matches the fact that $z^*$ is close to 1. $alpha$ does look more like a cat and $beta$ looks more like a dog which makes sense.
== Part E
#image("12_6/fig_e_heatmap.png", width: 70%)
// TODO: NOT SURE THIS MAKES SENSE, ASK KIRBY.
The dark parts of the image correspond to places where we have low correlation, in this case the ears, eyes and nose; which is where cats and dogs look more different. The background is hot probably because it is the same in most pictures.
== Part F
#image("12_6/fig_f_ccvectors.png", width: 100%)
Here we see that the weights are spread fairly evenly across all 99 images, i.e. No image or small number of images dominate.

= Problem 12.7
== Part A
$z^* approx 0.89$
== Part B
#image("12_7/fig_b_scatter.png", width: 100%)
The points cluster fairly tightly around the diagonal line, consistent with $z^* approx 0.89$.

The points are more spread on the negative of the diagonal side than the positive side of the diagonal, this means that some corn pixels have very different spectral signatures from woods in the negative direction of the canonical variable
== Part C
#image("12_7/fig_c_ccvectors.png", width: 100%)
The higher wavelength indices (roughly bands 100-200) are more important for determining the correlation between corn and woods than the lower bands.

= Problem 12.14
#image("12_14/fig_flag_mean.png", width: 100%)
We recover the dog which is the common vector in all subspaces. This is the same we observe in example 12.9 of the notes.

= Problem 13.1
== Part A
See the code for this problem. The output is:
$
  C = mat(
    0.8605, 0;
    0, 0.6045
  )
$

$
  S = mat(
    0, 0;
    0, 0;
    0.5095, 0;
    0, 0.7966
  )
$

$
  G = mat(
    -2.2979, 2.3916;
    -3.2579, -0.6215
  )
$

In this case $m=5,p=4,n=2$ (dimensions).

$k="rank"([A;B])=n=2$ (full rank case).

$r=0$ (no entries equal to 1 in matrix $C$).

$s = 2$ (both diagonal entries of $C$ are strictly between 0 and 1).
== Part B
Since $r = 0$, $s = 2$, $k = 2$, the block matrices are:

- $bold(I)_r = bold(I)_0$: empty (does not appear)
- $tilde(bold(C))_s = op("diag")(0.8605, 0.6045)$: the full $bold(C)$ matrix
- $tilde(bold(S))_s = op("diag")(0.5095, 0.7966)$: the nonzero block of $bold(S)$
- $bold(I)_(k-r-s) = bold(I)_0$: empty
- $bold(O)_(m-r-s, k-r-s) = bold(O)_(3,0)$: empty
== Part C
The generalized singular value pairs $(c_i, s_i)$ are:

$
  (c_1, s_1) = (0.8605, 0.5095)
$
$
  (c_2, s_2) = (0.6045, 0.7966)
$
== Part D
We verify Equation (13.5), i.e., $c_i^2 + s_i^2 = 1$ for $i = 1, 2$:

$
  c_1^2 + s_1^2 approx 0.8605^2 + 0.5095^2 approx 0.7405 + 0.2596 approx 1.0000
$

$
  c_2^2 + s_2^2 approx 0.6045^2 + 0.7966^2 approx 0.3654 + 0.6346 approx 1.0000
$

Both pairs satisfy the constraint, confirming Equation (13.5).

= Problem 13.10
== Part A
See code for this part.
== Part B
#image("13_10/fig_gsvd_pairs.png", width: 70%)
This means that for $i<134$ we have more signal than noise and for $i>134$ we have more noise than signal.
== Part C
#image("13_10/fig_band25_denoised.png", width: 90%)
The denoised image looks very similar, only a tiny bit smoother.

= Problem 13.16

== Estimated signals $T_1$
#image("13_16/fig_T1.png", width: 80%)
The four columns of $hat(U)$ are clean sinusoids of increasing frequency, which matches the structure of the trig dataset.

== Comparison with SVD of $A$
#image("13_16/fig_T1_vs_SVD.png", width: 100%)
Both decompositions produce sinusoidal columns, but the ordering differs. The GSVD orders by _smoothness_: it simultaneously diagonalizes $A^T A$ and $B^T B$, so the ratio $s_i\/c_i$ increases with frequency, placing the lowest-frequency sinusoid first. The SVD of $A$  is not ordered by  frequency. Also in this example the SVD doesn't recover the source signals.

= Problem 13.18
Writing $D_1 Q = Q D_2$ entry-wise, the $(i,j)$ entry gives

$
  (d_i - e_j) Q_(i j) = 0
$

where $d_i, e_j$ are the diagonal entries of $D_1, D_2$ respectively. Since both matrices have distinct entries in increasing order, $d_i = e_j$ implies $i = j$, so for $i != j$ we have $d_i - e_j != 0$ and therefore $Q_(i j) = 0$. Thus $Q$ is diagonal. Since $Q$ is orthogonal, $Q^T Q = I$ forces $Q_(i i)^2 = 1$, giving $Q = op("diag")(plus.minus 1, dots, plus.minus 1)$. $square$
