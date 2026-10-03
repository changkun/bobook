---
status: done
synopsis: "Vectors, matrices as maps of space, positive definite matrices, eigenvectors, the Cholesky factorization, determinants, and block matrices: the linear algebra a Gaussian process needs, with a picture for each idea and a covariance matrix in three dimensions to show what two dimensions hide."
sources: ["textbooks", "Strang 2016", "Golub and Van Loan 2013", "Rasmussen and Williams 2006, app. A"]
---

# The Linear Algebra of Uncertainty {#sec-linear-algebra}

@sec-prob-covariance introduced a table: the covariances between every pair of
uncertain quantities, the covariance matrix. A Bayesian optimizer keeps such a
table for the objective's values at every input it has evaluated and every
input it is considering, so after 50 evaluations the table has 50 rows at the
least. Each update of its belief,
each prediction, and each fit of the model's own settings is a computation
with that table. This chapter is about those computations.

Linear algebra is a large subject, and this chapter takes only the part that
the rest of the book uses. It treats a matrix as a map that moves the points of
space, because that picture explains the properties that matter: which tables
can be covariance matrices, what shape a covariance gives a cloud of
uncertainty, how to solve the equations of a Gaussian process without ever
inverting a matrix, and what a determinant measures. Two-dimensional pictures
carry most of the intuition, but a few facts appear only from three dimensions
on, so the chapter also turns a three-dimensional covariance in your hands.

The payoff comes in the next chapter, @sec-gaussian, where every result here
becomes a statement about the Gaussian distribution, and in
@sec-gp-regression, where those statements become an algorithm.

## Vectors {#sec-vectors}

A *vector* is an ordered list of numbers. Bold lowercase letters name vectors,
and by convention a vector is a column:

$$
\vx = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \R^d.
$$

The symbol $\R^d$ is the set of all such lists of $d$ real numbers, and $d$ is
the vector's *dimension*. Writing $\vx^\T = (x_1, \dots, x_d)$, read "$\vx$
transpose", turns the column into a row.

Vectors appear in this book in two roles, and keeping them apart prevents a
common confusion. In the first role a vector is a point in the space being
searched: a configuration of hyperparameters, the settings of an exoskeleton,
the parameters of a photo filter. @snoek2012practical, for instance, tuned nine
settings of an image classifier's training procedure at once, so each
configuration they tried was a vector in $\R^9$ (@sec-ard returns to what they
found). In the second role a vector holds the values of the objective at a
list of inputs,
$\vf = (f(\vx_1), \dots, f(\vx_n))^\T$. Its dimension is the number of inputs,
not the number of hyperparameters, and it grows with every evaluation. The
covariance matrices of a Gaussian process live in this second space.

Vectors add entry by entry, and multiplying a vector by a number multiplies
every entry. Geometrically, $\vx + \vy$ places the arrow of $\vy$ at the tip of
the arrow of $\vx$, and $2\vx$ is the same arrow twice as long.

### Inner products, lengths, and angles {#sec-linalg-inner-products}

The *inner product*, or dot product, of two vectors of the same dimension
multiplies them entry by entry and adds the results:

$$
\vx^\T\vy = \sum_{i=1}^d x_i y_i.
$$ {#eq-linalg-inner}

The notation reads as a row times a column, which is the rule for multiplying
matrices introduced in @sec-matrices. From the inner product come length and
distance. The *norm* of a vector, its length, is
$\lVert\vx\rVert = \sqrt{\vx^\T\vx}$, Pythagoras's theorem in $d$ dimensions,
and the distance between two points is the norm of their difference,
$\lVert\vx - \vy\rVert$.

The inner product also measures angle. For any two vectors,

$$
\vx^\T\vy = \lVert\vx\rVert\,\lVert\vy\rVert \cos\theta,
$$

where $\theta$ is the angle between them. Two vectors are *orthogonal*,
perpendicular, when their inner product is zero. Dividing by the two lengths
gives $\cos\theta$, which is 1 for vectors pointing the same way, 0 for
perpendicular ones, and $-1$ for opposite ones. The correlation of
@sec-prob-covariance is this cosine, computed for centered random
variables instead of lists of numbers.

Distance matters to Bayesian optimization because the most common kernels,
the functions that decide how strongly the objective's values at two inputs
are correlated (@sec-kernels), depend on the inputs only through the distance
between them. The RBF kernel,
$k(\vx, \vx') = \exp\!\left(-\lVert\vx - \vx'\rVert^2 / 2\ell^2\right)$, says
that nearby inputs have similar values and distant ones are unrelated, with
the lengthscale $\ell$ setting what "nearby" means.

### Distances in many dimensions {#sec-linalg-distances}

Our intuition about distance comes from two and three dimensions, and it
misleads in more. Scatter points uniformly at random in the unit cube
$[0, 1]^d$. Each coordinate contributes on average $1/6$ to the squared
distance between two points, so the typical distance grows like
$\sqrt{d/6}$: about 0.58 in two dimensions, 1.0 in six, and 2.9 in fifty.
The spread of the distances does not grow with it. In a simulation with 200
random points, the standard deviation of the pairwise distances stayed near
0.24 from two dimensions to a hundred. As a result, the ratio of each point's
nearest neighbor distance to its farthest neighbor distance, which was 0.04 for
a typical point in two dimensions, rose to 0.22 in six dimensions, 0.65 in
fifty, and 0.74 in a hundred.

In many dimensions, then, every point is far from every other, and all of
them are about equally far. A kernel with a lengthscale suited to a
two-dimensional problem would declare every pair of points in a
fifty-dimensional one unrelated, and the model would learn nothing from one
evaluation about any other. This is one root of the difficulty of Bayesian
optimization in many dimensions, and the remedies, such as lengthscales that
grow with the dimension and a lengthscale for each input (@sec-ard), start from
it (@sec-high-dimensions).

## Matrices as transformations {#sec-matrices}

A *matrix* is a rectangular table of numbers, written with a bold capital. An
$m \times n$ matrix $\mA$ has $m$ rows and $n$ columns, and $A_{ij}$ is the
entry in row $i$ and column $j$. The table is the matrix's storage format. Its
meaning is the map it defines.

### The product of a matrix and a vector {#sec-linalg-matvec}

Multiplying an $m \times n$ matrix by a vector of dimension $n$ gives a vector
of dimension $m$, whose $i$-th entry is the inner product of row $i$ with the
vector:

$$
(\mA\vx)_i = \sum_{j=1}^n A_{ij} x_j.
$$

The same product has a second reading, which is the one to picture. Write the
columns of $\mA$ as $\mathbf{a}_1, \dots, \mathbf{a}_n$. Then

$$
\mA\vx = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n.
$$ {#eq-linalg-columns}

The vector $\mathbf{e}_1 = (1, 0, \dots, 0)^\T$ picks out the first column:
$\mA\mathbf{e}_1 = \mathbf{a}_1$. So the columns of a matrix record where it
sends the coordinate directions, and @eq-linalg-columns says that every other
vector goes where its coordinates say: as much of $\mathbf{a}_1$ as $\vx$ had of
$\mathbf{e}_1$, and so on.

A map of this kind is *linear*: it sends sums to sums and multiples to
multiples, $\mA(\vx + \vy) = \mA\vx + \mA\vy$ and $\mA(c\vx) = c\,\mA\vx$. In the
plane, a linear map sends straight lines to straight lines, keeps the origin in
place, and turns a square grid into a grid of parallelograms. The unit circle,
the set of vectors of length one, becomes an ellipse.

```{figure}
//| figure: linalg-transform
//| label: fig-linalg-transform
//| fig-cap: "A 2×2 matrix as a map of the plane. The magenta and green arrows are the columns of $\mA$, the images of the two coordinate directions; drag their tips to change the matrix. The dashed circle and square are the unit circle and unit square, the blue ellipse and yellow parallelogram are their images, and the faint blue lines are the image of the grid. The black arrows are a vector $\vx$ on the unit circle and its image $\mA\vx$; drag $\vx$ around the circle. The readouts give the determinant and the eigenvalues, which @sec-eigen and @sec-determinants explain."
```

Some things to try:

- **Drag the tip of $\mA\mathbf{e}_1$.** Only the first column changes, and the
  whole grid follows: every point moves by its first coordinate times the
  change.
- **Choose the Rotation, Shear, and Stretch presets.** A rotation turns the
  circle without changing its shape; a stretch scales the two axes by
  different amounts; a shear slides one axis along the other and turns the
  square into a slanted parallelogram of the same area.
- **Choose Singular.** The two columns point the same way, the ellipse
  collapses to a line segment, and the whole plane is flattened onto a line.
  Every point on that line came from infinitely many points, so the map cannot
  be undone.

### Composition, identity, and inverse {#sec-linalg-composition}

Applying $\mathbf{B}$ and then $\mA$ is again a linear map, and its matrix is
the product $\mA\mathbf{B}$, whose columns are $\mA$ applied to the columns of
$\mathbf{B}$. Entry by entry, $(\mA\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$, the
inner product of row $i$ of $\mA$ with column $j$ of $\mathbf{B}$. Order
matters: stretching and then rotating is not the same as rotating and then
stretching, so in general $\mA\mathbf{B} \ne \mathbf{B}\mA$. Multiplying two
$n \times n$ matrices this way takes $n^3$ multiplications, one for each
combination of $i$, $j$, and $k$.

The *identity* matrix $\mI$, with ones on the diagonal and zeros elsewhere,
leaves every vector where it is. A square matrix $\mA$ is *invertible* when
another matrix $\mA^{-1}$ undoes it, $\mA^{-1}\mA = \mA\mA^{-1} = \mI$. A
matrix is invertible exactly when it does not flatten space: when no nonzero
vector is sent to zero, so no two points land on the same image. The Singular
preset of @fig-linalg-transform is the opposite case. Inverting a product
reverses the order, $(\mA\mathbf{B})^{-1} = \mathbf{B}^{-1}\mA^{-1}$, since the
last map applied must be the first one undone.

### Transpose {#sec-linalg-transpose}

The *transpose* $\mA^\T$ flips a matrix across its diagonal, so that
$(\mA^\T)_{ij} = A_{ji}$ and an $m \times n$ matrix becomes $n \times m$. A
column vector is an $n \times 1$ matrix, its transpose a $1 \times n$ row, and
the inner product $\vx^\T\vy$ is the product of a row with a column. The
reverse product $\vx\vy^\T$, a column times a row, is an $n \times n$ matrix
called the *outer product*. Transposing a product also reverses the order,
$(\mA\mathbf{B})^\T = \mathbf{B}^\T\mA^\T$.

The shapes in @sec-gp-regression follow these rules. With $n$ observed inputs
and $m$ inputs to predict at, the kernel matrix $\mK$ among observed inputs is
$n \times n$, the matrix $\mK_*$ between observed and new inputs is $n \times
m$, and the posterior mean $\mK_*^\T\mK^{-1}\vy$ multiplies an $m \times n$
matrix, an $n \times n$ matrix, and an $n$-vector to give one prediction for
each of the $m$ new inputs.

## Symmetric and positive definite matrices {#sec-positive-definite}

Not every square table of numbers can be a covariance matrix. This section
finds the condition, which turns out to be geometric: a covariance matrix must
not let any direction have negative variance.

### Quadratic forms {#sec-linalg-quadratic-forms}

A matrix is *symmetric* when it equals its transpose, $A_{ij} = A_{ji}$. A
covariance matrix is symmetric because $\Cov[x_i, x_j] = \Cov[x_j, x_i]$, and
so is a kernel matrix, because $k(\vx, \vx') = k(\vx', \vx)$.

For a symmetric matrix $\mA$, the expression $\vw^\T\mA\vw$ is a single number
for each vector $\vw$, called a *quadratic form*. In two dimensions, with
$\mA = \begin{bmatrix} a & b \\ b & d \end{bmatrix}$,

$$
\vw^\T\mA\vw = a w_1^2 + 2b\, w_1 w_2 + d\, w_2^2,
$$

a quadratic polynomial in the entries of $\vw$. Quadratic forms appear
whenever a covariance matrix meets a weighted sum, because of the following
identity.

::: {.derivation title="The variance of a weighted sum"}
Let $\vx$ be a random vector with mean $\vmu$ and covariance matrix $\mSigma$,
and let $\vw$ be a fixed vector of weights.

1. The weighted sum is $\vw^\T\vx = \sum_i w_i x_i$, with mean
   $\vw^\T\vmu$ by linearity of expectation (@eq-prob-linearity).
2. By the definition of variance (@eq-prob-variance),
   $\Var[\vw^\T\vx] = \E\big[(\sum_i w_i (x_i - \mu_i))^2\big]$.
3. Expanding the square of the sum gives every pairwise product:
   $\E\big[\sum_i \sum_j w_i w_j (x_i - \mu_i)(x_j - \mu_j)\big]$.
4. By linearity of expectation, this is
   $\sum_i \sum_j w_i w_j\, \E[(x_i - \mu_i)(x_j - \mu_j)]
   = \sum_i \sum_j w_i w_j \Sigma_{ij}$, by the definition of covariance.
5. The double sum is the quadratic form, so
   $\Var[\vw^\T\vx] = \vw^\T\mSigma\vw$.
:::

### Every direction has nonnegative variance {#sec-linalg-pd-definition}

A variance cannot be negative. By the derivation, a covariance matrix must
therefore satisfy $\vw^\T\mSigma\vw \ge 0$ for every $\vw$. This property has a
name.

::: {.definition #def-linalg-pd title="Positive definite and semidefinite matrices"}
A symmetric matrix $\mA$ is *positive semidefinite* if $\vw^\T\mA\vw \ge 0$ for
every vector $\vw$, and *positive definite* if $\vw^\T\mA\vw > 0$ for every
nonzero $\vw$.
:::

Read with $\mA = \mSigma$, positive definiteness says that every direction in
the space has positive variance: no weighted combination of the variables is
known exactly. A covariance matrix that is only semidefinite has some
combination with zero variance, a combination that is fixed. That happens in a
Gaussian process when two observed inputs coincide: the values $f(\vx)$ and
$f(\vx')$ are then the same random variable, and $f(\vx) - f(\vx')$ is exactly
zero. Every result below assumes the strict version, which is why
implementations add a little noise or "jitter" to the diagonal
(@sec-gp-computation).

In two dimensions the condition is easy to check. A covariance matrix with
standard deviations $\sigma_1, \sigma_2$ and correlation $\rho$ is positive
definite exactly when $\sigma_1, \sigma_2 > 0$ and $-1 < \rho < 1$, and every
correlation in that range is possible. From three dimensions on, this is no
longer enough.

### Three correlations that cannot coexist {#sec-linalg-impossible}

Suppose three quantities each have variance 1, the first is correlated $0.8$
with the second and $0.8$ with the third, and the second and third are
uncorrelated:

$$
\mSigma = \begin{bmatrix} 1 & 0.8 & 0.8 \\ 0.8 & 1 & 0 \\ 0.8 & 0 & 1 \end{bmatrix}.
$$

Each correlation is between $-1$ and $1$, and each pair on its own is a valid
two-dimensional covariance. Yet the weights $\vw = (-2, 1, 1)^\T$ give

$$
\vw^\T\mSigma\vw = 4 + 1 + 1 + 2\,(-2)(0.8) + 2\,(-2)(0.8) + 2\,(1)(0) = -0.4,
$$

a negative variance for the quantity $x_2 + x_3 - 2x_1$. No three random
variables have these correlations. Intuitively, if $x_1$ moves closely with
$x_2$ and closely with $x_3$, then $x_2$ and $x_3$ must move together to some
extent; they cannot be unrelated. @exr-linalg-range finds how strongly they
must be correlated.

This is not a curiosity. In finance, tables of correlations between many
assets can fail to be positive semidefinite, and the problem of repairing such
a table into the nearest valid correlation matrix has a literature of its own
[@higham2002computing]. For a Gaussian process it is the reason a kernel
cannot be an arbitrary similarity score. A natural-looking rule such as "two
inputs are fully correlated if they are closer than 1 and uncorrelated
otherwise" fails on the inputs $0$, $0.6$, and $1.2$: the first two are
fully correlated, the last two are fully correlated, and the first and last
are uncorrelated, an impossible triple of the same kind. The kernels of
@sec-kernels are built so that this can never happen: a valid kernel is one
whose matrices are positive semidefinite for every choice of inputs, and no
other function is (@sec-kernel-trick).

@fig-linalg-ellipsoid in the next section lets you try to build such a matrix
and see what goes wrong.

## Eigenvectors: the axes a matrix stretches {#sec-eigen}

The entries of a matrix say little at a glance about what it does. In
@fig-linalg-transform, most vectors $\vx$ are both turned and stretched by
$\mA$. Some directions, though, are only stretched: the image $\mA\vx$ points
along $\vx$ itself. If we could find those directions, we could describe the
map as a set of independent stretches.

::: {.definition #def-linalg-eigen title="Eigenvector and eigenvalue"}
A nonzero vector $\mathbf{u}$ is an eigenvector of a square matrix $\mA$, with
eigenvalue $\lambda$, if $\mA\mathbf{u} = \lambda\mathbf{u}$.
:::

An eigenvector is a direction the matrix does not turn; its eigenvalue is the
factor by which the direction is stretched, and a negative eigenvalue
reverses it. In @fig-linalg-transform, drag $\vx$ around the circle until
$\vx$ and $\mA\vx$ line up, and the readout reports the eigenvalue. For the
default matrix this happens near $18°$, with eigenvalue 1.4, and near
$135°$, with eigenvalue 0.6. Turn on *Show eigenvectors* to see both
directions. The Rotation preset has no real eigenvectors at all, because it
turns every direction.

### Symmetric matrices {#sec-linalg-spectral}

For a general matrix the eigenvectors need not be perpendicular, as the default
matrix shows. Symmetric matrices, the only kind a covariance can be, are much
better behaved.

::: {.theorem #thm-linalg-spectral title="Spectral theorem for symmetric matrices"}
A symmetric $d \times d$ matrix $\mA$ has $d$ real eigenvalues
$\lambda_1, \dots, \lambda_d$ and an orthonormal set of eigenvectors
$\mathbf{u}_1, \dots, \mathbf{u}_d$: unit vectors that are pairwise
perpendicular. Collecting the eigenvectors as the columns of a matrix
$\mathbf{U}$ and the eigenvalues on the diagonal of a matrix $\bm{\Lambda}$,

$$
\mA = \mathbf{U}\bm{\Lambda}\mathbf{U}^\T = \sum_{i=1}^d \lambda_i\, \mathbf{u}_i\mathbf{u}_i^\T.
$$
:::

The proof that the eigenvalues are real takes more space than it deserves here
[@strang2016introduction], but the perpendicularity is short.

::: {.derivation title="Eigenvectors of a symmetric matrix are perpendicular"}
Let $\mA\mathbf{u} = \lambda\mathbf{u}$ and $\mA\mathbf{v} = \mu\mathbf{v}$ with
$\lambda \ne \mu$.

1. Multiply the first equation on the left by $\mathbf{v}^\T$:
   $\mathbf{v}^\T\mA\mathbf{u} = \lambda\, \mathbf{v}^\T\mathbf{u}$.
2. Transpose the left side, a single number, which leaves it unchanged:
   $\mathbf{v}^\T\mA\mathbf{u} = \mathbf{u}^\T\mA^\T\mathbf{v} = \mathbf{u}^\T\mA\mathbf{v}$,
   using $\mA^\T = \mA$.
3. By the second equation, $\mathbf{u}^\T\mA\mathbf{v} = \mu\, \mathbf{u}^\T\mathbf{v}$.
4. So $\lambda\, \mathbf{v}^\T\mathbf{u} = \mu\, \mathbf{u}^\T\mathbf{v}$, and
   since $\mathbf{u}^\T\mathbf{v} = \mathbf{v}^\T\mathbf{u}$,
   $(\lambda - \mu)\, \mathbf{u}^\T\mathbf{v} = 0$.
5. Because $\lambda \ne \mu$, $\mathbf{u}^\T\mathbf{v} = 0$.
:::

The decomposition reads from right to left as a recipe. $\mathbf{U}^\T$ turns
space so that the eigenvectors line up with the coordinate axes, $\bm{\Lambda}$
stretches each axis by its eigenvalue, and $\mathbf{U}$ turns space back. A
symmetric matrix is a set of perpendicular stretches, nothing more. Choose the
Symmetric preset in @fig-linalg-transform and turn on the eigenvectors: they
are perpendicular, and they are the axes of the ellipse.

### Eigenvalues and positive definiteness {#sec-linalg-eigen-pd}

The spectral theorem turns the definition of positive definiteness into a
condition on the eigenvalues.

::: {.derivation title="Positive definite means positive eigenvalues"}
Let $\mA = \mathbf{U}\bm{\Lambda}\mathbf{U}^\T$ be symmetric, and let $\vw$ be
any vector.

1. Substitute the decomposition:
   $\vw^\T\mA\vw = \vw^\T\mathbf{U}\bm{\Lambda}\mathbf{U}^\T\vw$.
2. Write $\mathbf{c} = \mathbf{U}^\T\vw$, the coordinates of $\vw$ along the
   eigenvectors, $c_i = \mathbf{u}_i^\T\vw$. Then
   $\vw^\T\mA\vw = \mathbf{c}^\T\bm{\Lambda}\mathbf{c} = \sum_i \lambda_i c_i^2$.
3. If every $\lambda_i > 0$, the sum is positive whenever some $c_i \ne 0$,
   which holds for every nonzero $\vw$ because $\mathbf{U}$ is invertible.
4. Conversely, choosing $\vw = \mathbf{u}_j$ gives $c_j = 1$ and every other
   $c_i = 0$, so $\vw^\T\mA\vw = \lambda_j$, which must be positive.
:::

For a covariance matrix the derivation says more. Take $\vw$ of length one, a
direction. Its coordinates along the eigenvectors then satisfy
$\sum_i c_i^2 = 1$, so the variance in direction $\vw$ is a weighted average
of the eigenvalues, with weights $c_i^2$. The eigenvalues are the variances
along the eigenvectors, and every other direction's variance lies between the
smallest and the largest. The eigenvector with the largest eigenvalue is the
direction in which the uncertainty is widest. The shape of the uncertainty is an ellipsoid whose axes
point along the eigenvectors, with half-lengths $\sqrt{\lambda_i}$ (the
standard deviations along the axes), which @sec-gauss-shape derives for the
Gaussian.

### A covariance in three dimensions {#sec-linalg-ellipsoid}

In two dimensions the ellipse summarizes everything. In three, two things
appear that the plane cannot show. @fig-linalg-ellipsoid draws the ellipsoid of
a three-dimensional covariance, $\{\vx : \vx^\T\mSigma^{-1}\vx = 1\}$, the set
of points one standard deviation from the center in the sense that
@sec-gauss-shape makes precise, together with its three axes and its shadows on
the walls of the cube. Beside it are the three pairwise views, the ellipses of
the $2 \times 2$ blocks of $\mSigma$ that involve two of the three
coordinates.

```{figure}
//| figure: linalg-ellipsoid
//| label: fig-linalg-ellipsoid
//| fig-cap: "A 3×3 covariance matrix as an ellipsoid. The sliders set three correlations and three standard deviations; drag the cube to turn it. The black lines are the eigenvector axes with half-lengths $\sqrt{\lambda_i}$, the gray shapes on the back walls are the ellipsoid's shadows, and the three panels beside the cube are the pairwise views, which match the shadows. After *An impossible triple*, each pairwise view is still a valid ellipse, but the matrix is not positive definite: no ellipsoid exists, one eigenvalue is negative (the red direction), and the Cholesky factorization of @sec-cholesky fails. *Show samples* draws points $\mL\vz$ from a vector $\vz$ of independent random numbers with mean 0 and variance 1. The values are illustrative."
```

Some things to try:

- **Turn the cube.** The ellipsoid has three perpendicular axes, the
  eigenvectors, and from most angles none of them is aligned with a coordinate
  axis. The readout lists the eigenvalues; their square roots are the axes'
  half-lengths.
- **Compare the shadows with the pairwise views.** They are the same ellipses.
  The shadow of a covariance's ellipsoid on the plane of two coordinates is the
  ellipse of the corresponding $2 \times 2$ block, which is why ignoring a
  variable of a Gaussian amounts to deleting its row and column
  (@sec-gaussian-marginal).
- **Press An impossible triple.** The pairwise views show three valid
  ellipses, with correlations 0.8, 0.8, and 0. The ellipsoid vanishes, and the
  red dashed line marks the direction whose variance would be negative, the
  eigenvector of the eigenvalue $-0.13$. Now raise $\rho_{23}$ slowly: the
  ellipsoid reappears once $\rho_{23}$ passes 0.28 (at 0.29 on the slider), as
  a flat disk that thickens as $\rho_{23}$ grows.
- **Press A chain.** Here $x_1$ is linked to $x_2$, and $x_2$ to $x_3$, with
  correlation 0.8 each, and $\rho_{13} = 0.64$ is the product of the two. The
  ellipsoid becomes a cigar along the diagonal, with one eigenvalue much larger
  than the others: most of the uncertainty lies along a single direction.

The first lesson is that pairwise checks do not certify a covariance matrix;
its validity is a property of the whole table. The second is that a covariance
can be nearly flat in a direction that no single variable reveals: none of the
pairwise views of a nearly singular matrix need look degenerate.

### The eigenvalues of a kernel matrix {#sec-linalg-kernel-spectrum}

That second lesson is the everyday situation of a Gaussian process. The
kernel matrix of 100 inputs spaced evenly in $[0, 1]$, under an RBF kernel with
lengthscale 0.1, is a $100 \times 100$ positive definite matrix in exact
arithmetic. Its largest eigenvalues are 23.9, 21.2, and 17.5, but they decay so
quickly that only 28 of the 100 exceed $10^{-10}$. In double-precision
arithmetic the smallest ones are lost in rounding error; in our run with NumPy
the smallest computed eigenvalue even came out slightly negative,
$-4 \times 10^{-15}$. The function values at nearby inputs are so strongly
correlated that most directions in the 100-dimensional space have almost no
variance. In the same run, NumPy's Cholesky factorization of this matrix
failed; adding
$10^{-6}$ to the diagonal, the jitter of @sec-gp-computation, makes it succeed.
A smooth prior over functions has, in effect, far fewer than 100 independent
directions of uncertainty, which is part of why a Gaussian process can learn a
smooth function from a few evaluations.

The ratio of the largest to the smallest eigenvalue, the *condition number*,
measures how close to singular a matrix is. Double-precision numbers carry
about 16 significant decimal digits, and solving a system with a matrix of
condition number $10^k$ can lose about $k$ of them [@golub2013matrix]. Kernel
matrices of smooth kernels on closely spaced inputs routinely reach condition
numbers near $10^{16}$, where no digits remain, so jitter or observation noise
is a numerical necessity, not a modeling choice.

Eigenvalues of exactly zero carry information too. In @sec-comparison-graph,
the eigenvalues of a matrix built from a set of pairwise comparisons count the
directions in which the comparisons say nothing at all about a person's
preferences.

## Solving systems without inverting {#sec-cholesky}

The formulas of Gaussian process regression are full of inverse matrices: the
posterior mean $\vk(\vx)^\T\mK^{-1}\vy$ and the variance
$k(\vx, \vx) - \vk(\vx)^\T\mK^{-1}\vk(\vx)$ of @eq-gp-pointwise. Read
literally, they say: invert $\mK$, then multiply. Numerical practice says
never to do that. This section explains why, and what to do instead.

### Why not invert {#sec-linalg-why-not-invert}

A formula containing $\mK^{-1}\vy$ never needs $\mK^{-1}$ itself. It needs the
vector $\bm{\alpha}$ that solves the linear system $\mK\bm{\alpha} = \vy$. Two
reasons favor solving over inverting. Forming the inverse costs more, several
times as much as the factorization used below. And the computed inverse is
less accurate: multiplying by a rounded inverse loses more digits than solving
directly with a factorization [@golub2013matrix]. When the matrix is badly
conditioned, as kernel matrices often are, the difference decides whether the
answer is usable.

### Triangular systems {#sec-linalg-triangular}

Some systems are easy. A matrix is *lower triangular* when every entry above
the diagonal is zero. The system $\mL\vz = \mathbf{b}$ with such a matrix can
be solved one entry at a time, top to bottom: the first equation involves only
$z_1$, the second only $z_1$ and $z_2$, and so on.

$$
z_i = \frac{1}{L_{ii}}\Big(b_i - \sum_{k < i} L_{ik} z_k\Big), \qquad i = 1, \dots, n.
$$ {#eq-linalg-forward}

This *forward substitution* costs about $n^2$ operations, against $n^3$ for
general methods. An *upper triangular* system, with zeros below the diagonal,
is solved the same way from the bottom up, by *back substitution*. The whole
strategy for symmetric positive definite systems is to reduce them to two
triangular ones.

### The Cholesky factorization {#sec-linalg-cholesky-def}

::: {.definition #def-linalg-cholesky title="Cholesky factorization"}
Every symmetric positive definite matrix $\mA$ can be written uniquely as

$$
\mA = \mL\mL^\T,
$$ {#eq-linalg-cholesky}

where $\mL$ is lower triangular with positive diagonal entries. $\mL$ is the
*Cholesky factor* of $\mA$.
:::

The factor is a square root of the matrix. For a $1 \times 1$ matrix $[a]$ it is
$[\sqrt{a}]$, which exists exactly when $a > 0$. The $2 \times 2$ case shows how
the general algorithm works and why it needs positive definiteness.

::: {.derivation title="The Cholesky factor of a 2×2 matrix"}
Let $\mA = \begin{bmatrix} a & b \\ b & d \end{bmatrix}$ and look for
$\mL = \begin{bmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{bmatrix}$.

1. Multiplying out,
   $\mL\mL^\T = \begin{bmatrix} l_{11}^2 & l_{11}l_{21} \\ l_{11}l_{21} & l_{21}^2 + l_{22}^2 \end{bmatrix}$.
2. Match the top-left entry: $l_{11}^2 = a$, so $l_{11} = \sqrt{a}$, which needs
   $a > 0$.
3. Match the off-diagonal entry: $l_{11}l_{21} = b$, so $l_{21} = b / \sqrt{a}$.
4. Match the bottom-right entry: $l_{21}^2 + l_{22}^2 = d$, so
   $l_{22} = \sqrt{d - b^2/a}$, which needs $d - b^2/a > 0$.
5. Both conditions hold exactly when $\mA$ is positive definite: $a$ is the
   variance in direction $\mathbf{e}_1$, and $a\,(d - b^2/a) = ad - b^2$ is the
   determinant, the product of the two eigenvalues (@sec-determinants).
:::

The quantity $d - b^2/a$ in step 4 is a first glimpse of the Schur complement
of @sec-block-matrices. For a covariance matrix with $a = \sigma_1^2$,
$b = \rho\sigma_1\sigma_2$, and $d = \sigma_2^2$, it equals
$\sigma_2^2(1 - \rho^2)$: the variance of the second variable that remains once
the first is known, as @sec-gaussian-conditioning will show.

The general algorithm fills in $\mL$ one column at a time, in the same way.

::: {.algorithm #alg-linalg-cholesky title="Cholesky factorization"}
Input: a symmetric $n \times n$ matrix $\mA$.

1. For each column $j = 1, \dots, n$:
2. Compute $s = A_{jj} - \sum_{k < j} L_{jk}^2$. If $s \le 0$, stop: $\mA$ is not
   positive definite.
3. Set $L_{jj} = \sqrt{s}$.
4. For each row $i = j + 1, \dots, n$, set
   $L_{ij} = \big(A_{ij} - \sum_{k < j} L_{ik}L_{jk}\big) / L_{jj}$.
:::

Step 2 makes the algorithm a test as well as a factorization: it succeeds
exactly when the matrix is positive definite, and it is the cheapest such test
in practice. In @fig-linalg-ellipsoid, the impossible triple fails at the third
column, where $s$ comes out negative. The algorithm costs about $n^3/3$
floating-point operations [@golub2013matrix].

### Solving with the factor {#sec-linalg-cholesky-solve}

With $\mA = \mL\mL^\T$, the system $\mA\vx = \mathbf{b}$ splits into two
triangular systems. First solve $\mL\vz = \mathbf{b}$ by forward substitution,
then $\mL^\T\vx = \vz$ by back substitution. Then
$\mA\vx = \mL(\mL^\T\vx) = \mL\vz = \mathbf{b}$, as required. Once $\mL$ is
known, each new right-hand side costs only $O(n^2)$. This is
@alg-gp-regression: one factorization of the kernel matrix, then triangular
solves for the weights $\bm{\alpha}$ and for each predictive variance.

::: {.code title="NumPy and SciPy"}
```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

A = np.array([[4.0, 2.0], [2.0, 3.0]])
b = np.array([2.0, 4.0])

L = np.linalg.cholesky(A)                  # lower triangular, L @ L.T == A
z = solve_triangular(L, b, lower=True)     # forward substitution
x = solve_triangular(L.T, z, lower=False)  # back substitution
print(x)                                   # [-0.25  1.5 ]

c = cho_factor(A)                          # the same, packaged
print(cho_solve(c, b))                     # [-0.25  1.5 ]
```
:::

### The cost, in practice {#sec-linalg-cost}

The cubic cost decides how many observations an exact Gaussian process can
handle. @tbl-linalg-timing gives timings measured with NumPy in double
precision on the laptop used to prepare this chapter (an Apple M5 Pro); other
machines will differ, but the growth rate will not.

::: {.table #tbl-linalg-timing title="Time to factorize and to invert a symmetric positive definite n × n matrix with NumPy, and the memory to store it (8 bytes per entry). Measured once on one laptop; the ratios are the point, not the absolute numbers."}
| $n$    | Cholesky  | Inverse   | Memory   |
|-------:|----------:|----------:|---------:|
| 500    | 0.4 ms    | 2.3 ms    | 2 MB     |
| 1,000  | 2.0 ms    | 10 ms     | 8 MB     |
| 2,000  | 14 ms     | 82 ms     | 32 MB    |
| 4,000  | 132 ms    | 625 ms    | 128 MB   |
:::

Doubling $n$ multiplies the time by roughly eight, as $n^3$ predicts, and the
inverse costs about five times the factorization. A Bayesian optimization run
rarely has more than a few hundred observations, so a single factorization is
cheap. Fitting the kernel's own settings, such as its lengthscale, is what
multiplies the cost, because every step of that fit refactorizes the matrix
(@sec-marginal-likelihood). For larger data sets, the library GPyTorch
replaces the factorization by conjugate gradients, an iterative method that needs only products of the kernel matrix
with vectors, and runs them on graphics processors; its authors report that
this reduces the asymptotic cost of exact inference from $O(n^3)$ to $O(n^2)$
[@gardner2018gpytorch].

### The square root that makes samples {#sec-linalg-cholesky-samples}

The Cholesky factor is also a recipe for building correlated randomness out of
independent randomness. If $\vz$ has independent coordinates with variance 1,
then $\mL\vz$ has covariance $\mL\mL^\T = \mSigma$, a consequence of the rule
$\Cov[\mA\vz] = \mA\Cov[\vz]\mA^\T$ that @sec-gaussian-linear derives. Turn on
*Show samples* in @fig-linalg-ellipsoid to see 160 such points: each is
$\mL\vz$ for a draw $\vz$ of three independent random numbers with mean 0 and
variance 1 (standard normal numbers, the bell curve of @sec-gauss-standard),
and together they fill the ellipsoid's shape. Because $\mL$ is lower triangular,
the first coordinate uses only $z_1$, the second $z_1$ and $z_2$, and the third
all three, so each new coordinate is built from the randomness of the earlier
ones plus a fresh part of its own. This is how the posterior samples of
@sec-gp-posterior-samples are drawn.

## Determinants as volume {#sec-determinants}

The density of a multivariate Gaussian (@sec-gaussian-nd) and the marginal
likelihood used to fit kernels (@sec-marginal-likelihood) both contain a
determinant. Its meaning is geometric.

The *determinant* of a square matrix, $\det\mA$ or $\lvert\mA\rvert$, is the
factor by which the map $\mA$ multiplies areas in two dimensions, volumes in
three, and $d$-dimensional volumes in general. Its sign records whether the
map flips orientation, as a mirror does. For a $2 \times 2$ matrix,

$$
\det \begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc,
$$

the signed area of the parallelogram spanned by the two columns. This is the
yellow parallelogram in @fig-linalg-transform, the image of the unit square;
the Singular preset flattens it to zero area, and the Reflection preset turns
the determinant negative.

Three properties follow from the picture. Applying two maps multiplies their
volume factors, so $\det(\mA\mathbf{B}) = \det\mA \cdot \det\mathbf{B}$, and in
particular $\det(\mA^{-1}) = 1/\det\mA$. A matrix is invertible exactly when its
determinant is nonzero, when it does not flatten space. And for a symmetric
matrix $\mA = \mathbf{U}\bm{\Lambda}\mathbf{U}^\T$, the rotations
$\mathbf{U}$ and $\mathbf{U}^\T$ preserve volume while $\bm{\Lambda}$ stretches
axis $i$ by $\lambda_i$, so

$$
\det\mA = \prod_{i=1}^d \lambda_i.
$$ {#eq-linalg-det-eigen}

For a covariance matrix the determinant is therefore a single number for the
overall size of the uncertainty, sometimes called the *generalized variance*.
The ellipsoid of @fig-linalg-ellipsoid has half-axes $\sqrt{\lambda_i}$ and so
volume $\frac{4}{3}\pi\sqrt{\lambda_1\lambda_2\lambda_3} = \frac{4}{3}\pi
\sqrt{\det\mSigma}$. Correlation shrinks it: for three variables with unit
variance and correlations $\rho_{12}, \rho_{13}, \rho_{23}$,

$$
\det\mSigma = 1 + 2\rho_{12}\rho_{13}\rho_{23} - \rho_{12}^2 - \rho_{13}^2 - \rho_{23}^2,
$$

which falls from 1 for independent variables toward 0 as the ellipsoid
flattens, and becomes negative for the impossible triple ($1 - 0.64 - 0.64 =
-0.28$). In @sec-gp-information-gain, a determinant of this kind measures how
much a set of evaluations can teach about the objective, and the evaluations
that make it largest are the most informative.

### The log-determinant from the Cholesky factor {#sec-linalg-logdet}

The determinant of a triangular matrix is the product of its diagonal entries.
(In the $2 \times 2$ formula, $ad - bc$ with $b = 0$ is $ad$. In general, a
triangular matrix stretches coordinate axis $i$ by its $i$-th diagonal entry
and then shears, and a shear moves points parallel to the other axes without
changing any volume, as the Shear preset of @fig-linalg-transform shows.)
With $\mA = \mL\mL^\T$, the product rule for determinants gives

$$
\log\det\mA = 2\sum_{i=1}^n \log L_{ii}.
$$ {#eq-linalg-logdet}

The logarithm is not decoration. A kernel matrix with many tiny eigenvalues has
a determinant far below the smallest positive double-precision number, about
$5 \times 10^{-324}$. For the 100-input example of @sec-linalg-kernel-spectrum, with
the jitter that makes it factorizable, the product of the computed eigenvalues
underflows to zero, while @eq-linalg-logdet gives $\log\det\mA \approx -1139$,
a moderate number that costs nothing once the factor is known. Every implementation of the Gaussian log density and of the log marginal
likelihood computes the determinant this way (@sec-gauss-code).

## Block matrices and the Schur complement {#sec-block-matrices}

A Gaussian process works with two groups of variables at once: the function
values already observed and those to be predicted. Its matrices are therefore
naturally split into blocks, as in @eq-gp-joint. This section develops the
algebra of such blocks and ends with the one formula @sec-gaussian-conditioning
needs.

### Partitioned matrices {#sec-linalg-partitioned}

A symmetric matrix partitioned into blocks looks like

$$
\mathbf{M} = \begin{bmatrix} \mA & \mathbf{B} \\ \mathbf{B}^\T & \mathbf{D} \end{bmatrix},
$$ {#eq-linalg-block}

where $\mA$ is $p \times p$, $\mathbf{D}$ is $q \times q$, and $\mathbf{B}$ is
$p \times q$. Blocks multiply like numbers, with the rule that the order of
each product is kept, since matrices do not commute. For a covariance, $\mA$
and $\mathbf{D}$ are the covariances within each group and $\mathbf{B}$ holds
the covariances across groups.

### Elimination by blocks {#sec-linalg-schur}

Solving two equations in two unknowns by hand, we use the first equation to
eliminate the first unknown from the second. The same step with blocks
produces the central object of this section.

::: {.definition #def-linalg-schur title="Schur complement"}
For the partitioned matrix @eq-linalg-block with $\mA$ invertible, the Schur
complement of $\mA$ is

$$
\mathbf{S} = \mathbf{D} - \mathbf{B}^\T\mA^{-1}\mathbf{B}.
$$ {#eq-linalg-schur}
:::

For a $2 \times 2$ matrix with entries $a, b, d$, the Schur complement of $a$ is
the number $d - b^2/a$ from the Cholesky derivation. In general it is what
remains of $\mathbf{D}$ after the part explained by the first group is removed.

::: {.derivation title="Block elimination"}
1. Subtract $\mathbf{B}^\T\mA^{-1}$ times the first block row from the second.
   As a matrix product, with
   $\mathbf{E} = \begin{bmatrix} \mI & \mathbf{0} \\ -\mathbf{B}^\T\mA^{-1} & \mI \end{bmatrix}$,
   $\mathbf{E}\mathbf{M} = \begin{bmatrix} \mA & \mathbf{B} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}$,
   because the bottom-right block becomes $\mathbf{D} - \mathbf{B}^\T\mA^{-1}\mathbf{B}$.
2. Do the same to the columns: multiplying on the right by $\mathbf{E}^\T$
   clears the top-right block, and since $\mA$ is symmetric,
   $\mathbf{E}\mathbf{M}\mathbf{E}^\T = \begin{bmatrix} \mA & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}$.
3. $\mathbf{E}$ is invertible: its inverse is the same matrix with the sign of
   the off-diagonal block flipped. So
   $\mathbf{M} = \mathbf{E}^{-1}\begin{bmatrix} \mA & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}\mathbf{E}^{-\T}$.
:::

Three consequences follow, each used later in the book.

**Determinants.** $\mathbf{E}$ is block triangular with identity blocks on the
diagonal, so its determinant is 1 and
$\det\mathbf{M} = \det\mA \cdot \det\mathbf{S}$.

**Positive definiteness.** For any vector $\vw$, set
$\mathbf{v} = \mathbf{E}^{-\T}\vw$; then
$\vw^\T\mathbf{M}\vw = \mathbf{v}_1^\T\mA\mathbf{v}_1 + \mathbf{v}_2^\T\mathbf{S}\mathbf{v}_2$,
where $\mathbf{v}_1$ and $\mathbf{v}_2$ are the two parts of $\mathbf{v}$. So
$\mathbf{M}$ is positive definite exactly when both $\mA$ and its Schur
complement $\mathbf{S}$ are. This is the precise version of the impossible
triple: once $x_1$ is accounted for, what is left of $x_2$ and $x_3$ must still
be a valid covariance (@exr-linalg-range).

**The inverse.** Inverting step 3 block by block, using
$(\mathbf{E}^{-1})^{-1} = \mathbf{E}$, gives
$\mathbf{M}^{-1} = \mathbf{E}^\T \begin{bmatrix} \mA^{-1} & \mathbf{0} \\ \mathbf{0} & \mathbf{S}^{-1} \end{bmatrix} \mathbf{E}$,
and multiplying out,

$$
\mathbf{M}^{-1} =
\begin{bmatrix}
\mA^{-1} + \mA^{-1}\mathbf{B}\mathbf{S}^{-1}\mathbf{B}^\T\mA^{-1} & -\mA^{-1}\mathbf{B}\mathbf{S}^{-1} \\
-\mathbf{S}^{-1}\mathbf{B}^\T\mA^{-1} & \mathbf{S}^{-1}
\end{bmatrix}.
$$ {#eq-linalg-block-inverse}

The bottom-right block of the inverse is the inverse of the Schur complement.
@sec-gaussian-conditioning uses this, with $\mA$ the covariance of the
observed values and $\mathbf{D}$ that of the unobserved ones, to show that the
covariance after conditioning is the Schur complement:
$\mathbf{D} - \mathbf{B}^\T\mA^{-1}\mathbf{B}$, the prior covariance minus the
part the observations explain. @sec-gp-conditioning is the same formula with
kernel matrices in the blocks, and @sec-id-block and @sec-id-woodbury collect
it with related identities [@petersen2012matrix].

### Adding one observation {#sec-linalg-cholesky-update}

The block view also says how to update a Cholesky factor cheaply, which a
Bayesian optimizer needs after every evaluation. Suppose $\mK = \mL\mL^\T$ for
the $n$ inputs observed so far, and a new input arrives with covariances
$\vk$ to the old inputs and variance $\kappa$. The new kernel matrix and its
factor have the block forms

$$
\begin{bmatrix} \mK & \vk \\ \vk^\T & \kappa \end{bmatrix}
= \begin{bmatrix} \mL & \mathbf{0} \\ \mathbf{l}^\T & l_{\ast} \end{bmatrix}
\begin{bmatrix} \mL^\T & \mathbf{l} \\ \mathbf{0}^\T & l_{\ast} \end{bmatrix}.
$$

Matching blocks, $\mL\mathbf{l} = \vk$, one forward substitution costing
$O(n^2)$, and $l_{\ast}^2 = \kappa - \mathbf{l}^\T\mathbf{l}$. Since
$\mathbf{l}^\T\mathbf{l} = \vk^\T\mL^{-\T}\mL^{-1}\vk = \vk^\T\mK^{-1}\vk$, the
new diagonal entry squared is $\kappa - \vk^\T\mK^{-1}\vk$, the Schur
complement of $\mK$, which is also the Gaussian process's posterior variance
at the new input (@eq-gp-pointwise). Growing the factor by one row costs
$O(n^2)$ instead of the $O(n^3)$ of starting over, as long as the kernel's
settings stay fixed. And if the new input repeats an old one without
observation noise, $l_{\ast}$ is zero and the factorization breaks: the matrix
has become singular, for the reason given in @sec-linalg-pd-definition.

## Exercises {#sec-linalg-exercises}

::: {.exercise #exr-linalg-cholesky}
Let $\mA = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$. (a) Compute its
Cholesky factor by hand. (b) Use it to find $\det\mA$ and check with
$ad - bc$. (c) Solve $\mA\vx = (2, 4)^\T$ by forward and back substitution.

::: {.solution}
(a) By the $2 \times 2$ derivation, $l_{11} = \sqrt{4} = 2$,
$l_{21} = 2/2 = 1$, and $l_{22} = \sqrt{3 - 1} = \sqrt{2}$, so
$\mL = \begin{bmatrix} 2 & 0 \\ 1 & \sqrt2 \end{bmatrix}$.

(b) $\det\mA = (l_{11}l_{22})^2 = (2\sqrt2)^2 = 8$, and $4 \cdot 3 - 2 \cdot 2 = 8$.

(c) Forward: $z_1 = 2/2 = 1$ and $z_2 = (4 - 1 \cdot 1)/\sqrt2 = 3/\sqrt2$.
Back, with $\mL^\T = \begin{bmatrix} 2 & 1 \\ 0 & \sqrt2 \end{bmatrix}$:
$x_2 = (3/\sqrt2)/\sqrt2 = 1.5$ and $x_1 = (1 - 1.5)/2 = -0.25$. Check:
$4(-0.25) + 2(1.5) = 2$ and $2(-0.25) + 3(1.5) = 4$. These are the numbers the
code sketch in @sec-linalg-cholesky-solve prints.
:::
:::

::: {.exercise #exr-linalg-range}
Three variables have unit variances, and $\rho_{12} = \rho_{13} = 0.8$. Using
the Schur complement of the first variable, find every value of $\rho_{23}$ for
which the covariance matrix is positive definite. Interpret the Schur
complement.

::: {.solution}
Partition with $\mA = [1]$, $\mathbf{B} = (0.8, 0.8)$, and
$\mathbf{D} = \begin{bmatrix} 1 & \rho_{23} \\ \rho_{23} & 1 \end{bmatrix}$.
Then

$$
\mathbf{S} = \mathbf{D} - \mathbf{B}^\T\mathbf{B}
= \begin{bmatrix} 0.36 & \rho_{23} - 0.64 \\ \rho_{23} - 0.64 & 0.36 \end{bmatrix}.
$$

Since $\mA = [1]$ is positive definite, the whole matrix is positive definite
exactly when $\mathbf{S}$ is, which for a $2 \times 2$ matrix with positive
diagonal means $\det\mathbf{S} = 0.36^2 - (\rho_{23} - 0.64)^2 > 0$, so
$\lvert\rho_{23} - 0.64\rvert < 0.36$ and $0.28 < \rho_{23} < 1$. This is where
the ellipsoid of @fig-linalg-ellipsoid reappears. $\mathbf{S}$ is the
covariance of $x_2$ and $x_3$ after the part explained by $x_1$ is removed
(@sec-gaussian-conditioning): each keeps variance $1 - 0.8^2 = 0.36$, and their
remaining covariance $\rho_{23} - 0.64$ must be a valid one, at most 0.36 in
absolute value.
:::
:::

::: {.exercise #exr-linalg-two-inputs}
Two inputs have kernel value $\rho$, so their kernel matrix is
$\mK = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$. Find its
eigenvalues and eigenvectors, and its condition number. How many decimal digits
can solving with $\mK$ lose when $\rho = 0.9999$, as for two inputs very close
together under a smooth kernel?

::: {.solution}
$\mK(1, 1)^\T = (1 + \rho)(1, 1)^\T$ and $\mK(1, -1)^\T = (1 - \rho)(1, -1)^\T$,
so the eigenvalues are $1 \pm \rho$ with eigenvectors $(1, 1)/\sqrt2$ (the
average of the two values) and $(1, -1)/\sqrt2$ (their difference). The
condition number is $(1 + \rho)/(1 - \rho)$, which for $\rho = 0.9999$ is
$1.9999/0.0001 \approx 20{,}000$, so about four of the sixteen digits can be
lost. The small eigenvalue is the prior variance of the difference between the
two function values: two close inputs leave almost no room for their values to
differ, and that is the direction in which the matrix is nearly singular.
:::
:::

::: {.exercise #exr-linalg-update}
In @sec-linalg-cholesky-update, suppose the observations are noisy, so the
matrix to factor is $\mK + \sigma_n^2\mI$. What is the new diagonal entry
$l_{\ast}^2$ now? Show that it can no longer be zero, even when the new input
repeats an old one.

::: {.solution}
The new matrix has $\kappa + \sigma_n^2$ in the corner and
$\mK + \sigma_n^2\mI$ in the old block, so
$l_{\ast}^2 = \kappa + \sigma_n^2 - \vk^\T(\mK + \sigma_n^2\mI)^{-1}\vk$. The
quantity $\kappa - \vk^\T(\mK + \sigma_n^2\mI)^{-1}\vk$ is the noisy posterior
variance of $f$ at the new input, $\sigma^2(\vx)$ of @eq-gp-noisy, so
$l_{\ast}^2 = \sigma^2(\vx) + \sigma_n^2$,
the variance of a new noisy measurement there. Because $\sigma^2(\vx) \ge 0$,
$l_{\ast}^2 \ge \sigma_n^2 > 0$: observation noise keeps the matrix positive
definite, which is why noisy models rarely need jitter.
:::
:::

## Further reading {#further-reading .unnumbered}

- @strang2016introduction is a patient introduction to matrices as maps,
  eigenvalues, and positive definite matrices, with the geometric emphasis of
  this chapter.
- @golub2013matrix is the standard reference for how these computations are
  done in floating point: triangular systems, the Cholesky factorization,
  operation counts, and conditioning.
- @rasmussen2006gaussian, appendix A, collects the matrix identities and the
  Cholesky-based computations that Gaussian process regression uses.
- @petersen2012matrix is a compact catalog of identities, including the block
  inverse and determinant formulas, useful for checking a derivation.
- @sanderson2016essence is an animated video series that shows matrices
  acting on the plane and in space, the picture behind @fig-linalg-transform.
- @higham2002computing treats the problem of repairing an invalid correlation
  matrix, the practical side of @sec-linalg-impossible.
