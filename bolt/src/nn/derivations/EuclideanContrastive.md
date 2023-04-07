## Definition of Contrastive Loss
The equation for contrastive loss from [Dimensionality Reduction by Learning an Invariant Mapping](https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) is:
$$L(Y, X_1, X_2) = (1 - Y)\frac{1}{2}(D)^2 + (Y)\frac{1}{2}\max(0, m - D)^2$$
Here, $Y = 0$ if the points are deemed "similar" and $Y = 1$ if they are deemed dissimilar. $D$ is a distance function; here, we consider Euclidean distance:
$$D = \sqrt{(X_{1, 1} - X_{2, 1})^2 + \ldots + (X_{1, n} - X_{2, n})^2}$$

## Derivation of Derivative
We will now find $\frac{\partial L}{\partial X_{1, i}}$:


$$ \frac{\partial L}{\partial X_{1, i}} = \frac{\partial ((1 - Y)\frac{1}{2}(D)^2)}{\partial X_{1, i}} + \frac{\partial ((Y)\frac{1}{2}\max(0, m - D)^2)}{\partial X_{1, i}}$$

We now examine the left term first and apply the chain rule:

$$ \frac{\partial ((1 - Y)\frac{1}{2}(D)^2)}{\partial X_{1, i}} = D(1 - Y) \frac{\partial D}{\partial X_{1, i}}$$

We can again use the chain rule to take the derivative of $D$, so that then this simplifies to:
$$D(1 - Y)\frac{1}{2D} 2(X_{1, i} - X_{2, i}) = (1 - Y)(X_{1, i} - X_{2, i})$$

Moving on to the right term, if $m < D$, then the derivative of the term is just $0$ . Thus we consider the case $m > D$, so we can simplify the $\max$ and apply the chain rule:

$$\frac{\partial ((Y)\frac{1}{2}(m - D)^2)}{\partial X_{1, i}} = (m - D)(Y) \frac{\partial (m - D)}{\partial X_{1, i}}$$

Using our result for the derivative of Euclidean distance above, we can further simplify:

$$(m - D)(Y)(X_{2, i} - X_{1, i})$$

Thus the entire equation is:

$$\frac{\partial L}{\partial X_{1, i}} = (1 - Y)(X_{1, i} - X_{2, i}) + \max(m - D, 0)(Y)(X_{2, i} - X_{1, i})$$

Simplifying, we have:


$$\frac{\partial L}{\partial X_{1, i}} = (X_{1, i} - X_{2, i})(1 - Y - Y\max(m - D, 0))$$

Note that this also generalizes to $Y \in [0, 1]$.