## Definition of Contrastive Loss
We use the equation for contrastive loss from [Dimensionality Reduction by Learning an Invariant Mapping](https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) except we use $U$ and $V$ for the two vectors and we change the meaning of the label $Y$ to be more intuitive, so $Y = 1$ for a 'positive' similar pair $U, V$ and $Y = 0$ for a 'negative' dissimilar pair $U, V$:
$$L(Y, U, V) = Y\frac{1}{2}(D)^2 + (1 - Y)\frac{1}{2}\max(0, m - D)^2$$
Here, $Y = 0$ if the points are deemed "similar" and $Y = 1$ if they are deemed dissimilar. $D$ is a distance function; here, we consider Euclidean distance:
$$D = \sqrt{(U_1 - V_1)^2 + \ldots + (U_n - V_n)^2}$$

## Derivation of Derivative
We will now find $\frac{\partial L}{\partial U_i}$:


$$ \frac{\partial L}{\partial U_i} = \frac{\partial ((Y)\frac{1}{2}(D)^2)}{\partial U_i} + \frac{\partial ((1 - Y)\frac{1}{2}\max(0, m - D)^2)}{\partial U_i}$$

We now examine the left term first and apply the chain rule:

$$ \frac{\partial ((Y)\frac{1}{2}(D)^2)}{\partial U_i} = DY \frac{\partial D}{\partial U_i}$$

Examining just the partial derivative of Euclidean distance, we can again use the chain rule:

$$\frac{\partial D}{\partial U_i} 
    = \frac{\partial \sqrt{(U_1 - V_1)^2 + \ldots + (U_n - V_n)^2}}{\partial U_i} 
    = \frac{\frac{\partial}{\partial U_i}((U_1 - V_1)^2 + \ldots + (U_n - V_n)^2)}{2\sqrt{(U_1 - V_1)^2 + \ldots + (U_n - V_n)^2}}
    = \frac{U_i - V_i}{D}$$

Plugging in to the left hand term, we have:
$$DY\frac{U_i - V_i}{D} = Y(U_i - V_i)$$

Moving on to the right term, if $m < D$, then the derivative of the term is just $0$ . Thus we consider the case $m > D$, so we can simplify the $\max$ and apply the chain rule:

$$\frac{\partial ((1 - Y)\frac{1}{2}(m - D)^2)}{\partial U_i} 
    = (m - D)(1 - Y) \frac{\partial (m - D)}{\partial U_i}
    = -(m - D)(1 - Y) \frac{\partial D}{\partial U_i}$$

Using our result for the derivative of Euclidean distance above, we can further simplify:

$$-(m - D)(1 - Y)\frac{U_i - V_i}{D}$$

Thus the entire equation is:

$$\frac{\partial L}{\partial U_i} = Y(U_i - V_i) - \frac{\max(m - D, 0)}{D}(1 - Y)(U_i - V_i)$$

Simplifying, we have:


$$\frac{\partial L}{\partial U_i} = (U_i - V_i)(Y - (1 - Y)\frac{\max(m - D, 0)}{D})$$

Note that this also generalizes to $Y \in [0, 1]$.