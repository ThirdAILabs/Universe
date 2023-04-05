# Derivation of LayerNorm Backpropagation 

## Forward Pass 

$$ \mu = \frac{1}{N} \sum_{i=1}^N x_i $$

$$ \sigma^2 =  \frac{1}{N}  \sum_{i=1}^N (x_i - \mu)^2$$

$$ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$

$$ y_i = \gamma_i \cdot \hat{x}_i + \beta_i $$ 


## Backpropagation

First lets consider the partial derivatives of $\mu$ and $\sigma^2$ with respect to some element of the input vector $x_i$.

$$ \frac{\partial \mu}{\partial x_i} = \frac{1}{N} $$ 

$$ \frac{\partial \sigma^2}{\partial x_i} = \frac{1}{N} \left[ \sum_{j = 1}^N 2(x_j - \mu) (-\frac{1}{N}) + 2(x_i - \mu) \right] = \frac{2}{N} (x_i - \mu) $$

Note that $ \sum_{i=1}^N (x_i - \mu) = 0$. 

Now we can use this to compute the parital derivative of $\frac{1}{\sqrt{\sigma^2 + \epsilon}}$ with respect to an element of the input $x_i$.

$$ \begin{align*} 
  \frac{\partial}{\partial x_i}\frac{1}{\sqrt{\sigma^2 + \epsilon}} &= -\frac{1}{2} \frac{1}{\sqrt{\sigma^2 + \epsilon}^3} \frac{\partial \sigma^2}{\partial x_i} \\
  &= -\frac{1}{2} \frac{1}{\sqrt{\sigma^2 + \epsilon}^3} \frac{2}{N} (x_i - \mu) \\
  &= -\frac{1}{N}\frac{1}{\sqrt{\sigma^2 + \epsilon}^2} \hat{x}_i 
\end{align*} $$

Now we can use this to write a formula for an arbitrary element of the Jacobian ($J_f$)of the function $f: (x_1, x_2, ..., x_N) \to (\hat{x}_1, \hat{x}_2, ..., \hat{x}_N) $. 

$$ \begin{align*} 
  \frac{\partial \hat{x}_i}{\partial x_j} &= \left( \frac{\partial x_i}{\partial x_j} - \frac{\partial \mu}{\partial x_j} \right) \frac{1}{\sqrt{\sigma^2 + \epsilon}} + (x_i - \mu) \frac{\partial}{\partial x_j}\frac{1}{\sqrt{\sigma^2 + \epsilon}} \\
  &= \left( 1_{i = j} - \frac{1}{N} \right) \frac{1}{\sqrt{\sigma^2 + \epsilon}} + (x_i - \mu) \left( - \frac{1}{N} \right)\frac{1}{\sqrt{\sigma^2 + \epsilon}^2} \hat{x}_j \\ 
  &= \left( 1_{i = j} - \frac{1}{N} \right) \frac{1}{\sqrt{\sigma^2 + \epsilon}} - \hat{x}_i \frac{1}{N} \frac{1}{\sqrt{\sigma^2 + \epsilon}} \hat{x}_j
\end{align*} $$

Now the partial derivatives of the loss $L$ with respect to each of the inputs can be written as:

$$ \left( \frac{\partial L}{\partial x_1}, \frac{\partial L}{\partial x_2}, ... , \frac{\partial L}{\partial x_N} \right) = \left( \frac{\partial L}{\partial \hat{x}_1}, \frac{\partial L}{\partial x_2}, ... , \frac{\partial L}{\partial x_N} \right) J_f $$ 

If we consider a single element of the input, we can write its partial derivative as:

$$ \frac{\partial L}{\partial x_i} = \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial x_i} $$

Using the equation for $y_i$ we get:

$$ \frac{\partial L}{\partial \hat{x}_j} = \frac{\partial L}{\partial y_j} \gamma_j $$ 

This gives us the equation:

$$ \begin{align*} 
  \frac{\partial L}{\partial x_i} &= \sum_{j=1}^N \frac{\partial L}{\partial y_j} \gamma_j \frac{\partial \hat{x}_j}{\partial x_i} \\
  &= \sum_{j=1}^N \frac{\partial L}{\partial y_j} \gamma_j \left[ \left( 1_{i = j} - \frac{1}{N} \right) \frac{1}{\sqrt{\sigma^2 + \epsilon}} - \hat{x}_j \frac{1}{N} \frac{1}{\sqrt{\sigma^2 + \epsilon}} \hat{x}_i \right] \\
  &= \frac{\partial L}{\partial y_i} \gamma_i \frac{1}{\sqrt{\sigma^2 + \epsilon}} - \sum_{j=1}^N \frac{\partial L}{\partial y_j} \gamma_j \frac{1}{N} \frac{1}{\sqrt{\sigma^2 + \epsilon}} - \sum_{j=1}^N \frac{\partial L}{\partial y_j} \gamma_j \hat{x}_j \frac{1}{N} \frac{1}{\sqrt{\sigma^2 + \epsilon}} \hat{x}_i \\ 
  &= \frac{1}{N\sqrt{\sigma^2 + \epsilon}} \left[ N \frac{\partial L}{\partial y_i} \gamma_i - \sum_{j=1}^N \frac{\partial L}{\partial y_j} \gamma_j - \hat{x}_i \sum_{j=1}^N \frac{\partial L}{\partial y_j} \gamma_j \hat{x}_j \right]
\end{align*}$$

Lastly we can compute the gradients with respect to the learnable parameters:

$$ \frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i} \hat{x}_i $$ 

$$ \frac{\partial L}{\partial \beta_i} = \frac{\partial L}{\partial y_i} $$

Note that these gradients for $\gamma$ and $\beta$ should be summed accross the elements of a batch. 