# Derivation of Layer Norm Backpropagate

## Forward pass

Let $N$ be the dimension of the vectors that are being normalized. Let $x_1, x_2, ..., x_N$ be the elements of the vector that are being normalized, and $y_1, y_2, ..., y_N$ be the normalized components of the vector. 

$$ \mu = \frac{1}{N}\sum_{i=1}^N x_i $$
$$ \sigma^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \mu) $$

$$ \hat{x} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$

$$ y_i = \gamma_i \cdot \hat{x} + \beta_i $$

Where $\gamma_i$ and $\beta_i$ are learned. 

## Backpropagation

### Partial derivative w.r.t. the input

$$ \frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^N \frac{\partial L}{\partial y_i} \gamma_i (x_i - \mu) (-\frac{1}{2}) (\sigma^2 + \epsilon)^{-3/2} $$

$$ \frac{\partial L}{\partial \mu} = \sum_{i=1}^N \frac{\partial L}{\partial y_i} \gamma_i \frac{-1}{\sqrt{\sigma^2 + \epsilon}} - \frac{\partial L}{\partial \sigma^2} \frac{2}{N} \sum_{i=1}^N (x_i - \mu) $$ 

$$ \frac{\partial L}{\partial x_i} = \frac{\gamma_i}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \frac{2}{N}(x_i - \mu) + \frac{\partial L}{\partial \mu} \frac{1}{N} $$ 

### Partial derivative w.r.t. the learned parameters

Note: These need to be summed over the batch, the following are just for a single element of the batch.

$$ \frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i} \hat{x_i} $$ 

$$ \frac{\partial L}{\partial \beta_i} = \frac{\partial L}{\partial y_i} $$