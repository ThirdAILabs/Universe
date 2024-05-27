# Derivation of CosineSimilarity Backpropagation 

## Forward Pass

Let $u$, $v$ be the vectors we are computing the cosine similarity for.

$$ sim(u,v) = \frac{u \cdot v}{\|u\| \|v\|} $$


## Backpropagation

Let $u = [u_1, u_2, ..., u_n]$ and $v = [v_1, v_2, ..., v_n]$. 

First lets consider the partial derivative with of the $\|u\|^{-1}$ (which is also the same for $v$).

$$ \begin{align*} 
  \frac{\partial \|u\|^{-1}}{\partial u_i} &= \frac{\partial}{\partial u_i} (u_1^2 + u_2^2 + ... + u_n^2)^{-1/2} \\
  &= -\frac{1}{2} (u_1^2 + u_2^2 + ... + u_n^2)^{-3/2}\ 2 u_i \\
  &= -\frac{1}{2} \|u\|^{-3}\ 2u_i \\
  &= -u_i\|u\|^{-3}
\end{align*} $$

Now we can compute the full partial derivative. Again note that this will be done for $u$ but it is the same for $v$.

$$ \begin{align*}
\frac{\partial sim(u,v)}{\partial u_i} &= \frac{\partial}{\partial u_i} \frac{u \cdot v}{\|u\| \|v\|} \\
&= \frac{1}{\|v\|} \left( \frac{1}{\|u\|}\frac{\partial u \cdot v}{\partial u_i} + u \cdot v \frac{\partial \|u\|^{-1}}{\partial u_i}\right) \\
&= \frac{1}{\|v\|} \left(v_i  \frac{1}{\|u\|}- u_i\frac{u \cdot v}{\|u\|^3} \right) \\
&= \frac{v_i}{\|u\|\|v\|} - u_i\frac{u \cdot v}{\|u\|^3\|v\|} \\
&= \frac{v_i}{\|u\|\|v\|} - u_i\frac{sim(u,v)}{\|u\|^2}
\end{align*} $$