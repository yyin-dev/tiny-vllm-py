# Rotary Position Embedding (RoPE)

RoPE applies a position-dependent **rotation** to the query and key vectors.

There are three different levels to understand:

1. **Geometry**: RoPE rotates vectors in many independent 2D planes.
2. **Convention**: We must decide which hidden dimensions form each 2D plane.
3. **Implementation**: Once the pairing is chosen, we implement the rotation efficiently.

---

## 1. Geometry: rotating a 2D vector

Consider a 2D vector

$$
v=
\begin{bmatrix}
a\\
b
\end{bmatrix}.
$$

Rotating it by angle $\theta$ is

$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}.
$$

Applying the rotation gives

$$
\begin{aligned}
a' &= a\cos\theta - b\sin\theta,\\
b' &= a\sin\theta + b\cos\theta.
\end{aligned}
$$

This can be rewritten as

$$
R(\theta)v
=
\cos\theta\,v
+
\sin\theta
\begin{bmatrix}
-b\\
a
\end{bmatrix}.
$$

The vector

$$
\begin{bmatrix}
-b\\
a
\end{bmatrix}
$$

is simply the original vector rotated by **90°**.

Therefore, instead of explicitly multiplying by a rotation matrix, RoPE computes

```python
x = x * cos + rotate(x) * sin
```

where `rotate()` computes

```text
(a,b) → (-b,a)
```

for every 2D vector.

---

## 2. Convention: which dimensions form a 2D vector?

RoPE requires us to divide the hidden dimensions into independent 2D planes.

There is no unique way to do this.

### Original RoPE (RoFormer)

The original paper pairs adjacent dimensions.

```
(0,1)
(2,3)
(4,5)
(6,7)
```

For a hidden vector

```
[a,b,c,d,e,f,g,h]
```

the 2D vectors are

```
(a,b)
(c,d)
(e,f)
(g,h)
```

---

### Llama / Qwen / HuggingFace

Modern LLMs instead pair

```
(0,4)
(1,5)
(2,6)
(3,7)
```

For the same hidden vector

```
[a,b,c,d,e,f,g,h]
```

the 2D vectors become

```
(a,e)
(b,f)
(c,g)
(d,h)
```

Mechanically, this is the only difference.

This is also why checkpoints trained with one convention cannot be used with the other convention—the learned query/key projections assume a particular pairing of dimensions.

---

## 3. Implementation

### Original RoPE

Since adjacent dimensions form each pair,

```text
[a,b,c,d,e,f,g,h]
```

`rotate()` produces

```text
[-b,a,-d,c,-f,e,-h,g]
```

Implementation:

```python
def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    return torch.stack((-x2, x1), dim=-1).flatten(-2)
```

---

### Llama / Qwen (`rotate_half`)

Since the pairs are

```
(a,e)
(b,f)
(c,g)
(d,h)
```

`rotate()` becomes

```text
[-e,-f,-g,-h,
 a, b, c, d]
```

Implementation:

```python
def rotate_half(x):
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]

    return torch.cat((-x2, x1), dim=-1)
```

Although this looks different, it is still performing exactly the same

```
(a,b) → (-b,a)
```

rotation on each 2D plane.

---

## Why `rotate_half`?

The advantage is **not** mathematical—the two implementations perform the same rotation.

The advantage is implementation efficiency.

`rotate_every_two` accesses

```
0,2,4,6,...
```

and

```
1,3,5,7,...
```

which are strided memory accesses.

`rotate_half` instead accesses

```
[:d/2]
```

and

```
[d/2:]
```

which are contiguous blocks of memory.

This leads to cleaner code and better GPU memory access patterns, which is why nearly all modern LLMs (Llama, Qwen, Mistral, Gemma, etc.) adopt the `rotate_half` implementation.

---

## Conceptual equivalence

The two conventions look different because they pair different hidden dimensions.

Mathematically, however, they are related by a fixed change of basis (a permutation of coordinates). There exists a permutation under which the `rotate_half` formulation becomes exactly the original RoPE formulation.

This permutation is a proof of equivalence—it is **not** something that Llama or Qwen perform at runtime.