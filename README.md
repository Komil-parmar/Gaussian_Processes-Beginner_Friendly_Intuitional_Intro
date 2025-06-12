# 🧠 Gaussian Processes: A Complete Guide

Welcome! This is a deep-dive into one of the most elegant tools in machine learning — **Gaussian Processes** (GPs).

Over the past week, I’ve spent nearly **2 hours a day** immersing myself in Gaussian Processes — not just learning how to apply them, but understanding the *why* behind every formula, every assumption, and every intuition.

---

### 0.1 🎯 Why I Made This

I'm currently a **first-year online degree student**, and everything I learn is the result of **self-study** — from textbooks, academic papers, and online resources.

My purpose for making this repo?

> To make Gaussian Processes **approachable**, even if you're **only** familiar with the basics of **random variables** — especially Gaussian ones.

This guide is written **by a learner, for learners** — explaining both **what** the formulas say and **why** they work.

---

### 0.2 🙌 Who Is This For?

* Are you **new to Gaussian Processes**? Start here.
* Do you already know the basics and want to **rebuild your understanding from intuition**? You’re in the right place.
* Are you here to **validate your/my knowledge** or leave feedback? I welcome it!

Whether you're a student, researcher, or ML practitioner, I’d love to hear your thoughts.

---

### 0.3 🤝 Let’s Connect

If you found this guide useful, insightful, or just want to share ideas, please feel free to reach out:

📬 **LinkedIn**: [komil-parmar-488967243](https://www.linkedin.com/in/komil-parmar-488967243/)
📧 **Email**:

* `komilparmar57@gmail.com`
* `komil.parmar@op.iitg.ac.in`

Let’s grow and learn together!

---

## 📚 Table of Contents

1. [🎲 Introduction: How Does Knowing the Random Variable Help?](#introduction)
2. [📐 The Gaussian Process Assumption](#assumption)
3. [📊 Understanding Joint Multivariate Normal Distribution](#joint-distribution)
4. [🔗 Kernel Functions](#kernel-functions)
5. [🧮 Kernel Matrix](#kernel-matrix)
6. [🎯 Parameter Estimation: Predicting Mean and Covariance](#parameter-estimation)
7. [🧾 Mathematical Formulation](#mathematical-formulation)
8. [📉 Understanding K⁻¹ (Kernel Matrix Inverse)](#kernel-inverse)
9. [⚙️ Practical Implementation](#implementation)
10. [✅ Conclusion](#conclusion)

---

## 1. 🧠 Introduction: How Does Knowing the Random Variable Help?<a name="introduction"></a>

### 📌 Scenario 1: Classification with Known Distribution

Imagine two classes: **Class A** and **Class B**. You have data showing the *hours of study* and *results* for students from each class.

Now, suppose you're given:

* A dataset belonging to **Class A**, and
* A new data point (student) whose class is unknown.

You're asked: **Does this new student belong to Class A?**

You can't answer this directly without a learning algorithm.
But what if you were told:

* The data follows a **multivariate normal distribution**, and
* You're given the **parameters** (mean and covariance) of this distribution.

Then you can:

* Estimate the **likelihood** that the new data point comes from this distribution (through its pdf function),
* And hence compute the **probability** that it belongs to Class A.

---

### 📌 Scenario 2: Regression Within a Class

Now consider a different question:
Given that the new student is from **Class A**, can you predict their *result* based on their *hours of study*?

Suppose:

* Both *hours of study* and *results* are **individually normally distributed**.

Here’s the **key insight**:

> 📎 *If one normal variable is conditioned on another normal variable, the resulting distribution is still a normal distribution.*

This means:

* You can compute the **conditional distribution** of results given hours studied,
* Which is again a **normal distribution**.

What’s left?

* Just find out the **mean** and **covariance** of this conditional distribution.

🎯 **This is exactly what Gaussian Processes do. So let's now get into Gaussian Process starting with it's assumption**

---

## 2. 📐 The Gaussian Process Assumption <a name="assumption"></a>

In most real-world scenarios, it's **highly unlikely** that both the input data `X` and the corresponding target values `y` are **individually Gaussian distributed**.

This is where the **core assumption of a Gaussian Process (GP)** comes in which is really well thought, powerful and elegant.

---

### 2.1 🔍 What Does GP Assume Instead?

Rather than assuming `X` and `y` are individually Gaussian, we assume that:

- There exists some **unknown function** $f$, such that for *any finite subset* of inputs $x_1, x_2, \dots, x_n$, the outputs $f(x_1), f(x_2), \dots, f(x_n)$ are **jointly multivariate normally distributed**.

This implies:

* $f(x_1), f(x_2), \dots, f(x_n)$ follow a **multivariate normal distribution**,
* And the same holds for $f(y_1), f(y_2), \dots, f(y_n)$.

Importantly, we **don't need to know** the exact form of the function $f$—Gaussian Processes just works on the belief that this function exists and it still surprisingly works.

---

### 2.2 📊 What Are We Trying to Estimate?

The key quantity we want to compute is the **conditional distribution**:

$$
f(y_1), f(y_2), \dots, f(y_n) \mid f(x_1), f(x_2), \dots, f(x_n)
$$

> 🧠 This is **not** a collection of independent distributions like:
>
> $$
> f(y_1) \mid f(x_1), \quad f(y_2) \mid f(x_2), \quad \dots
> $$

Instead:

* We work with a **single multivariate normal distribution** over **all the test points together**,
* Conditioned on **all the observed data**.

This formulation allows Gaussian Processes to:

* Model complex, nonlinear functions (since the unknown imaginary function can be literally any function),
* Capture **correlations** between outputs (since we are taking into account all the training points at the same time),
* And make **joint predictions** with associated uncertainty (since the output for new inputs is drawn from a conditional multivariate normal distribution).

---

## 3. 📊 Understanding Joint Multivariate Normal Distribution <a name="joint-distribution"></a>

### 3.1 🔍 What Does *Jointly Multivariate Normally Distributed* Mean?

When we say that a set of values is *jointly multivariate normally distributed*, we mean:

> All the values can be thought of as being sampled **together** from a single multivariate (i.e., multi-dimensional) **normal distribution**.

If you're working with $n$ transformed data points, then:

* You're sampling from an **$n$-dimensional** multivariate normal distribution.

> 🧠 **Important Note:**
> $n$ refers to the **number of data points**, not the **dimensionality $d$** of the individual feature vectors in $X$.
> So, this is **not** a distribution over your feature space—it’s a distribution over the **function values** associated with the data points.

Just like a standard normal distribution gives the probability of a specific value,
a **multivariate normal distribution over functions** gives us the probability of a particular function being the true one that explains the data.

---

### 3.2 🤔 Still Wondering Why It's $n$-Dimensional?

If you select $n$ data points $x_1, x_2, \dots, x_n$, then the multivariate normal distribution becomes $n$-dimensional.

Why?

Because:

* Each sample drawn from this distribution represents a **possible function** $f$,
* And this function will produce values $f(x_1), f(x_2), \dots, f(x_n)$,
* So each sample corresponds to an **$n$-dimensional vector** of function values.

Hence, you're modeling a distribution **over functions**, by treating their values (outputs) at selected inputs as jointly normal.

* You're sampling from an **$n$-dimensional** multivariate normal distribution.

> 🧠 **Important Note:**
> $n$ refers to the **number of data points**, not the **dimensionality $d$** of the individual feature vectors in $X$.
> So, this is **not** a distribution over your feature space—it’s a distribution over the **function values** associated with the data points.

Just like a standard normal distribution gives the probability of a specific value,
a **multivariate normal distribution over functions** gives us the probability of a particular function being the true one that explains the data, i.e. makes our original data normally distributed after applying the function as we assumed earlier.

---

### 3.3 📈 Visualizing It

Suppose we select 3 input points: $x_1, x_2, x_3$.
Then a single sample from the GP prior gives us one function $f$, which evaluates to:

$$
\begin{bmatrix}
f(x_1) \\
f(x_2) \\
f(x_3)
\end{bmatrix}
\sim \mathcal{N}
\left(
\begin{bmatrix}
\mu_1 \\
\mu_2 \\
\mu_3
\end{bmatrix},
\begin{bmatrix}
\sigma_{11} & \sigma_{12} & \sigma_{13} \\
\sigma_{21} & \sigma_{22} & \sigma_{23} \\
\sigma_{31} & \sigma_{32} & \sigma_{33}
\end{bmatrix}
\right)
$$

Each such vector corresponds to one **possible function**, and the GP defines a **probability distribution over all such functions**.

> This is how Gaussian Processes enable learning from data by modeling an infinite space of functions—but using only finite data points and their joint distributions.

---

## 4. 🧮 Kernel Functions <a name="kernel-functions"></a>

### 4.1 🔍 What Is a Kernel Function?

A **kernel function** is a mathematical tool that takes **two input points**—say $x$ and $x'$—and returns a value that reflects how **similar** they are:

> * If $x$ and $x'$ are **close together**, the kernel outputs a **large value**.
> * If they are **far apart**, it outputs a **small value**.

In this way, a kernel function behaves somewhat like the **inverse** of a distance metric (like L1 or L2 distance), but instead of measuring how far points are, it measures **how much they "belong together."**

---

### 4.2 📈 How Does the Kernel Influence the GP?

The **kernel determines the relationship between inputs and their outputs** in the Gaussian Process.

* If the kernel says two points $x$ and $x'$ are similar,
  → the GP assumes their function values $f(x)$ and $f(x')$ should also be **close**.

* If the kernel says they’re dissimilar,
  → the GP allows the function values to be **different**.

This helps the GP decide which functions are likely to fit the data, essentially **filtering out unrealistic functions** and favoring those that respect the local structure of the input space.

---

### 4.3 🧪 Common Kernel Functions

Here are some widely used kernels:

- **Radial Basis Function (RBF) / Gaussian Kernel**
  This is the most popular kernel for smooth functions:

  $$k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2l^2}\right)$$

* **Linear Kernel**
  Suitable for modeling linear relationships:

  $$k(x, x') = \sigma^2 (x \cdot x')$$

* **Polynomial Kernel**
  Adds flexibility by introducing non-linear interactions:

  $$k(x, x') = (x \cdot x' + c)^d$$

Each kernel introduces different assumptions about the underlying function (e.g., smoothness, linearity, periodicity). The choice of kernel is critical—it defines the *shape* of the functions your GP believes in. Hence you will notice that since we are using RBF (Gaussian Kernel) in this intro, the graphs that follow will resemble gaussian shapes. 

---

## 5. 🧮 Kernel Matrix <a name="kernel-matrix"></a>

### 5.1 📐 What Is a Kernel Matrix?

A **kernel matrix** (also called a **Gram matrix**) is built by applying the kernel function to all pairs of data points.
Each entry $K(i, j)$ represents the similarity between the $i^\text{th}$ and $j^\text{th}$ inputs:

$$
K(i, j) = k(x_i, x_j)
$$

> 🔁 **Note:**
> Most kernel functions are **symmetric**, i.e., $k(x_i, x_j) = k(x_j, x_i)$.
> Therefore, the kernel matrix is typically **symmetric** as well.

We are going to use this matrix as the **covariance matrix** of the multivariate normal distribution over the outputs $f(x)$ not conditional one.
However we will use it there (in the conditional distribution) as well but with some modification.

---

### 5.2 🤔 But why Replace the Covariance Matrix With a Kernel Matrix?

In a multivariate Gaussian, the **covariance matrix** encodes how each output value correlates with the others:

* An entry $\Sigma(i, j)$ tells you **how much the value at index $i$** changes **with** the value at index $j$ or simply influences the value at index $j$.
* Replacing it with a **kernel matrix** means:
  → **Nearby points influence each other more**,
  → **Farther points have weaker influence** (as defined by the kernel).

This lets the GP **adapt to local variations** in the data more naturally than a fixed covariance.

---

### 5.3 🎓 Practical Example

Imagine you know:

* Student A: 4 hours of study → 60% score
* Student B: 8 hours of study → 90% score
* New student: 7 hours of study

Now you ask: “Will the new student’s result be closer to 60% or 90%?”

> ✅ Clearly, it should be closer to 90%, because **7 is closer to 8** than to 4.

But now imagine the new student studied **8.5 hours** instead. Will that lead to 100% score or 95%?
Clearly, student B alone cannot help us to answer this. The 60% result of Student A is going to help here. Beacuse, to find the new student's result, the influence of Student A’s 60% result on the result of the new student won’t vanish—but it’s just **weaker**.
So we now need a mechanism to control **how quickly influence decays with distance**.

---

### 5.4 📏 The Length-Scale Parameter

Many kernel functions (like the RBF kernel) have a **length-scale** parameter $l$ that governs:

* How **fast** the kernel value drops as distance increases.
* In other words, **how quickly function values stop influencing each other**.

**Effect of Length-Scale $l$:**

* 🔹 **Small $l$** → Very local influence → More **wiggly/complex** functions
  → e.g., at 7 h.o.s., estimate might be $\geq 85\%$
* 🔹 **Large $l$** → Smoother, broader influence → More **averaging**
  → estimate might drop to around $\sim 75\%$

---

<div style="display: flex; justify-content: center; margin: 20px 0;">
    <img src="length_scale_1.png" alt="GP with small length scale" style="width: 48%; margin-right: 2%;"/>
    <img src="length_scale_2.png" alt="GP with larger length scale" style="width: 48%;"/>
</div>

> On the **left**, a smaller length-scale ($l = 1.0$) leads to a more flexible fit.
> On the **right**, a larger length-scale ($l = 2.0$) leads to a smoother function.

---

### 5.5 🔊 Bonus: Noise Variance

Gaussian Processes also add a **"noise variance"** term to the kernel.
This accounts for random noise in the data, preventing the model from **overfitting every point** exactly.

You’ll explore this more in the upcoming **Parameter Estimation** section.

---

### 6. 🧾 Understanding $K^{-1}$: The Inverse Kernel Matrix <a name="kernel-inverse"></a>

### 6.1 🔍 What Does $K^{-1}$ Represent?

The inverse of the kernel matrix $K(X, X)$, denoted $K^{-1}$, plays a crucial role in **solving for predictions** and **isolating the influence of each data point**.

* $K K^{-1} = I$ — the identity matrix
* This "undoes" the interactions encoded in $K$, helping us **disentangle** the overlapping contributions of correlated points

---

### 6.2 📐 Key Mathematical Properties

1. $K(X, X)$ is an $n \times n$ symmetric, **positive semi-definite** matrix
2. $K^{-1}$ exists if $K$ is **positive definite** (non-singular)
3. $K K^{-1} = I$ ensures mathematical **invertibility**

---

### 6.3 🎭 Intuitive Analogy: The Weather Poll

Imagine you're asking three friends about the whether:

* Two are together and say: “☀️ It’s sunny!”
* One, farther away, says: “🌥️ It’s cloudy.”

If you naively count all votes, you think: “Majority says sunny.”

But the two together are likely to **share the same perspective**. So their inputs are redundant.
The **inverse kernel matrix** helps you **deweight redundant opinions**, giving more weight to unique viewpoints.

---

### 6.4 🧮 Matrix Example: Highly Correlated Points

Let’s consider a simple kernel matrix $K$ where two points are highly similar:

$$
K = \begin{bmatrix}
1.0 & 0.9 & 0.1 \\
0.9 & 1.0 & 0.1 \\
0.1 & 0.1 & 1.0
\end{bmatrix}
$$

* Points 1 and 2 are very close (high similarity = 0.9)
* Point 3 is distant from both (low similarity = 0.1)

**Inverse of $K$:**

$$
K^{-1} = \begin{bmatrix}
5.26 & -4.74 & 0.53 \\
-4.74 & 5.26 & 0.53 \\
0.53 & 0.53 & 1.05
\end{bmatrix}
$$

---

### 6.5 🧠 What's Going On Here?

* 🔁 **Redundancy Compensation**:
  The high similarity (0.9) between Points 1 and 2 is penalized in $K^{-1}$ via **negative correlation** (-4.74)

* ✅ **Balanced Weighting**:
  The "cloudy" observation (Point 3) is relatively **independent** and gets a **fair share of influence**

* 🧮 **Mathematical Intuition**:
  The negative off-diagonal terms in $K^{-1}$ act as a **redundancy corrector** — they **reduce the overemphasis** of similar inputs in prediction.

---

### 6.6 ❓ What If Two Points Were Identical?

If two data points are **exactly the same**, then:

* The kernel matrix $K$ becomes **singular** (non-invertible)
* $\det(K) = 0$, so $K^{-1}$ does not exist

✅ **Solution**: Add a **small noise term** to the diagonal — also known as **regularization**.

---

### 6.7 🔧 Numerical Stability: Adding Noise Variance

We stabilize the kernel matrix using:

$$
K_{\text{regularized}} = K + \sigma^2 I
$$

Where:

* $\sigma^2$: Small **noise variance**
* $I$: Identity matrix

This avoids overfitting and allows inversion even in near-singular conditions.

---

<div style="display: flex; justify-content: center; margin: 20px 0;">
    <img src="sigma_001.png" alt="Without Noise Variance" style="width: 48%; margin-right: 2%;"/>
    <img src="sigma_01.png" alt="With Noise Variance" style="width: 48%;"/>
</div>

> 📊 On the **left**, the GP fits the training points **too perfectly** (high confidence, no noise)
> 📉 On the **right**, the GP is **smoother**, accepting that there's **some uncertainty** in the observations.

---

## 7. 🎯 Parameter Estimation: Predicting Mean and Covariance <a name="parameter-estimation"></a>

We now reach the **final step** of the Gaussian Process: using it to predict the **mean** and **uncertainty** for a new test input `x_*`, based on the training data.

---

### 7.1 🤔 What's the Best Estimate for the Target?

If we have a new test point `x_*`, a good guess for the target value `y_*` is a **weighted average** of the training targets `y`, where:

* The weights come from how **similar** `x_*` is to each training point
* But... as we learned earlier, similar training points may give **redundant** information

✅ So, instead of just doing:
→ `similarity × y`

We correct for redundancy using the **inverse kernel matrix**:
→ `similarity × K(X,X)^{-1} × y`

---

### 7.2 🧮 Formula: Predictive Mean

The best estimate of the **mean** for a test input `x_*` is:

```math
\mu(x_*) = K(x_*, X) \, K(X, X)^{-1} \, y
```

Where:

* `K(x_*, X)`: A vector of kernel values between the test point `x_*` and the training data
* `K(X, X)^{-1}`: Inverse of the kernel matrix
* `y`: Training targets

> 🧠 Interpretation: A corrected weighted sum of known outputs, accounting for both similarity and redundancy.

---

### 7.3 📉 What About the Covariance (Uncertainty)?

Initially, before seeing any data, our **uncertainty is high**. But after observing nearby training data, our uncertainty **should drop** — unless that data is redundant.

So we compute:

1. **Prior uncertainty** at the test point: `K(x_*, x_*)` .
2. **Correction term** (what uncertainty we *lose* because of training data):

```math
K(x_*, X) \, K(X,X)^{-1} \, K(X, x_*)
```

> This second term is **larger** if nearby training data is available
> It ensures we **don’t become overconfident** due to repeated (similar) data

---

### 7.4 🧮 Formula: Predictive Covariance

```math
\Sigma(x_*) = K(x_*, x_*) - K(x_*, X) \, K(X,X)^{-1} \, K(X, x_*)
```

Where:

* `K(x_*, x_*)`: Prior variance at the test input
* The second term: Reduction in uncertainty due to training observations

---

### 7.5 🧠 Unifying Intuition

* For **mean**: We used the weights to combine known outputs `y`
* For **variance**: We used the same weights, but this time combining **kernel values** instead of **target values**, to estimate how much variance should be **subtracted**

So:

> Same weights. Different things to multiply them with.

---

## 8. 🧾 Mathematical Formulation <a name="mathematical-formulation"></a>

---

### 8.1 🔢 Complete Gaussian Process Prediction (Single Test Point)

For a new test point `x_*`, the predictive distribution is:

```math
f_* \mid X, y, x_* \sim \mathcal{N}(\mu(x_*), \Sigma(x_*))
```

Where:

```math
\mu(x_*) = K(x_*, X) \, K(X,X)^{-1} \, y
```

```math
\Sigma(x_*) = K(x_*, x_*) - K(x_*, X) \, K(X,X)^{-1} \, K(X, x_*)
```

* `μ(x_*)`: Predictive mean
* `Σ(x_*)`: Predictive variance

---

### 8.2 🔢 Complete Prediction for Multiple Test Points

For a set of multiple test inputs `X_*`, the joint predictive distribution is:

```math
f_* \mid X, y, X_* \sim \mathcal{N}(\mu_*, \Sigma_*)
```

Where:

```math
\mu_* = K(X_*, X) \, K(X,X)^{-1} \, y
```

```math
\Sigma_* = K(X_*, X_*) - K(X_*, X) \, K(X,X)^{-1} \, K(X, X_*)
```

---

### 8.3 📚 Kernel Matrix Notation

* $K(X,X)$: $n \times n$ kernel matrix between all pairs of training points
* $K(X_*, X)$: $m \times n$ kernel matrix between test points and training points
* $K(X_*, X_*)$: $m \times m$ kernel matrix between all pairs of test points
* $K(x_*, x_*)$: Scalar kernel value of the test point with itself

---

## 9. 🛠️ Practical Implementation <a name="implementation"></a>

---

### 9.1 🔁 Step-by-Step Algorithm

1. **Choose a Kernel Function**
   For example, the RBF kernel with parameters `σ²` (signal variance) and `l` (length scale)

2. **Compute the Kernel Matrix**

   ```math
   K = K(X, X)
   ```

3. **Add Noise for Stability**
   Add small noise to the diagonal for numerical stability:

   ```math
   K = K + \sigma_{\text{noise}}^2 I
   ```

4. **Compute the Inverse (or Solve Using Cholesky Decomposition)**

   ```math
   K^{-1}
   ```

5. **For a New Test Point `x_*`**:

   * Compute the cross-kernel vector:

     ```math
     k_* = K(x_*, X)
     ```
   * Predictive mean:

     ```math
     \mu_* = k_*^T \, K^{-1} \, y
     ```
   * Predictive variance:

     ```math
     \sigma_*^2 = K(x_*, x_*) - k_*^T \, K^{-1} \, k_*
     ```

---

### 9.2 🧪 Hyperparameter Learning

The kernel parameters `(σ², l, σ_noise²)` are typically learned by **maximizing the log marginal likelihood**:

```math
\log p(y \mid X) = -\frac{1}{2} y^T K^{-1} y 
                  - \frac{1}{2} \log |K| 
                  - \frac{n}{2} \log(2\pi)
```

> 📌 *Note: I'll soon add the optimization code for this step.*

---

### 9.3 ✅ Advantages of Gaussian Processes

1. **Uncertainty Quantification**
   Provides **confidence intervals** along with predictions
2. **Non-parametric**
   Doesn't assume any fixed form of the function
3. **Bayesian Nature**
   Incorporates prior knowledge via the kernel
4. **Perfect Interpolation**
   Gives exact predictions on training points (if noise is zero)

---

### 9.4 ⚠️ Limitations

1. **Computational Complexity**
   Training: `O(n³)` due to matrix inversion
   Prediction: `O(n)` per test point

2. **Kernel Sensitivity**
   Performance depends heavily on the choice and tuning of the kernel

3. **Scalability**
   Not ideal for **very large datasets** unless approximations (e.g., sparse GPs) are used

---

## 10. ✅ Conclusion <a name="conclusion"></a>

Gaussian Processes offer a **principled and flexible** way to model data, especially when **uncertainty** and **interpretability** are important.

---

### 10.1 🔍 What Makes GPs Special?

By treating functions as samples from a **multivariate normal distribution**, Gaussian Processes allow us to:

1. **Predict with Uncertainty**
   GPs don't just give point estimates — they provide **confidence intervals** around predictions.

2. **Use Prior Knowledge**
   Through the **kernel**, we can encode beliefs about smoothness, periodicity, or other structure in the data.

3. **Avoid Overfitting Automatically**
   The Bayesian nature of GPs means **regularization is built-in** via the marginal likelihood.

---

### 10.2 📈 Where Are GPs Used?

In machine learning, GPs are commonly applied in:

* **Regression problems** (with noisy or sparse data)
* **Bayesian optimization**
* **Hyperparameter tuning**

Anywhere **quantifying uncertainty** matters, GPs shine.

---

### 10.3 🧠 The Core Insight

> The kernel matrix encodes our beliefs about function similarity and smoothness.
> The GP equations then give us **optimal predictions**, assuming those beliefs are true.

---

### 10.4 🌟 Why It Matters

The elegance of GPs is that they offer both:

* **Accurate predictions**
* **Meaningful uncertainty**

This makes them uniquely useful in real-world applications where confidence is as important as accuracy — especially in **scientific discovery**, **autonomous systems**, and **active learning**.

---

### 10.5 🤝 Let’s Connect

📬 If you found this guide helpful, would like to discuss further or believe we might inspire or support each other's growth, I warmly invite you to **connect with me on LinkedIn**:

👉 [LinkedIn profile](https://www.linkedin.com/in/komil-parmar-488967243/)

Or feel free to drop an email:

* `komilparmar57@gmail.com`
* `komil.parmar@op.iitg.ac.in`

Let’s keep learning and growing — together.

---
