# Continuous Coordinate-Set Diffusion for Unconditional Image Random Functions

## 1. Problem definition
We model a continuous random field $f:[0,1]^2\to\mathbb{R}^3$. For any finite coordinate set $S=\{u_i\}_{i=1}^n$, define finite-dimensional marginal
\[
p_\theta(f(S)) = p_\theta(f(u_1),\ldots,f(u_n)).
\]
A valid resolution-agnostic generator should provide consistent marginals under restriction: for nested sets $S_c\subset S_f$,
\[
\text{Proj}_{S_c}(p_\theta(f(S_f))) \approx p_\theta(f(S_c)).
\]

## 2. Model
We use a coordinate-conditioned diffusion score network:
- global branch: CNN encoder produces context $z_g$ from training image.
- local branch: coordinate Fourier features + noisy RGB at coordinates + time embedding + $z_g$ are mapped by MLP to predicted noise.
- training inputs are random coordinate subsets (regular and irregular).

This is **not** fixed-grid latent diffusion + terminal decoder: denoising model directly operates on random finite coordinate sets $S$ during training and sampling.

## 3. Diffusion objective
Given clean values $x_0=f(S)$ and noise $\epsilon\sim\mathcal{N}(0,I)$, forward process
\[
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon.
\]
Main loss:
\[
\mathcal{L}_{\text{main}} = \mathbb{E}_{S,t,\epsilon}\|\epsilon-\epsilon_\theta(x_t,S,t,z_g)\|_2^2.
\]

## 4. Subset/coarse-to-fine consistency
For $S_c\subset S_f$, we add
\[
\mathcal{L}_{\text{cons}} = \mathbb{E}\|\epsilon_\theta(x_t^{(c)},S_c,t,z_g)-\epsilon_\theta(\tilde x_t^{(c)},S_c,t,z_g)\|^2,
\]
and optimize
\[
\mathcal{L}=\mathcal{L}_{\text{main}}+\lambda\mathcal{L}_{\text{cons}}.
\]

## 5. Bayesian inverse problems
Observation model:
\[
y=\mathcal{A}(f)+\eta.
\]
Tasks (inpainting/super-resolution/denoising/sparse coordinates) correspond to different $\mathcal{A}$. Posterior
\[
p(f|y)\propto p(y|f)p_\theta(f).
\]
Implementation provides MAP-like posterior sampling by optimizing image variable initialized from the diffusion prior sample:
\[
\hat f=\arg\min_f \underbrace{\|\mathcal{A}(f)-y\|^2}_{-\log p(y|f)} + \beta R_\theta(f),
\]
where $R_\theta$ is weak prior regularization; framework is extensible to score guidance / DPS.

## 6. Simplifications and limits
- current sampler reconstructs regular grid for export; irregular-coordinate ancestral sampler can be further strengthened.
- mixed dataset training placeholder can be enabled via concatenated dataloaders.
- FFHQ/CelebA-HQ download may require manual placement due licensing/auth constraints; script surfaces explicit errors.
