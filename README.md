# Meta-Learned Basis Adaptation for Parametric Linear PDEs

Official implementation for the paper:

**Meta-Learned Basis Adaptation for Parametric Linear PDEs**
![KAPI overview](FIG_OPENER.png)

## Abstract
We introduce a hybrid physics-informed framework for parametric linear PDEs that combines a meta-learned predictor with a least-squares corrector. The full predictor, called KAPI (Kernel-Adaptive Physics-Informed meta-learner), is a shallow task-conditioned model that maps a query point and PDE parameters to a predicted solution value. Its task dependence is mediated through a lightweight internal meta-network that maps PDE parameters to an interpretable task-adaptive Gaussian basis geometry. The corrector then reuses this predictor-generated basis structure in an enriched hidden dictionary and solves the PDE through a one-shot physics-informed Extreme Learning Machine (PIELM) style least-squares step. Unlike iterative residual-adaptive refinement methods, which are typically tied to a single PDE instance, the proposed framework amortizes basis adaptation across an entire parametric family and deploys it in one shot at inference time after offline meta-training. We evaluate the method on four PDE families spanning diffusion-dominated, transport-dominated, mixed advection--diffusion, and variable-speed transport regimes. The predictor alone already captures meaningful physics by identifying localized source regions and transport-aligned spacetime structure, while the corrector further improves accuracy, often by one or more orders of magnitude, including in several extrapolative settings. Comparisons with parametric PINNs and physics-informed DeepONet, together with ablations against uniform-grid PIELM correctors and single-instance PINNs, show that the improvement arises specifically from predictor-guided basis adaptation. Overall, the proposed framework provides an interpretable and efficient approach for parametric PDE solving.


## Citation

If you find this repository useful, please cite:

```bibtex
