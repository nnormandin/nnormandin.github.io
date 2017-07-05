---
layout: post
title:  "The YellowFin optimizer: analysis and implementation in keras"
date:   2017-07-01
categories: ["science"]
---

YellowFin is a new type of optimizer for deep neural networks that is purported to achieve faster convergence and better results than state-of-the-art optimizers like Adagrad, Adam, and RMSProp in many domains. I'm going to:

- quickly review the concept of optimizers in deep learning
- give a summary of YellowFin and the concepts behind it
- review the TensorFlow implementation code, and how I hacked together Keras compatibility
- test it on some more obscure data to see how it performs 'in the wild'
<br><br>

## optimization in deep learning

The choice of optimizer (and associated parameters) in the process of training a deep neural network plays a large role in the rate of convergence and the overall loss of the model. An optimizer that converges slowly is computationally costly, and one that gets stuck in local minima will prevent the parameters from being tuned optimally. The standard in deep learning is the family of *gradient descent* optimizers.

Generally speaking, given a model with \\(d\\) parameters \\(\theta \in \mathbb{R}^d\\) you need to minimize a loss function \\( J(\theta) \\). We can do this by calculating the gradient of the loss function with respect to our parameters, \\(\nabla_\theta J(\theta)\\), and then descending towards a lower value by moving in the opposite direction of the gradient. There's an excellent [**overview of gradient descent optimization algorithms**](http://sebastianruder.com/optimizing-gradient-descent/) by Sebastian Ruder that goes into great detail. He's updated it as of June 2017 to include *AdaMax* and *Nadam*.
<br><br>

## *YellowFin* and momentum tuning

Introduced by Jian Zhang, Ioannis Mitliagkas, and Christopher Ré in the paper [**YellowFin and the Art of Momentum Tuning**](https://arxiv.org/abs/1706.03471), this method is novel due to the tuning of the *momentum parameter*.

Most state-of-the-art methods store gradients from previous time steps with some decay parameter to form a 'momentum' term that is combined with the gradient in the current time step. However, the parameters for the momentum term (eg. \\(\beta_1\\) and \\(\beta_2\\) for the *Adam* optimizer) are static during training. Here are my key takeaways from the *YellowFin* paper:

- Recent [**literature**](https://arxiv.org/abs/1705.08292) has suggested that algorithms conducting element-wise gradient tuning yield marginal benefits to vanilla stochastic-gradient descent. Furthermore, models trained with these techniques actually generalize more poorly.
- In another recent [**paper**](https://arxiv.org/abs/1605.09774), Mitliagkas et al. showed that asynchronous training introduces 'momentum-like dynamics' into optimization (amplifying the momentum that has already been specified in the parameters of the chosen optimizer).
- The authors have found that tuning momentum as part of the optimizer parameter search improves convergence, but note that conducting a grid-search for the correct momentum and learning rates is challenging. Therefore, they propose *automatic momentum tuning*.
- The authors demonstrate that *momentum is robust to learning rate misspecification*. The source of this property is that the spectral radius of the momentum operator, \\(\rho(\mathbf{\mathit{A}}_t)\\) is constant in some subset of the hyperparameter space. The spectral radius of a matrix \\(\rho(\mathit{X})\\) is the largest absolute value of its eigenvectors.

	In fact (for a 'one dimensional strongly convex quadratic objective'), the spectral radius is constant at \\(\rho(\mathbf{\mathit{A}}_t) = \sqrt\mu\\) as long as

	$$(1-\sqrt\mu)^2/h \leq \alpha \leq (1+\sqrt\mu)^2 / h$$

	where \\(\mu\\) is the momentum parameter with learning rate \\(\alpha\\) and curvature \\(\mathit{h}\\). This means that as long as the learning rate falls within the 'robust region' identified in the inequality above, the asymptotic behavior will be the same. The authors point out that this allows them to focus on freely tuning the momentum parameter for optimality, given that the above condition is satisfied.

- The authors next extend the results regarding learning rate robustness to functions other than strongly smooth convex ones in order to demonstrate *robustness to curvature*. They define a 'Generalized condition number' or GCN, which is a measure of curvature along a scalar slice of a function. They derive optimal hyperparameters \\(\mu^\*\\) and \\(\alpha^\*\\), but do not provide a convergence rate guarnatee. Instead, they provide 'empirical evidence to support this intuition.'
- Using a noisy one-dimensional quadratic to model gradient descent with a noisy gradient (is in mini-batch descent), the authors develop tuning rules for the YellowFin optimizer on learning rate \\(\alpha\\) and momentum \\(\mu\\). The rules are dependent on an estimate of the model's distance to a local minima \\(\mathit{D}\\), an estimate for gradient variance \\(\mathit{C}\\), and the largest/smallest generalized curvatures \\(\mathit{h_{max}}\\) and \\(\mathit{h_{min}}\\).
	
	At each step, the authors solve what they call the *'SingleStep'* equation, which is:

	$$ \mu_t,\alpha_t = \underset{\mu}{\mathrm{argmin}}( \mu D^2 + \alpha^2C) $$

	subject to:

	$$ \mu \geq (\frac{\sqrt{h_{max}/h_{min}}-1} {\sqrt{h_{max}/h_{min}}+1})^2 $$

	and:

	$$\alpha = \frac{(1-\sqrt\mu)^2}{h_{min}}$$

- Using these rules, the authors demonstrate evidence of superior performance on CIFAR100 on ResNet:

![results image](/assets/images/yellowfin_result.png)

<br><br>

# making the TensorFlow version work in Keras


# benchmark: financial time series forecasting