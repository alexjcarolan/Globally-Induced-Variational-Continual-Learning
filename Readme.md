# Globally Induced Variational Continual Learning

The field of artificial intelligence has undergone a transformation, as advancements in deep learning continue to expand the horizons of what it can achieve. Although significant progress has been realised, artificial intelligence still falls short of human intelligence, most notably perhaps when it comes to continual learning. This is the ability to acquire knowledge not just of a single task but to accumulate knowledge from multiple sequential tasks, whilst only ever having access to the current task data. Variational continual learning implements this by leveraging the intrinsically continual process emerging from the recursive application of Bayesian inference, which combines what the current task data indicates about the model parameters, with what previous task data indicated about the model parameters. Naturally this can be combined with Bayesian neural networks which apply Bayesian inference over the model parameters, such that the wide variety of complex tasks typically solved by artificial neural networks, may also be solved in a continual setting. Typically exact Bayesian inference can’t be performed and instead an approximate inference scheme must be chosen. The objective of this project is to build on the work of variational continual learning, by merging it with the recent work of global inducing point variational inference. The hypothesis being that because global inducing points act as pseudo data points, they could behave as an episodic memory for the model, thereby increasing performance. Extensive detail can be found in the associated dissertation [1](Dissertation.pdf).

### Theory

The extension of artificial neural networks from a frequentist scheme to that of a Bayesian scheme, with the inclusion of uncertainty in the parameters and predictions, produces a Bayesian neural network. Constructed through the fusion of Bayesian networks and artificial neural networks, with the intention being to combine the probabilistic guarantees and continuous function approximation of each, into a single model. Whilst artificial neural networks effectively perform maximum likelihood estimation, Bayesian neural networks naturally perform Bayesian inference, thereby representing the parameters of the network not as point estimates but rather as probability distributions. The mean field variational inference scheme used in the variational continual learning research papers, assumes that the joint distribution over the parameters of the network can be factorised out into independent distributions, thereby averaging over the degrees of freedom. In contrast the global inducing point variational inference scheme, defines a set of global inducing points, which act as pseudo data points in order to induce a distribution over the parameters of the network. A number of advantages arise from this inference scheme, the most significant of which is that it models cross layer correlations, as global inducing points are defined jointly for all layers, and therefore model not only the input to output transformation of individual layers but the input to output transformation of the whole network. Furthermore the parameters of the network are defined over function space rather than parameter space, which provides a more intuitive interpretation as it is the function rather than the parameters that is usually of interest.

|Artificial Neural Networks|Bayesian Neural Networks|
|:------------------------:|:----------------------:|
|![](plots/ann.png)|![](plots/bnn.png)|

### Practice

|Standard|Permuted|
|:------:|:------:|
|![](plots/standard.png)|![](plots/permuted.png)|

### Results

|Results|
|:-----:|
|![](plots/original_results.png)|

|Evidence Lower Bound|Logistic Loss|Predictive Accuracy|
|:------------------:|:-----------:|:-----------------:|
|![](plots/original_evidence.png)|![](plots/original_loss.png)|![](plots/original_accuracy.png)|