# Structure Learning Hyperparameters

This table below describes the hyperparameters that are supported by one or more CausalIQ structure learning algorithms.

| Hyperparameter                    | Type       | Description |
|-----------------------------------|------------|-------|
| [alpha](#alpha)                   | constraint | p-value threshold used in Conditional Independence tests |
| [ci_test](#ci_test)               | constraint | Conditional Independence tests used in constraint-based learning |
| [iss](#iss)                       | constraint | Imaginary Sample Size used to weight priors in Bayesian scores |
| [max_elapsed](#max_elapsed)       | general    | Maximum elapsed time allowed in seconds |
| [max_iterations](#max_iterations) | score      | Limit on the number of iterations in score-based learning
| [no_increase](#no_increase)       | score      | Number of iterations allowed where the score does not increase |
| [penalty_weight](#penalty_weight) | score      | Weighting assigned to the penalty component of scores such as AIC and BIC |
| [score](#score)                   | score      | Objective score used in score-based learning |
| [tabulist_len](#tabulist_len)     | score      | Length of the Tabu list |


## General hyperparameters

### max_elapsed

Maximum allowed elapsed execution time in seconds specified as a positive integer.


## Score-based hyperparameters

### iss

A real value defining the Imaginary Sample Size (ISS) used to weight the prior in Bayesian scores.

### max_iterations

A positive integer placing a limit on the number of iterations.

### no_increase

A non-negative integer if iterations where the score is allowed not to increase.

### penalty_weight

A positive real number weighting the penalty component of the AIC and BIC scores relative to the log. likelihood.

### score
Defines the objective score used by score-based algorithms and the score-based phases of hybrid algorithms. Supported values are:

- `aic`: Aikike Information Criterion
- `bdeu`: Bayesian Equivalent Uniform
- `bge`: Bayesian Gaussian Equivalent
- `bic`: the Bayesian Information Criterion
- `k2`: K2
- `loglik`: Log Likelihood

### tabulist_len
A positive integer defining the length of the Tabu list which contains the highest-scoring recently visited graphs.

## Constraint-based hyperparameters

### alpha
A positive real-value below 1.0 which defines the p-value threshold below which the CI test is interpreted as indication conditional independence.

### ci_test
Defines the conditional independence (CI) test used by constraint-based algorithms and the constraint-based phases of hybrid algorithms. Supported values are:

- `mi`: Mutual Independence test
- `x2`: Chi-squared test