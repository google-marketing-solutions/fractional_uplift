# Fractional Uplift
### Flexible uplift modelling with complex or constrained metrics.
##### This is not an official Google product.

## Introduction

Fractional Uplift is a flexible Python package for uplift modelling with meta learners, which can be used to target adverts, promotions or other treatments towards the most incremental customers, in order to maximise the return on investment (RoI) or another KPI. 

Fractional Uplift has been designed with the following goals in mind:

1. It is designed to optimise for complex metrics or metrics with constraints, which typically cannot be optimized with regular uplift modelling solutions. This is especially important when optimizing promotions with costs that can vary per customer.
2. It is designed to be flexible, so that it can be used with any machine learning and data processing packages. 
3. It is designed to make the resulting models easy to deploy, by using [Knowledge Distillation [1]](https://arxiv.org/abs/1503.02531) to distill the complex meta learners into a single ML model.

## Installation

```
$ pip install fractional_uplift
```

## Quick Start

First you can load some example data and train a fractional uplift model:

```python
import tensorflow_decision_forests as tfdf
import fractional_uplift as fr
import pandas as pd


# Load example dataset
criteo = fr.example_data.CriteoWithSyntheticCostAndSpend.load()
train_data = criteo.train_data
distill_data = criteo.distill_data
test_data = criteo.test_data


# Create the training dataset, and define the KPI's so that
# you are maximizing conversions with a target RoI.

# Note: you must set the treatment propensity, but if your data is coming from
# a randomised experiment or A/B test then this will be a constant column, where
# every row is the fraction of traffic in the treatment group. 

train_data = fr.datasets.PandasTrainData(
    features_data=train_data[criteo.features],
    maximize_kpi=train_data["conversion"].values,
    constraint_kpi=train_data["cost"].values,
    is_treated=train_data["treatment"].values,
    treatment_propensity=train_data["treatment_propensity"].values,
    sample_weight=train_data["sample_weight"].values,
    shuffle_seed=1234
)

# Fit the model using a tensorflow decision forest as the base learner
base_learner = fr.base_models.TensorflowDecisionForestRegressor(
    tfdf.keras.GradientBoostedTreesModel, 
    init_args=dict(verbose=2, max_depth=6, num_trees=300, shrinkage=0.1),
    fit_args=dict(verbose=2)
)
fractional_learner = fr.meta_learners.FractionalLearner(base_learner)
fractional_learner.fit(train_data)
```

The uplift model is a meta learner, so it is a combination of many ML models. This can be hard to maintain and deploy, 
so it can be beneficial to distill the uplift model into a single ML model with [knowledge distillation [1]](https://arxiv.org/abs/1503.02531).

```python
# Create a separate dataset for the distillation, to avoid overfitting.
distill_dataset = fr.datasets.PandasDataset(
    features_data=distill_data[criteo.features]
)

# Distill the model
distilled_model = fr.base_models.TensorflowDecisionForestRegressor(
    tfdf.keras.GradientBoostedTreesModel, 
    init_args=dict(verbose=2, max_depth=6, num_trees=300, shrinkage=0.1),
    fit_args=dict(verbose=2)
)
fractional_learner.distill(distill_dataset, distilled_model)
```

Next, you can analyze the performance of the uplift models. It's best to evaluate both the original and distilled models:

```python
# Predict with both models on the test data
test_dataset = fr.datasets.PandasDataset(
    features_data=test_data[criteo.features]
)
test_data["fractional_learner_score"] = fractional_learner.predict(test_dataset)
test_data["distill_model_score"] = distill_fractional_t_learner.predict(test_dataset)


# The analysis method is very generic, and can be subclassed to 
# create any evaluation metrics. For example, here we want to evaluate the iRoI
# as a function of the incremental conversions.

class RoIUpliftEvaluator(fr.evaluate.UpliftEvaluator):
  def __init__(self, **kwargs):
    kwargs["metric_cols"] = ["spend", "conversion", "cost"]
    super().__init__(**kwargs)

  def _calculate_composite_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
    data["roi__inc_cum"] = data["spend__inc_cum"] / data["cost__inc_cum"]
    data["roi__inc"] = data["spend__inc"] / data["cost__inc"]
    return data


# Run the evaluation
evaluator = RoIUpliftEvaluator(
    is_treated_col="treatment", 
    treatment_propensity_col="treatment_propensity"
)

models = {
    "fractional_learner_score": "Fractional Learner",
    "distill_model_score": "Distilled Model",
}

results = evaluator.evaluate(criteo.test_data, score_cols=list(models.keys()))
```
The variable `results` is a Pandas dataframe you can inspect and plot, containing 
the cumulative incremental spend, incremental conversions, cost and RoI as a 
function of model score thresholds for both models.

Once you are happy with your model, you can export the distilled model to be deployed as a regular tensorflow decision forest model.

```python
exported_model = distill_model.export()
```

## How does fractional uplift compare with other uplift modelling packages?

If your goal is to estimate the conditional average treatment effect (CATE) of a single metric, then there are a variety of alternative python packages which are more appropriate, such as [CausalML](https://causalml.readthedocs.io/en/latest/), [EconML](https://github.com/Microsoft/EconML) and [UpliftML](https://github.com/bookingcom/upliftml). Typically, optimizing promotions based on the CATE of a single metric is only optimal when you want to maximise a single KPI and your treatment has deterministic costs or no cost. For example, you have two different designs for your landing page and you want to show some users one landing page and others another one, to maximise conversion rate. 

However, many use-cases for uplift modelling do not meet this criteria, especially those where the costs of your treatment vary per customer, for example a discount or promotion that the customer may or may not use, or where the value might vary depending on what they purchase. In this case, simply targeting customers with the highest CATE might not be optimal, because those customers might also have higher costs. 

In these cases, Fractional Uplift can be used to find the most optimal segments of customers to target, by directly modelling the CATE of the costs of your treatment as well as your target KPIs. This is a generalisation of the fractional approximation approach proposed by [Goldenberg, Albert, Bernardi and Estevez [2]](https://dl.acm.org/doi/10.1145/3383313.3412215). It works by estimating the following function, which is generalised from the a composition of the Conditional Average Treatment Effects (CATE's) of three different metrics:

```math
f_\delta(X)= 
\begin{cases}
    \frac{\text{CATE}_\alpha (X)}{\text{CATE}_\beta(X) - \frac{\text{CATE}_\gamma (X)}{\delta}},& \text{CATE}_\beta(X) > \frac{\text{CATE}_\gamma (X)}{\delta}\\
    \infty,              & \text{otherwise}
\end{cases}
```

Where the CATE of a metric $y$ is defined as:

```math
\text{CATE}_y(X) = E[y | T=1, X] - E[y | T=0, X]
```

Where $T$ indicates whether the sample was treated, and $X$ is a set of predictors. 
The four metrics $\alpha$, $\beta$, $\gamma$ and $\delta$ are referred to as:

* $\alpha$ is the "Maximize KPI" - the KPI to be maximized
* $\beta$ is the "Constraint KPI" - the KPI that acts as a constraint, and we want to keep as low as possible.
* $\gamma$ is the "Constraint Offset KPI" - the KPI that can offset the constraint. This is optional.
* $\delta$ is the "Constraint Offset Scale" - a constant which scales the constraint_offset_kpi. Not needed if there is no constraint_offset_kpi.

Many business optimisation problems can be posed as uplift modeling problems like this, by choosing the appropriate Maximize, Constraint and Constraint Offset KPIs. Perhaps the most common case is discussed in detail by [Goldenberg, Albert, Bernardi and Estevez [2]](https://dl.acm.org/doi/10.1145/3383313.3412215). They consider a problem where you want to use a promotion to maximize the number of converting customers, while meeting an ROI constraint. Their solution can be rephrased as a fractional uplift modelling problem by targeting customers based on $f_\delta(X)$ where:

* Maximize KPI = The number of conversions
* Constraint KPI = The cost of the treatment
* Constraint offset KPI = The revenue from that customer
* Constraint offset scale = The target RoI

See `examples/end_to_end_example.ipynb` for more examples of how fractional uplift can be used to solve different problems.

## Data requirements

Training a fractional uplift model requires user level data, containing examples of users who were exposed to a specific treatment, and examples where they were not. 
For best results, this data should be generated from a randomised controlled trial (A/B test), but it can work with observational data too. 

### Data from a randomised controlled trial (recommended)

The input data must be a table, where each row is a different customer, and with the following schema:

- `is_treated` (int): Was the customer exposed to the treatment, 1 = yes, 0 = no
- `treatment_propensity` (float): The probability that this user would be assigned to the treatment group. As this data comes from a randomised controlled trial, the treatment propensity will be the same for every user, and should be equal to the fraction of traffic in the experiment where the treatment was applied.
- `maximize_kpi` (float): The value of the maximize KPI for this customer.
- `constraint_kpi` (float): The value of the constraint KPI for this customer.
- `constraint_offset_kpi` (float): The value of the constraint offset KPI for this customer (optional).
- `feature_1` ... `feature_n`: (float or string): The features to be used to predict the uplift (multiple columns).

To avoid leakage, all features should be known **before** the treatment was applied to the user. 
For example, if the treatment is offering a discount to a customer when they reach a certain page on your website, then the features should be 
information you knew about that user before they landed on that page. This means you could use "time_on_site_before_landing_on_page", 
but you couldn't use "total_time_on_site" because that will include the time after the user saw the discount.

### Observational data

You want exactly the same schema, but now the `treatment_propensity` will not be a constant. Because you are using observational data, some users will be more likely to have been treated than others. 
Therefore, prior to fitting the uplift model, you will need to fit a propensity model to predict the treatment propensity. Make sure to use cross-fitting, to ensure that you don't overfit to the training data.

## Available learners

| Learner                        | Summary                                                                                                                                                                                                                                                                                                                                                                              | Constraints                          | When to use                                                                                                                                                                                                            | Reference |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| TLearner                       | A simple meta learner for  estimating the CATE of a  single metric.  Uses two models, one to predict the  target metric in the treatment group,  and another to predict the  target metric in the control group.                                                                                                                                                                     |                                      | Mainly used as a baseline for comparison  with the other models. If you are using this,  it's likely you should be using another  package like CausalML, EconML or UpliftML.                                           | [3]       |
| RetrospectiveLearner           | A simple meta learner for estimating the  **relative** CATE of a single metric.  It does this by training a single model  to predict the probability that a  positive sample was seen in control or  treatment, where the samples are weighed by the target metric.  This is a very efficient algorithm, as it only requires rows where the target KPI is non-zero to fit the model. | The target KPI must always be >= 0   | Under certain assumptions, this model can be  considered equivalent to the more general fractional learners below, but requires training fewer models with less data.   See [1] for a discussion on when this applies. | [2]       |
| FractionalLearner              | This model combines multiple T-Learners,  to estimate the CATE of the Maximize,  Constraint and Constraint Offset KPIs.   This makes it a very general purpose  algorithm for solving fractional uplift  modelling problems.                                                                                                                                                         |                                      | Use this for any problem that can be formulated  with a Maximize, Constraint and Constraint Offset KPI.                                                                                                                | This work |
| FractionalRetrospectiveLearner | This is a generalisation of the  RetrospectiveLearner, to make it work  with the general case of optimizing with  a Maximize, Constraint and Constraint Offset KPIs.  Similarly to the RetrospectiveLearner, this only requires data where any of the three target KPIs are >= 0. Rows where all KPIs are 0 are not used by the model.                                               | All target KPI's must always be >= 0 | Use this for any problem that can be formulated  with a Maximize, Constraint and Constraint Offset KPI,  but the complete dataset is too large  for the FractionalLearner.                                             | This work |

### Fractional Learner Algorithm

#### Stage 1

Train up to 6 ML models:

```math
\alpha_i(X) = E[\alpha | T=i, X] \quad \text{for} \, i \in [0, 1]
```

```math
\beta_i(X) = E[\beta | T=i, X] \quad \text{for} \, i \in [0, 1]
```

```math
\gamma_i(X) = E[\gamma | T=i, X] \quad \text{for} \, i \in [0, 1]
```

When training these models, the samples must be weighed by the inverse propensity weights, $w_\text{ipw}$. These control for the likelihood that the sample was treated. If the data was generated in a randomised experiment, then the inverse propensity weights will be fixed and always equal to the inverse of the fraction of traffic that was assigned to the treatment group (for example 2.0 if it was a 50/50 split, or 4.0 if only 25% of traffic was treated). For an observational study they will need to be learned with a propensity model. 

#### Stage 2

Define $f_\delta(X)$ as:

```math
f_\delta(X) = \frac{\alpha_1(X) - \alpha_0(X)}{\beta_1(X) - \beta_0(X) - \frac{\gamma_1(X) - \gamma_0(X)}{\delta}}
```

### Fractional Retrospective Learner Algorithm

Here we transform the problem from regression problems into classification problems.

#### Stage 1

Train 1 ML model per component to learn the incrementality of each component. 
These models use an adaptation of the retrospective estimation method described by [Goldenberg, Albert, Bernardi and Estevez [2]](https://dl.acm.org/doi/10.1145/3383313.3412215), where the problem is transformed into estimating:

```math
y_\alpha = \frac{E[\alpha | T=1, X]}{E[\alpha | T=1, X] + E[\alpha | T=0, X]}
```

```math
y_\beta = \frac{E[\beta | T=1, X]}{E[\beta | T=1, X] + E[\beta | T=0, X]}
```

```math
y_\gamma = \frac{E[\gamma | T=1, X]}{E[\gamma | T=1, X] + E[\gamma | T=0, X]}
```

These can be models as simple binomial classification problems. For example, to estimate $y_\alpha$, you would train a classifier with the following training data:

* Features: X
* Label: T
* Weights: $\alpha \times w_\text{ipw}$

Here $w_\text{ipw}$ are the inverse propensity weights, the same as in the Fractional Learner Algorithm.

Note: The classifier used must be [calibrated](https://www.unofficialgoogledatascience.com/2021/04/why-model-calibration-matters-and-how.html), meaning it must produce a score that can be interpreted as a probability.

#### Stage 2

Now we need to learn the weights of the three components, $\alpha$, $\beta$ and $\gamma$:

```math
w_\alpha = \frac{E[\alpha | X]}{E[\alpha | X] + E[\beta | X] + E[\gamma | X]}
```

```math
w_\beta = \frac{E[\beta | X]}{E[\alpha | X] + E[\beta | X] + E[\gamma | X]}
```

```math
w_\gamma = \frac{E[\gamma | X]}{E[\alpha | X] + E[\beta | X] + E[\gamma | X]}
```

We learn these using a single multi-class classifier on an augmented dataset. We create three copies of the training data:

Dataset 1:

* Features: X
* Label: 0
* Weights: $\alpha \times w_\text{ipw} $

Dataset 2:

* Features: X
* Label: 1
* Weights: $\beta \times w_\text{ipw}$

Dataset 3:

* Features: X
* Label: 2
* Weights: $\gamma \times w_\text{ipw}$

Then we concatenate these and train the mutli-class classifier on the resulting dataset. 
The outputs of the multiclass classifier, $p_\text{class=c}(X)$ can then be interpreted as:

```math
w_\alpha = p_\text{label=0}(X)
```

```math
w_\beta = p_\text{label=1}(X)
```

```math
w_\gamma = p_\text{label=2}(X)
```

#### Stage 3

Now we can calculate $f_\delta(X)$ as:

```math
f_\delta(X) = \frac{(2 y_\alpha (X) - 1) \, w_\alpha (X)}{(2 y_\beta (X) - 1) \, w_\beta (X)  - \frac{(2 y_\gamma (X) - 1) \, w_\gamma (X)}{\delta}}
```

## Citing Fractional Ulift

To cite this repository:

```
@software{fractional_uplift,
  author = {Sam Bailey and Christiane Ahlheim},
  title = {Fractional Uplift: Flexible uplift modelling with complex or constrained metrics.},
  url = {https://github.com/google/fractional_uplift},
  version = {0.0.1},
  year = {2023},
}
```

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
2. Dmitri Goldenberg, Javier Albert, Lucas Bernardi, Pablo Estevez Castillo. Free Lunch! Retrospective Uplift Modeling for Dynamic Promotions Recommendation within ROI Constraints. In Fourteenth ACM Conference on Recommender Systems (pp. 486-491), 2020.
3. Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 2019.
