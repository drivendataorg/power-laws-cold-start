# Business Understanding
<!--- --->
## Challenge description
<!--- Look at the challenge description, understand the goal of the challenge
and write it here with your own words. Use images if they improve the explanation--->

The objective of this competition is to **forecast energy consumption** from varying amounts of "cold start" data, and little other building information. That means that for each building in the test set you are given a small amount of data and then asked to predict into the future.

<img src="media/mlscheme.png" width="800"/>

Three time horizons for predictions are distinguished. The goal is either:

* To forecast the consumption for each hour for a day (24 predictions).
* To forecast the consumption for each day for a week (7 predictions).
* To forecast the consumption for each week for two weeks (2 predictions).

In the test set, varying amounts of historical consumption and temperature data are given for each series, ranging from 1 day to 2 weeks. The temperature data contains a portion of wrong / missing values.

## Evaluation
<!--- Understand the metric used on the challenge, write it here and study
the characteristics of the metric --->

The performance metric is a normalized version of mean absolute error.

$$NMAE = \frac{1}{N} \sum_{i=1}^N|\hat{y_i} - y_i|c_i $$

* $N$ - The total number of consumption predictions submitted, including all hourly, daily, and weekly predictions
* $\hat{y_i}$ - The predicted consumption value
* $y_i$ - The actual consumption value
* $c_i$ - The normalization coefficient that weights and scales each prediction to have the same impact on the metric

The normalization coefficient $c_i$ for the $i^{th}$ prediction is composed of a ratio of two numbers,

$$ c_i = \frac{w_i}{m_i}$$

* $w_i$ is a weight that makes weekly (24 / 2), daily (24 / 7), and hourly (24 / 24) predictions equally important. This means that weekly predictions are 12 more important than hourly.
* $m_i$ is the true mean consumption over the prediction window under consideration (this mean is unknown to competitors). As far I understand for hourly prediction the mean of the 24 hours is used, for daily prediction the mean of the 7 days and for weekly prediction the mean of the 2 weeks.

Multiplying predictions by this coefficient makes each prediction equally important and puts hourly, daily, and weekly predictions on the same scale.

## Assess situation
<!---This task involves more detailed fact-finding about all of the resources,
constraints, assumptions, and other factors that should be considered in determining
the data analysis goal and project plan

* timeline. Is there any week where I could not work on the challenge?
* resources. Is there any other project competing for resources?
* other projects. May I have other more interesting projects in the horizon?
 --->

I have just finished my holidays, so I have 7 weeks without interruptions for the challenge.

The only competing project is the rocket, but it should be finished soon.

### Terminology
<!--- Sometimes the field of the challenge has specific terms, if that is the
case write them here, otherwise delete this section.--->


## Project Plan
<!--- Write initial ideas for the project. This is just initial thoughts,
during the challenge I will have a better understanding of the project and
with better information I could decide other actions not considered here.--->

I need to first understand the data. LSTM could be a good choice for time-series.