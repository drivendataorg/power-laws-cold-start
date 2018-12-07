# Modeling

## Select modeling technique
<!---Document the actual modeling technique that is to be used. If multiple
techniques are applied, perform this task separately for each technique.
Many modeling techniques make specific assumptions about the data—for example,
that all attributes have uniform distributions, no missing values allowed,
class attribute must be symbolic, etc. Record any such assumptions made. --->

## Generate test design
<!---Describe the intended plan for training, testing, and evaluating the models.
A primary component of the plan is determining how to divide the available dataset
into training, test, and validation datasets.

Doing a plot of score vs train size could be helpful to decide the validation strategy

Depending on the size of the data we have to decide how we are going to use submissions.
The less the submissions the most confidence we can have on the score. However sometimes
the data distribution is very different, or the size of the data is small and we have
to make a lot of submissions. Sometimes is not easy to have a good correlation between
validation score and LB score
--->
In this problem we have an open field for preparing the validation strategy. Let's
first describe what we found on the test set:
* Between 1 and 14 days of data
* A prediction that could be 24 hours, 7 days, 2 weeks.

In the other hand the train dataset has 4 weeks of data of each series id. Let's try
to get a formula of how many evaluations can we make on the train set.

## Iteration 1. Baseline
<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

### Goal
The goal of the first iteration is to see which score we can get by simply repeating
the past data.

### Development
I have make a first try that tried to use the same weekday for making predictions and
otherwise returns an average of the same type of day (working and off.)

I have also tried to use simply the closest day.

### Results
I get scores of 0.3704 and 0.358 with this strategy. This is much better than the
score of 0.52 that is the competition baseline.

I have found that there are two series with outliers that have to be removed because
they greatly affect the mean values of the scores. On following experiments I will
be removing 102571, 101261 series.
<img src="media/outliers_plot.png" width="800"/>

I have also found that there is a big difference between working days and days off.
I have prepared two plots showing the similarity between a day and the previous ones.

<img src="media/working_days_similarity_matrix.png" width="800"/>
<img src="media/off_days_similarity_matrix.png" width="800"/>

If we make an average over the val_idx we get the following plot for working days.
<img src="media/working_days_error_vs_days_diff.png" width="800"/>
It is clear that the closer the day the better the prediction. We can also see
some periodicity in multiples of 7 but that's all.


## Iteration 2. Linear regression
### Goal
My goal is to reframe the problem on this iteration. On the baseline I defined
hand crafted rules for repeating past data for making predictions. I think that is a
good approach, but instead of defining rules I think we need to learn those rules.

The idea is to divide the problem into multiple subproblems. I have seen that there
is a great difference between working and days off. So I'm going to use that as my
separation criteria.

For example If I have to predict for a working day and I have two previous days one
working and the other off I will encode that as 001. I will gather all those cases
on the train set and compute the optimum weights for combining the past days to predict
the present one.

My idea is to use Lasso regression forcing the coefficients to be positive. Also
I will be limitting the size of the input to 7 days. That means I will train 2**8=256 models
on the worst case.
The number of parameters will be smaller than 256*7=1792. We will have at least 758*21=15918 samples
for training. So it seems that the number of parameters is not too big.

### Development
On a first step I have to gather train data and prepare it for training. It will be interesting
to find how many different keys can we find on the train and test set.

### Results
We have improved the score from 0.3580 to 0.3336, current rank is 5.

| method                                             	| LB score 	|
|----------------------------------------------------	|----------	|
| simple_repeat                                      	| 0.3704   	|
| even_simple_repeat                                 	| 0.3580   	|
| even_simple_repeat + hourly with linear regression 	| 0.3483   	|
| linear regression                                  	| 0.3379   	|
| linear regression + test set                       	| 0.3336   	|

### Analysis
We have to analize the weaknesses of this approach to be able to improve it.

I can think of the following ideas for improving:
* Try using more than 7 input days
* Think of how to use trends in data
* Using holidays may also improve
* Adding more capacity to the model may help


## Iteration 3. Holidays hunting
### Goal
The goal is to search for holidays in the train and test dataset that may help
to improve the predictions.

### Development
I'm going to prepare a visualization of all the data to see if there are clear
holidays.

<img src="media/holiday_hunting.png" width="800"/>

I have identified typical holidays. The only complication was to detect easter holidays.

### Results
The score is 0.3288, nice improvement on LB. So holidays matter. My current rank
is 3rd so I'm on the money.

<img src="media/rank_3_20180929.png" width="800"/>

## Iteration 4. Input days on Linear regression
### Goal
The goal is to study the influence of the number of input days in the score of the model.
In the study of the results of the previous implementation of the linear model it was quite
clear that the score improved with the number of days until the 7 day. So maybe just by
adding more days I can improve the score.

### Development
### Validation strategy
On a first step I have to take a decission for dividing the train set between train and validation.
This model has a significant number of parameters so I can't just say that I'm using all train set.

The split has to be done based on series_id. I know that there are clusters of series_id. However
I don't think I have to do a hard separation because there are also clusters on test set and I know
them. So a random split or taking a instance every n would be good.

The split should be useful in the future when using more powerful methods. So I'm thinking of leaving
20% of the data for validation. That would be 5 splits If I want to make cross-validation.

### Results
The best result happens when the number of input days is 7. So there is no score gain on
this iteration.

## Iteration 5. Using clusters
### Goal
The goal is to study if I can use cluster information to improve linear regression.
I'm going to train a model for each cluster.

### Development
I have made a notebook for implementing this.

### Results
When using just the cluster models the score was worse. But when averaging
with the general model the score improved to 0.3213.

| method                                             	| LB score 	|
|----------------------------------------------------	|----------	|
| simple_repeat                                      	| 0.3704   	|
| even_simple_repeat                                 	| 0.3580   	|
| even_simple_repeat + hourly with linear regression 	| 0.3483   	|
| linear regression                                  	| 0.3379   	|
| linear regression + test set                       	| 0.3336   	|
| linear regression + test set + holidays               | 0.3288   	|
| linear regression + clusters                          | 0.3213   	|

This probes that the clusters have relevant information for this challenge.

## Intermezzo. Learnings from Linear regression
The strategy of using a weighted combination of the previous days has been succesfull.
I don't know if it is the best way to solve this problem but I think it's possible
to improve the current result by using more information.

I think I have to continue with this strategy until I find that I can't improve more. At
that point I could switch to predicting the output instead of weighting the input. I could
also predict a weight for each of the timesteps, maybe that is better than creating an output
from zero.

I have seen that using the keys of working day/day off has proven to be a very good feature.
However I would like to add more information such us the trends in data, temperature or metadata.

So I think the next iterations should be about using taylor made models to predict weights for averaging
the input to predict the output.


## Iteration 6. Rise of Taylor Made Models
### Goal
The goal of this iteration is to train taylor made models using only a binary
input with working/off days. I expect to have same or similar results to the linear
model. If that is true I could add more information to the input for improving the
score.

### Development
#### Preparing the data
The first step will be to prepare functions for collecting data and addapting it
to the format required for training NN. I could use linear regression functions as
a start point.
#### Taylor made model
The idea is to have two inputs to the model. One input will be the binary representation
of working-day/day off that was used as a key on previous linear model. The second input
will be the past days that we want to combine to create the new prediction.

#### MetaModel
The metamodel will have multiple models, and they will be organized using a dictionary
with two levels: window and number of input days.
I will have a function for preparing input for the model, that will also be a dictionary.
The model will use the name of its inputs to select the relevant data for input.

So I need to implement the MetaModel and a function for preparing the data.

### Results
The score is 0.3342, similar to the score I get when using linear regression without holidays.
It's a good start point.

I think it's time to close the iteration. I have probe that we can get similar score using
taylor made models. I see two challenges:
* Optimization is harder, so there is room for improvement there
* I should try to add more inputs and see if results improve.


## Iteration 7. Adding more inputs and improving optimization
### Goal
The goal is to explore how much can we improve the score but using more information
as input for the model and by improving the training process.

### Development
I have implemented a method for training multiple models in parallel. However it has
suddenly stopped and moreover the score I got on LB was bad 0.3257.

I'm going to modify the method so each worker loads the dataset, and I will be
training for 5 folds.

I'm getting the following error:

    Exception in thread Thread-2:
    Traceback (most recent call last):
      File "/home/guillermo/miniconda2/envs/coldstart/lib/python3.6/threading.py", line 916, in _bootstrap_inner
        self.run()
      File "/home/guillermo/miniconda2/envs/coldstart/lib/python3.6/threading.py", line 864, in run
        self._target(*self._args, **self._kwargs)
      File "/home/guillermo/miniconda2/envs/coldstart/lib/python3.6/concurrent/futures/process.py", line 272, in _queue_management_worker
        result_item = reader.recv()
      File "/home/guillermo/miniconda2/envs/coldstart/lib/python3.6/multiprocessing/connection.py", line 251, in recv
        return _ForkingPickler.loads(buf.getbuffer())
    TypeError: __init__() missing 3 required positional arguments: 'node_def', 'op', and 'message'

I could try making a python2 environment. When using the python 2 the errors were more verbose and I
have found the problem. The problem was that on the process pool I was creating new sessions with
different gpus on the same process. That was causing the problems. However we can see that the output
was very uninformative. Only by using joblib and python=2 I was able to discover the issue.

That fact that it worked during all the night was because I was lucky that two processes did not end at the
same time until all night passed.

### Results
On a first step I have trained with 4 different folds and made a submission averaging them.
The score was 0.3060, very close to 1 and 2 position.
I see two ways for improving:
* Architecture tuning
* Adding more inputs such as temperature

Using more than 7 days of input does not improve the scores.

I have used cluster_id ohe as input and it greatly improves the validation error. However
when making the submission the result is worse. I have to average it with a normal prediction
to be able to improve the score, when doing that my score was 0.3006 very close to first position.

I have tried using weights of 0.4 and 0.6 for merging the submissions but I get similar worse scores.

By tuning the architecture I have been able to lower the score to 0.3033. When combining it with models trained with cluster_id ohe scoring 0.3039 I get a combined score of 0.2959 which is much better than the score using linear regression 0.3213.

| method                                             	| LB score 	|
|----------------------------------------------------	|----------	|
| simple_repeat                                      	| 0.3704   	|
| even_simple_repeat                                 	| 0.3580   	|
| even_simple_repeat + hourly with linear regression 	| 0.3483   	|
| linear regression                                  	| 0.3379   	|
| linear regression + test set                       	| 0.3336   	|
| linear regression + test set + holidays             | 0.3288   	|
| linear regression + clusters                        | 0.3213   	|
| taylor NN                                           | 0.3030   	|
| taylor NN + cluster_id ohe                          | 0.2959   	|

So great improvement on this iteration. The pity is that now I'm on third position. The leader has 0.2918 and the second one 0.2927.

My validation scores show that using cluster_id ohe I'm able to improve the validation
score by 0.02. So I think there is room for improvement if I create a better representation
for the clusters.

## Iteration 8. Better representation for cluster
### Goal
The goal of this iteration is to create a representation of the cluster better than ohe.
This will help to generalize better to the test set that has less data.

### Development
One way of creating this representation is computing metrics about the behaviour of consumption
on the cluster.

### Results
I have created a first series of features that allow to get a score of 0.2949 using a single set
of models. Previously my best score with a single set of models was 0.3030. So this is a great
improvement. The pity is that when mixing with the submission that does not use cluster information
the improvement is very small (0.2945)

Currently there are 3 ways of improvement:
* visualize errors and try to understand why they are produced.
* improvements in architecture
* radical changes in modelling

#### Reflexions over the errors
* Many errors are impossible to avoid, simply the data is too random.
* However on hourly predictions I see a disconection between the end of the last day and
the start of the current day. That kind of behaviour won't happen when using LSTM.
Maybe I can think of giving something as input that could help to improve those situations.

The visualizations are available at:
/media/guillermo/Data/DrivenData/Cold_Start/visualizations/2018_10_19_cluster

#### Architecture optimization
I'm going to analize the architecture search to narrow the future search.

## Iteration 9. LSTM
### Goal
The situation is desperate. Forces of the empire have seized the 3 first positions
with astonishing ease. I don't see a clear path for improving my taylor made NN.

I have seen that on hourly window continuty between last day and prediction day
is not preserved. That should probably be captured easily by LSTM.

So the goal of this iteration is to explore the use of LSTM for hourly window. If
I'm able to improve my previous scores then I will try to do it for the other time windows.

Hopefully by using LSTM I will be able to improve 0.01 to catch with the leader.

<img src="media/rank_4_20181020.png" width="800"/>

### Development
I don't have too much experience with LSTM. So I'm going to use this 11 days remaining
to learn as much as possible from this recurrent models.

Let's enumerate some doubts.
* How to deal with inputs with different lenght?
* How to make predictions for 24 timesteps?
* How to arrange the train data?
* How to normalize the data?
* Which cost function to use?

#### Learnings about LSTM
* Elements inside the batch are independent
* When using stateful=True the i element on the batch will be connected to the i
element of the following batch
* Using many timesteps is a way to initialize the state of the model. However if
I want to use stateful I should use timestep=1 otherwise it will be hard to connect
the batches.
* To reset the states of your model, call .reset_states() on either a specific layer, or on your entire model.
* If I want to make predictions for more than one timestep I have to provide as input the output
of the model on the previous timestep.
* I could simple run the model over all timesteps at train. However that won't explore all coldstart
options. So I should probably do it randomly. Or maybe train many models like I did previously. However
I think it has more sense to train a single model


**Can I set timesteps to None?**
What if I use 2 timesteps as input instead of 2 features?
    I don't think that has sense when using stateful.

**Can I set batch_size to None?**
No if I use stateful

**Using bigger batches?**
It is possible however that affects also to prediction. One alternative is to create a new model with batch size 1 and transfer the weights.
I have seen that using a batch size bigger than one is beneficial.

**Good start point?**
Initialization of the model state before fitting or predicting is very important.

#### Different input sizes
I believe that I should use different models depending on the input size. And the reason is that a model
having 7 days of input must learn long term relationships whereas a model with one day of input does not.

Moreover this arises the question of how the model will learn those long term relationships if I only use
a timestep of 1. Maybe using stateful model is only useful when making the predictions but not for training.
I asume that when using a model with multiple timesteps as input the gradient is propagated through those timesteps.
I could make an experiment using a simple input like being zero all the time except for multiples of 10.
My hipothesis is that a model with a timestep of 1 won't be able to learn it, that a model with a large enough timestep
will be able to learn it and that I can transfer the weights to a stateful model with timestep=1 and make it work.

I have seen that with stateful and timestep=1 the model is unable to learn long range relationships.
This means that I should use a timesteps as big as possible.

I have tried turning the model to stateful for prediction but it does not work and I don't understand why.
If that becomes an issue I will work on it later.

I should probably prepare a generator for feeding data.

#### Batch size
I have experienced that using a big batch size speeds up the training by many factors.

| batch size 	| epoch time (s) 	|
|------------	|----------------	|
| 32         	| 284            	|
| 64         	| 169            	|
| 128        	| 92             	|
| 256        	| 40             	|
| 512        	| 21             	|
| 1024       	| 10             	|

It is also obvious that when using a bigger batch size there are less weight updates per
epoch, however the train remains faster. An extreme example is a model that took almost a
day yesterday now I can train it in 2 hours (Also lr policy is better because more frequent evaluations)

### Results
I have been able to make a submission with a score of 0.3088. So no improvement over previous models.

The problem is that LSTM only predicts one timestep and I need to predict 24. However I have learned a lot
from LSTM and I know how to fix this: with seq2seq. Because the problem is a seq2seq problem, so it makes
sense to use a seq2seq model.

I have also noticed a new problem that may arise on LB. I have seen great discrepances when evaluating
on validation set between simple_arrange and whole_arrange. (0.1683 vs 0.2016). I have to consider that on first
case 300 instances are evaluated and on the second one about 4200. The test size is 600, so maybe the score
is not very trustable.

## Iteration 10. seq2seq
### Goal
The goal of this iteration is to implement a seq2seq pipeline that hopefully will
improve my score. If it does I will close the iteration and open a new one for tuning.

### Development
On a first step I want to play with seq2seq just like I did previously with LSTM. I will like
to learn the sin function and also the clock function.

### Results
I have been able to improve the score to 0.2881. Now I'm back on 3rd position. I have discrepancies
between the train score and validation score. They may be caused by the normalization but I have to
understand them.

## Iteration 11. Frankenstein architecture
### Goal
The goal of this iteration is to make a mixture of taylor NN, lstm and seq2seq to try to improve
daily and weekly predictions. I believe some features are better encoded by NN and others by LSTM.

I will focus on daily and weekly, but I could also apply it to hourly.

The taylor-made NN used a mask over consumption to make predictions. Now I will be using a raw prediction
hoping that merging the models will be better because they are more different.

### Development
On a first step I have to play with the layers to understand how can I create the frankenstein architecture.

#### Reframing the problem
The problem is a seq2seq problem. We have different input and output sizes.

* Input sizes: From 1 day to 7 days
* Output sizes: 24, 7 and 2

Also I have features that do not change over time: cluster features.

The thing is that in canonical seq2seq architecture the model learns a state that later is used
to create the prediction. However in this problem I also have to give hints to the output like is_day_off.

#### The frankenstein architecture
I think that using LSTM to create a representation of the past is the good way. So one of the inputs
to the model will be the past consumption, days off and weekday. The ouptut of this first stage
will be state of the encoder.

On a next step a decoder will take the state of the encoder as its initial state. It will receive as input
the days off and weekday of the prediction window. It will output features useful for predicting consumption
in the desired window.

The final step will be a NN that will receive as input the output of the decoder and the cluster features
and it will make predictions of the consumption.

#### Implementation
On a first step I will do it on keras, checking that outputs of the layers are as desired. Next I will
check seq2seq implementation to see if I can tune it. I have interest in the peek=True parameter.

### Results
Even when on validation scores I saw improvements over the taylor model when combining the submissions
the result was worse. This is very unfortunate.


## Iteration 12. MegaEnsemble
### Goal
The goal is to create a megaensemble for the final submission.

### Development
On a first step I have to develop a new cross-validation function that allows to
use different random seeds.

### Results

## Iteration 1. Iteration_title
### Goal

### Development

### Results