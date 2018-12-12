<a href="https://www.drivendata.org/">
  <img src="https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png" alt="DrivenData Logo">
</a>
<br><br>

![Banner Image](https://s3.amazonaws.com/drivendata-public-assets/se-challenge-1-banner.jpg)

#  Power Laws: Cold Start Energy Forecasting

## Goal of the Competition

Building energy forecasting has gained momentum with the increase of building energy efficiency research and solution development. Indeed, forecasting the global energy consumption of a building can play a pivotal role in the operations of the building. It provides an initial check for facility managers and building automation systems to mark any discrepancy between expected and actual energy use. Accurate energy consumption forecasts are also used by facility managers, utility companies and building commissioning projects to implement energy-saving policies and optimize the operations of chillers, boilers and energy storage systems.

Usually, forecasting algorithms use historical information to compute their forecast. Most of the time, the bigger the historic dataset, the more accurate the forecast. This requirement presents a big challenge: how can we make accurate predictions for new buildings, which don't have a long consumption history?

The goal of this challenge is to build an algorithm which provides an accurate forecast from the very start of a building's instrumentation.

The best algorithms were generally ensembles comprised of deep learning and XGBoost models. The winners all thought carefully about how to combine the limited historical consumption information provided with useful meta-data (for example, holidays) correlated to consumption trends.

## What's in this Repository
This repository contains code from winning competitors in the [Power Laws: Cold Start Energy Forecasting](https://www.drivendata.org/competitions/55/schneider-cold-start/) DrivenData challenge.

#### Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).


## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | ---
1 | last_minute_team | 0.272444    | 0.257776   | Teamed up near the end of the competition to go from 3rd to 1st using an ensemble of LSTMs, custom neural networks to handle variable length inputs, and linear regression. Built solution in Python on top of Google’s Tensorflow library. Inferred holidays from training data and built custom list of holidays. Identified buildings with similar patterns and created clusters. Identified 5 buildings with anomalous data, and removed those from the model training and evaluation.
2 | valilenk | 0.289542    | 0.259703 | Tried many models, but simplified final solution to a single neural network, built solution in Python on top of fast.ai library and Facebook’s PyTorch library. Single model predicts hourly forecasts, and then he aggregates for daily and weekly (simplest implementation of the winners). Created working day and holiday features.
3 | LastRocky | 0.298222 | 0.261503 |Trained model on training data, and cold-start data. Tuned hyperparameters by validating performance on the cold start data (so model is tuned to the cold-start buildings). Prediction models, built one LSTM based neural network and one lightgbm model. Built both an hourly model and a daily model (daily model used for weekly predictions as well) Built solution in Python on top of Google’s Tensorflow and Keras libraries.

Short write-ups describing solutions can be found under the `reports/` directory in each competitor's code. For the first place team, addition slides describing the solution are included at the top level of this repository.

#### Benchmark Blog Post: ["Benchmark - How To Use An LSTM For Timeseries And The Cold-Start Problem"](http://drivendata.co/blog/benchmark-cold-start-lstm-deep-learning/)
