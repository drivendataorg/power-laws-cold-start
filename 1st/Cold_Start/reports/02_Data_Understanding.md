# Data Understanding
## Collect initial data
<!---Acquire the data (or access to the data) listed in the project resources.
This initial collection includes data loading, if necessary for data understanding.
For example, if you use a specific tool for data understanding, it makes perfect
sense to load your data into this tool. This effort possibly leads to initial data
preparation steps.
List the dataset(s) acquired, together with their locations, the methods used to
acquire them, and any problems encountered. Record problems encountered and any
resolutions achieved. This will aid with future replication of this project or
with the execution of similar future projects.

>	Indeed it's a pain downloading huge files. Especially when there are connection issues. I used "wget" to download the dataset with an option "-c" for resuming capability in case the download fails.  You would need to save the cookies in the page using a chrome extension Chrome Extension  save the cookies as cookies.txt from the extension  Then you can download the files by using the following command

	wget -c -x --load-cookies cookies.txt https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data?train_wkt.csv.zip


--->

The data was downloaded easily and it's lightweight.

## External data
<!--- It is allowed in this challenge? If so write it here ideas of how to find
it and if people have already posted it on the forum describe it. --->

External data is allowed but the rules are not clear.

> Per the problem description as long as the model and weights that can be released under and open source license, you do not need to share which pretrained models you are using ahead of time.
https://community.drivendata.org/t/pre-trained-models-and-external-data/2597

## Describe data
<!---Describe the data that has been acquired, including the format of the data,
the quantity of data (for example, the number of records and fields in each table),
the identities of the fields, and any other surface features which have been
discovered. Evaluate whether the data acquired satisfies the relevant requirements. --->

### Data

* `series_id` - An ID number for the time series, matches across datasets
* `timestamp` - The time of the measurement
* `consumption` - Consumption (watt-hours) since the last measurement
* `temperature` - Outdoor temperature (Celsius) during measurement from nearby weather stations, some values missing

### Metadata
* There are 7 different categories of surface
* Two categories of base-temperature

## Explore data
<!---This task addresses data mining questions using querying, visualization,
and reporting techniques. These include distribution of key attributes (for example,
the target attribute of a prediction task) relationships between pairs or small
numbers of attributes, results of simple aggregations, properties of significant
sub-populations, and simple statistical analyses.

Some techniques:
* Features and their importance
* Clustering
* Train/test data distribution
* Intuitions about the data
--->
### Train set
* All the series have 4 weeks of data.
* There are big differences in consumptions between buildings, 4 orders of magnitude. I should
probably normalize the numbers to train the model
* The distributions of data seem similar on train and test
* The range of temperature is between -10 and 40. There are big differences between buildings

### Test set
* The number of days available has a uniform distribution from 1 to 14
* The type of submissions are: 192, 191, 242 for weekly, daily and hourly
* There doesn't seem to be a relation between the type of prediction and the available data

<img src="media/test_submission_plot.png" width="800"/>
This plot shows the relation between the test data (blue) and the required prediction (orange)

### Repeated buildings
There is a very strong evidence supporting that the number of buildings is smaller than the number of series. I have seen that both on data and metadata.

I have prepared visualizations of the clusters of train data. I get the following conclusions:
* In some buildings days off are equal to normal days. 100517 For those cases having
more data is very useful.
* Many situations are completely impossible to predict. 103496 Maybe holidays?
* Some clusters have very similar elements


## Verify data quality
<!---Examine the quality of the data, addressing questions such as: Is the data
complete (does it cover all the cases required)? Is it correct, or does it contain
errors and, if there are errors, how common are they? Are there missing values in
the data? If so, how are they represented, where do they occur, and how common are they? --->
About 40% of the temperatures are missing, the rest of the data is clean.

## Amount of data
<!---
How big is the train dataset? How compared to the test set?
Is enough for DL?
--->
The train set has 758 series with 672 timestamps each. That is about 3k of weeks.
It's not enormous, but we could try with DL.
