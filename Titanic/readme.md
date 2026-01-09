# Titanic Survival Prediction

Uses a Random Forest Classifier to predict survival of passengers with 83.8% accuracy.

## Survival Trends

Using the 'Survived' column in the training data, create charts of how many people survived vs how many died.
![](figure_1.png)

Using the 'Survived' and 'Sex' columns in the training data, create charts of how many people survived based on their Sex.
![](figure_2.png)

## Feature Engineering
Feature Engineering refines the data by mapping categorical data into numerical form, populating null values with mode values for that category, and by removing irrelevant (ie non predictive) features.

Since 'Cabin', 'Name', and 'Ticket' have no predictability, they were immediately dropped from both test and training datasets.

All null values of 'Embarks' were populated with 'S'.

'Age' was converted into a categorical group, 'AgeGroup, where numerical values were assigned and mapped, and missing values were filled in using mode for each category. 'Age' was then dropped, as 'AgeGroup' is more useful for predictability.

Numerical values were also assigned and mapped to 'Sex' and 'Embarks'.

All missing values in 'Fare' were filled out based on the mean fare for that 'Pclass'. Then 'Fare' was mapped into quartiles (into 'FareBand' column) and 'Fare' was dropped.


## Model Training & Evaluation

Using 'Survived' and 'PassengerId' from the training dataset as predictors and identifying 'Survived' as the prediction target, split the training data 80/20 (train/test).
Then, initalize the Random Forest classifier and fit the training data to it. Track the target predictions using `RandomForestClassifier.predict()`.

Calculate the model accuracy score by `round(accuracy_score(yPredictions, yValueActual)*100,2)`. This model has an accuracy score of **83.8%**

Next, predict survival based on passenger id by using `RandomForestClassifier.predict()` on our test data without 'PassengerId' and output results to [results.csv](results.csv).