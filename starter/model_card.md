# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is created to predict the income levels of individual.
Used logistic regression as baseline performance. Further improvements can be made using
different ensemble of machine learning algorithms.
Used scikit learn logistic regression model and default parameters.

## Intended Use
To predict income level if whether above $50,000/year or not.
This data comes mostly from America. So, use it with a grain of salts.

## Training Data
This data comes with UCI machine learning dataset library.
It is a census of income data.
* Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.

## Evaluation Data
We used same dataset as evaluation. We split 80, 20 ratio of dataset to train and test.

## Metrics
We use precision, recall and fbeta to get model performance metrics.

Overall Performance
- Precision: 0.7315270935960592
- Recall: 0.26271561256081377
- Fbeta: 0.3865929059550927

Slice Performance
Category: sex, Value: Male
- Precision: 0.812199036918138
- Recall: 0.2661756970015781
- Fbeta: 0.4009508716323296
Category: sex, Value: Female
- Precision: 0.5668449197860963
- Recall: 0.3045977011494253
- Fbeta: 0.3962616822429907

## Ethical Considerations
As the data collection does not represent worldwide, we should be careful to use prediction.
The metrics are not satisfactory so we should only use it for educational purposes.
Also, the gender bias is apparent here as slice performance indicates high variance.

## Caveats and Recommendations
We need to improve the model by exploring other machine learning algorithms.
Ensemble models may improve the metrics greatly.
