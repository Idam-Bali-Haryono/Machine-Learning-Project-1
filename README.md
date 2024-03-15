# Machine Learning Project 
# *Upgini Utilization To Improve Model Accuracy*

![header](https://github.com/Idam-Bali-Haryono/Machine-Learning-Project-1/assets/115137963/6bd6a07d-5318-4b46-9a1f-e09d0aacd284) <img width="622" alt="logo" src="https://github.com/Idam-Bali-Haryono/Machine-Learning-Project-1/assets/115137963/dc533d1e-57a5-4e49-879c-7782efc1caa6"> ![catbooost](https://github.com/Idam-Bali-Haryono/Machine-Learning-Project-1/assets/115137963/0d6b7a07-064e-41fb-88db-81a706d44763)

## Sumarry
This project showcases Upgini's ability to enhance machine learning-based sales forecasting, yielding a notable **7.5%** improvement in model accuracy, thereby demonstrating its potential impact in streamlining predictive processes.


## Introduction

Before diving into the details of Upgini, it's essential to understand that machine learning, a branch of artificial intelligence, involves teaching computers to learn from data and make predictions or decisions without being explicitly programmed. Python, a popular programming language known for its simplicity and versatility, serves as the backbone for many machine learning projects, providing powerful libraries and tools for data analysis and model development.

Upgini, currently in its beta stage, offers a powerful tool for improving machine learning pipelines. It includes a Python library and a smart data search engine designed to make the process easier. The platform brings together data from different sources, including public, community-shared, and premium providers, to help users find what they need. It uses advanced techniques like Large Language Models' data augmentation, RNNs, and GraphNN to generate features automatically, improving the accuracy of machine learning models.

One of Upgini's strengths is its ability to enhance sales prediction models. By automatically discovering features and adding details like postal codes, it enriches the sales model with proven accuracy. It also calculates metrics accurately, giving users a clear understanding of how external features affect model performance. Our ongoing project aims to test Upgini thoroughly and see how it improves model accuracy. By leveraging its unique features, including automated feature generation, we expect to see significant improvements. **This project not only shows Upgini's potential but also sets us up to use it for future enhancements in our sales forecasting processes.**


## Work Flow
### Data 
The dataset provided spans a duration of 5 years and comprises sales data for 50 distinct items across 10 different stores. Each record in the dataset represents a unique combination of an item and a store, capturing the historical sales information over the specified time period. The dataset includes various features such as the item identifier, store identifier, and corresponding sales figures. [access the data](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview)
![dataset](https://github.com/Idam-Bali-Haryono/Machine-Learning-Project-1/assets/115137963/d2585005-4f74-44ec-9e67-8afbfe411fb8)
![dataset2](https://github.com/Idam-Bali-Haryono/Machine-Learning-Project-1/assets/115137963/8082e256-59a9-426e-bb6c-6b39db302334)



### Data Preparation
```python

df_sample = df.sample(n=9_000, random_state=0)
df_sample['store'] = df_sample['store'].astype(str)
df_sample['item'] = df_sample['item'].astype(str)

df_sample['date'] = pd.to_datetime(df_sample['date'])
df_sample.sort_values(['date'], inplace=True)
df_sample.reset_index(inplace=True, drop=True)
```
A sample dataset (df_sample) has been created by randomly selecting 9,000 entries from the original dataset (df). This subset is then modified: the 'store' and 'item' columns are converted to strings, the 'date' column is transformed to datetime format, and the dataset is sorted chronologically based on the 'date' column. The resulting sample dataset is ready for further analysis.


```python
train = df_sample[df['date'] < '2017-01-01']
test = df_sample[df['date'] >= '2017-01-01']

train_features = train.drop(columns=['sales'])
train_target = train['sales']
test_features = test.drop(columns=['sales'])
test_target = test['sales']
```

Here, we divides df_sample into training and testing sets based on a specified date threshold, creating separate sets of features (excluding the 'sales' column) and corresponding target variables for both training and testing datasets.

### Upgini
``` python
%pip install -Uq upgini
from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType

enricher = FeaturesEnricher(
    search_keys= {
        "date": SearchKey.DATE,
    },
    cv = CVType.time_series
)

enricher.fit(train_features,
             train_target,
             eval_set =  [(test_features, test_target)])
```
Here, we utilizes the upgini library to create and train a FeaturesEnricher instance for time series data, with a specified search key for the "date" feature and specifying the data type as time series (CVType.time_series). After setting up the enricher, it fits the model using training features (train_features), training targets (train_target), and evaluation set ((test_features, test_target)). In this case, upgini enrich the data set with Nasdaq score, Crude Oil Price, Gold Price, Weather data, etc.
![upgini](https://github.com/Idam-Bali-Haryono/Machine-Learning-Project-1/assets/115137963/f9e30a22-9b4c-47a6-a2ce-47e20ea44428)


### Catboost 


CatBoost is a high-performance, open-source gradient boosting library designed for efficient machine learning on tabular data. Notable for its excellent handling of categorical features, CatBoost requires minimal preprocessing, supports GPU acceleration, and provides built-in features for preventing overfitting. It is user-friendly and widely used for regression and classification tasks.
```python
pip install -Uq catboost
from catboost import CatBoostRegressor

model = CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=0)

```


### Result
To assess the impact of Upgini on the model, we are employing the differentiation in Mean Absolute Percentage Error (MAPE).
```python

enricher.calculate_metrics(
    train_features, train_target, eval_set=[(test_features, test_target)],
    estimator =  model,
    scoring = "mean_absolute_percentage_error"
```
*output:*


*Calculating accuracy uplift after enrichment...*


MAPE |  Enriched MAPE | Uplifts 
--- | --- | ---
0.240629 | 0.165222 | 0.075407


Upon scrutinizing the tabulated results, a discernible enhancement in the predictive accuracy of the CatBoost model is evident. Specifically, the Mean Absolute Percentage Error (MAPE) demonstrates a notable reduction from 24% before data enrichment to 16.5% after enrichment, signifying a substantial 7.5% uplift in forecasting precision.

## Conclusion

**This project exemplifies Upgini's prowess in enhancing machine learning-driven sales forecasting processes, _showcasing a notable 7.5% improvement in Mean Absolute Percentage Error (MAPE)_, underscoring its impactful contributions to predictive accuracy and performance.**
