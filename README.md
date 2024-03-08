# Machine Learning Project
# Upgini Utilization To Improve Model Accuracy 

## Abstract
## Introduction

**Upgini**, currently in its beta stage, introduces a powerful Python library and an intelligent data search engine tailored for elevating **machine learning** pipelines. This platform seamlessly integrates relevant features from diverse data sources, categorizing them into public, community-shared, and premium providers. Employing advanced techniques such as Large Language Models' data augmentation, RNNs, and GraphNN, Upgini ensures automated feature generation that transcends mere correlation, significantly improving machine learning model accuracy. With dynamic search key augmentation, accurate metric calculations, and a focus on addressing stability concerns in external data dependencies, Upgini empowers users to efficiently enrich and stabilize machine learning models.

Leveraging Upgini's innovative capabilities becomes particularly impactful in enhancing sales prediction models. The platform's automated feature discovery and search key augmentation, including elements like postal codes, offer a comprehensive enrichment of the sales model with accuracy-proven features. The ability to calculate accurate metrics provides a quantifiable assessment of the impact of external features on the performance of machine learning models. In our ongoing project, we rigorously evaluate Upgini's influence on improving machine learning model accuracy. Through systematic testing and leveraging Upgini's unique features, including automated feature generation, we anticipate witnessing substantial enhancements in accuracy metrics and uplifts. **This project not only serves as a testament to Upgini's potential but strategically positions us to harness its capabilities for ongoing improvements in our machine learning-based sales forecasting processes.**


## Methods
#### Data
The dataset provided spans a duration of 5 years and comprises sales data for 50 distinct items across 10 different stores. Each record in the dataset represents a unique combination of an item and a store, capturing the historical sales information over the specified time period. The dataset includes various features such as the item identifier, store identifier, and corresponding sales figures. 
[access the data](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview)
#### Data Preparation
```python
df_sample = df.sample(n=9_000, random_state=0)
df_sample['store'] = df_sample['store'].astype(str)
df_sample['item'] = df_sample['item'].astype(str)

df_sample['date'] = pd.to_datetime(df_sample['date'])
df_sample.sort_values(['date'], inplace=True)
df_sample.reset_index(inplace=True, drop=True)
```
A sample dataset (df_sample) has been created by randomly selecting 9,000 entries from the original dataset (df). This subset is then modified: the 'store' and 'item' columns are converted to strings, the 'date' column is transformed to datetime format, and the dataset is sorted chronologically based on the 'date' column. The resulting sample dataset is ready for further analysis, with information regarding data types and structure provided by the df_sample.info() function.

#### Upgini
``` python
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
#### Catboost
CatBoost is a high-performance, open-source gradient boosting library designed for efficient machine learning on tabular data. Notable for its excellent handling of categorical features, CatBoost requires minimal preprocessing, supports GPU acceleration, and provides built-in features for preventing overfitting. It is user-friendly and widely used for regression and classification tasks.
```python
pip install -Uq catboost
from catboost import CatBoostRegressor
from catboost.utils import eval_metric

model = CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=0)

```
Dataset |  MSE | Enriched MSE 
--- | --- | ---
Train | `renders` | **nicely**
Eval1 | 2 | 3




#### Symmetric Mean Absolute Percentage Error (SMAPE)

### Result

### Conclusion
