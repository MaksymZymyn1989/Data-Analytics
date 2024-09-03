# 1 Data cleaning
''' IKEA Database.
Download this IKEA dataset. '''

import re
from collections import defaultdict
from io import StringIO

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as stsm
from statsmodels.tools import add_constant
import statsmodels.formula.api as smf
from sklearn.utils import resample
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import xgboost as xgb
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu, wilcoxon, ranksums, kruskal, zscore
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

def fetch_data(url):
    response = requests.get(url)

    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text))
        return data
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None


# Define the constant URL
link = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv"

# Call the function
data = fetch_data(link)

# Search for duplicates
# 1) Screening out duplicates by 'item_id'
unique_data = data.drop_duplicates(subset=['item_id'], keep='first').copy()

# 2) Search for duplicates by 'designer'
unique_data['designer'] = unique_data['designer'].apply(lambda x: "/".join(sorted(x.split("/"))))

designer_indices = defaultdict(list)

for i, designer in enumerate(unique_data['designer']):
    sorted_designer = "/".join(sorted(designer.split("/")))
    designer_indices[sorted_designer].append(i)

duplicate_indices = [indices for indices in designer_indices.values() if len(indices) > 1]

designer_duplicates_data = unique_data.iloc[np.concatenate(duplicate_indices)].reset_index(drop=True)

# 3) Search for duplicates by 'designer' and 'depth', 'height', 'width'
same_dims_data = []

for i, row in designer_duplicates_data.iterrows():
    dims = sorted([float(x) if isinstance(x, str) and not pd.isna(x) else x for x in [row['depth'], row['height'], row['width']]])
    key = tuple(dims)
    row['dims'] = str(key)
    row['link'] = re.sub(r'\d+', '', row['link'])
    same_dims_data.append(row)

same_dims_data = pd.DataFrame(same_dims_data)
same_dims_data = same_dims_data[same_dims_data['dims'].duplicated(keep=False)]

# 4) Search for duplicates by 'depth', 'height', 'width' and 'name', 'price', 'old_price', 'designer'
grouped_data = same_dims_data.groupby(['name', 'price', 'old_price', 'designer']).filter(lambda x: len(x) > 1)

# 5) Find duplicates by 'name', 'price', 'old_price', 'designer', 'depth', 'height', 'width' and 'short_description'
description_duplicates = grouped_data[grouped_data['short_description'].str.contains(r'\d+')]
description_duplicates = description_duplicates.copy()
description_duplicates['short_description'] = description_duplicates['short_description'].apply(lambda x: "/".join(sorted(re.findall(r'\d+', x))))
description_duplicates = description_duplicates.groupby(['name', 'price', 'old_price', 'designer', 'short_description']).filter(lambda x: len(x) > 1)

description_duplicates = pd.concat([description_duplicates,
                                    grouped_data[grouped_data['short_description'].str.contains(r'\d+') == False].groupby(['name', 'price', 'old_price', 'designer']).filter(lambda x: len(x) > 1)])

''' 6) Search for duplicates by 'name', 'price', 'old_price', 'designer', 'depth', 'height', 'width', 'short_description' and 'link' '''
color_names = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'purple', 'pink', 'brown',
               'light-grey', 'dark-grey', 'dark-brown', 'golden', 'brown', 'grey', 'beige', 'brass-colour',
               'dark-red', 'light-brown', 'light-beige', 'orange', 'light-antique', 'steel-colour',
               'anthracite', 'antique', 'turquoise', 'multicoloured-dark', 'multicolour']

color_pattern = r'\b(' + '|'.join(color_names) + r')\b'

for i, row in description_duplicates.iterrows():
    color_matches = re.findall(color_pattern, row['link'])
    if color_matches:
        new_text = '/'.join(sorted(color_matches))
        description_duplicates.loc[i, 'link'] = new_text

same_link = (description_duplicates.groupby(['name', 'price', 'old_price', 'designer', 'short_description', 'link']).filter(lambda x: len(x) > 1))

# Sampling for exploratory analysis
same_link_item_ids = same_link['item_id']
analysis_data = unique_data[~unique_data['item_id'].isin(same_link_item_ids)]

# Display the cleaned data in Markdown format
print(analysis_data.to_markdown(index=False))

# 7) Remove outliers
# Price outliers
def remove_price_outliers(df):
    # Calculate quartiles and interquartile range (IQR)
    q1 = df['price'].quantile(0.25)
    q3 = df['price'].quantile(0.75)
    iqr = q3 - q1

    # Calculate lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter rows without outliers
    return df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# Remove outliers based on 'price' across the entire DataFrame
data_without_price_outliers = remove_price_outliers(analysis_data).copy()

# Display the cleaned data in Markdown format
print(data_without_price_outliers.to_markdown(index=False))

# 2 Descriptive statistics
''' Perform exploratory analysis on a dataset including
descriptive statistics and visualizations. Describe the results. '''

# Dataset information
print(analysis_data.shape)
print(analysis_data.info())

# Descriptive statistics
print(analysis_data.describe())

# Descriptive statistics by price
price_descriptive_stats = analysis_data["price"].describe()

# Convert to Markdown
price_descriptive_stats_markdown = price_descriptive_stats.to_markdown()
print(price_descriptive_stats_markdown)

plt.hist(analysis_data["price"], bins=10)
plt.title("Price statistics")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.axvline(analysis_data["price"].mean(), color='b', linestyle='dashed', linewidth=1,
            label="mean: {:.1f}".format(analysis_data["price"].mean()))
plt.axvline(analysis_data["price"].median(), color='y', linestyle='dashed', linewidth=1,
            label="median: {:.1f}".format(analysis_data["price"].median()))
plt.axvline(analysis_data["price"].max(), color='r', linestyle='dashed', linewidth=1,
            label="max: {:.1f}".format(analysis_data["price"].max()))
plt.axvline(analysis_data["price"].min(), color='black', linestyle='dashed', linewidth=1,
            label="min: {:.1f}".format(analysis_data["price"].min()))

plt.text(analysis_data["price"].mean(), 2020, "mean", rotation=90, va="baseline")
plt.text(analysis_data["price"].median(), 2020, "median", rotation=90, va="baseline")
plt.text(analysis_data["price"].max(), 2020, "max", rotation=90, va="baseline")
plt.text(analysis_data["price"].min(), 2020, "min", rotation=90, va="baseline")

plt.legend()
plt.show()

# 2.1 Prices and manufacturers

# Product information: names of manufacturers and number of items by name
print(analysis_data["name"].unique())
value_counts_by_name = analysis_data.groupby("name")["price"].agg(["count"]).reset_index()
value_counts_by_name = value_counts_by_name[value_counts_by_name['count'] >= 30]\
    .sort_values('count', ascending=False).reset_index(drop=True)

# Convert to Markdown
value_counts_by_name_markdown = value_counts_by_name.to_markdown(index=False)
print(value_counts_by_name_markdown)

plt.figure(figsize=(10, 6))
gradient_colors = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
plt.bar(value_counts_by_name['name'], value_counts_by_name['count'],
        color=gradient_colors(value_counts_by_name['count'] / value_counts_by_name['count'].max()))

plt.grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(value_counts_by_name['count']):
    plt.text(i, v + 2.5, str(v), ha='center', va='center', color="b",
             bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.01'))

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Number of titles by Manufacturers")
plt.xlabel("")
plt.ylabel("")
plt.grid(axis='y', linestyle='--')
plt.ylim(10, value_counts_by_name['count'].max() * 1.1)
plt.subplots_adjust(bottom=0.15, left=0.1)

plt.show()

# Descriptive statistics for price by manufacturer name
price_stats_by_name = analysis_data.groupby("name")['price'].describe()

# Convert to Markdown
price_stats_by_name_markdown = price_stats_by_name.reset_index().to_markdown(index=False)
print(price_stats_by_name_markdown)

# Median price depending on manufacturer name
median_price_name = analysis_data.groupby("name")["price"].median().reset_index()
median_price_name = median_price_name.sort_values('price', ascending=False).reset_index(drop=True)
median_price_name = median_price_name.iloc[:21]

# Convert to Markdown
median_price_name_markdown = median_price_name.to_markdown(index=False)
print(median_price_name_markdown)

# Visualization of Median price by Manufacturers
fig, ax = plt.subplots(figsize=(10, 6))
cmap = mcolors.LinearSegmentedColormap.from_list("", ["#FF7F00", "#FFFF00", "#00FF00"])
sns.barplot(x="price", y="name", data=median_price_name,
            palette=cmap(median_price_name['price']/median_price_name['price'].max()),
            orient='horizontal')

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax)

ax.set_xlim(150, median_price_name['price'].max() * 1.1)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.title("Median price by Manufacturers")
plt.ylabel("")
plt.xlabel("")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Gradient in the background
gradient_cmap = mcolors.LinearSegmentedColormap.from_list("", ["#002FFF", "#FF0000"])
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
im = ax.imshow(gradient, cmap=gradient_cmap, aspect='auto', extent=[150, median_price_name['price'].max() * 1.1,
                                                                    -0.5, len(median_price_name)-0.5])

for i, v in enumerate(median_price_name['price']):
    ax.text(v + 180, i, str(round(v)), ha='center', va='center', color="black", fontsize=11,
            bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.05'))

plt.subplots_adjust(left=0.32, bottom=0.1)
plt.gca().invert_yaxis()
plt.show()

# Statistical analysis using OLS regression
formula = "price ~ name"
model = smf.ols(formula=formula, data=analysis_data).fit()

# Display the regression summary
print(model.summary())

''' Result of multiple linear regression model using conventional method Ordinary Least Squares (OLS).
    The dependent variable is “price” and the model includes 605 independent variables.

    An R-square value indicates that the model explains 51.5% of the variance in the dependent variable.
    The adjusted R-squared value is slightly lower, suggesting that some independent 
variables may not contribute much to the explanatory power of the model.
    The F-statistic of 4.023 and the associated p-value of 1.31e-127 suggest that the overall model is
statistically significant.
    The model includes 2900 observations and the log likelihood is -24029.
    AIC and BIC are measures of model fit that take into account the number of parameters in the model.
The lower the AIC and BIC values, the better the model fit. The AIC value is 4.927e+04 and the
BIC is equal to 5.289e+04.
    Finally, the "nonrobust" covariance type indicates that the standard errors of the coefficients are assumed
homoscedastic (i.e. constant at all levels of independent variables) and normal distributed.
    The omnibus test is a test of the normality of the residuals of a regression model. Value 1095.226
is the test statistic and the associated p-value is less than 0.05, as indicated by “Probability
(omnibus): 0.000." This suggests that the residuals are not normally distributed, which may be the reason
for concerns about model validity.
    The Durbin-Watson test is a test of autocorrelation of residuals. A value of 1.81 indicates that
there is some positive autocorrelation, although it is not strong enough to make model is invalid.
    The Jarque-Bera (JB) test is another test for the normality of residuals, and its statistical significance
the test value of 9370.045 and the associated p value of 0.00 indicate that the residuals are not distributed
Fine.
    A skewness value of 1.552 suggests that the distribution of residuals is positively skewed,
and the kurtosis value of 11.241 indicates that the distribution is heavy-tailed.
    The last line shows the condition number of the matrix of independent variables. High number
conditionality indicates that the model may have problems with multicollinearity, which may
make the coefficients difficult to interpret. The value of 1.33e+03 is relatively low, which indicates that
that multicollinearity may not be a significant problem in this model. '''

# 2.2 Prices and designers

# Let's prepare the data set for descriptive statistics on price depending on the designer's name.
# Weed out unrecognizable designer names
digit_mask = analysis_data['designer'].str.contains('\d')
digit_designer_data = analysis_data[~digit_mask].reset_index(drop=True)
print(digit_designer_data)

# Product information: designer names and number of items by designer name
print(digit_designer_data["designer"].unique())
value_counts_by_designer = digit_designer_data.groupby("designer")["price"].agg(["count"]).reset_index()
value_counts_by_designer = value_counts_by_designer[value_counts_by_designer['count'] >= 30]\
    .sort_values('count', ascending=False).reset_index(drop=True)
print(value_counts_by_designer)

plt.figure(figsize=(10, 6))
gradient_colors = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
plt.bar(value_counts_by_designer['designer'], value_counts_by_designer['count'],
        color=gradient_colors(value_counts_by_designer['count'] / value_counts_by_designer['count'].max()))

plt.grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(value_counts_by_designer['count']):
    plt.text(i, v + 20, str(v), ha='center', va='center', color="b",
             bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.01'))

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Number of titles by Designers")
plt.xlabel("")
plt.ylabel("")
plt.grid(axis='y', linestyle='--')
plt.ylim(10, value_counts_by_designer['count'].max() * 1.1)
plt.subplots_adjust(bottom=0.3, left=0.1)

plt.show()

# Descriptive statistics for price by designer name
price_stats_by_designer = digit_designer_data.groupby("designer")['price'].describe()

# Convert to Markdown
price_stats_by_designer_markdown = price_stats_by_designer.reset_index().to_markdown(index=False)
print(price_stats_by_designer_markdown)

# Median price depending on designer name
median_price_designer = digit_designer_data.groupby("designer")["price"].median().reset_index()
median_price_designer = median_price_designer.sort_values('price', ascending=False).reset_index(drop=True)
median_price_designer = median_price_designer.iloc[:16]

# Convert to Markdown
median_price_designer_markdown = median_price_designer.to_markdown(index=False)
print(median_price_designer_markdown)

# Visualization of Median price by Designers
fig, ax = plt.subplots(figsize=(10, 6))
cmap = mcolors.LinearSegmentedColormap.from_list("", ["#FF7F00", "#FFFF00", "#00FF00"])
bar_heights = np.arange(len(median_price_designer))
bar_widths = median_price_designer['price']
colors = cmap(median_price_designer['price']/median_price_designer['price'].max())

ax.barh(bar_heights, bar_widths, color=colors)

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax)

ax.set_xlim(150, median_price_designer['price'].max() * 1.12)
ax.set_yticks(bar_heights)
ax.set_yticklabels(median_price_designer['designer'])
plt.title("Median price by Designers")
plt.ylabel("")
plt.xlabel("")

plt.grid(axis='x', linestyle='--', alpha=0.5)

# Gradient in the background
gradient_cmap = mcolors.LinearSegmentedColormap.from_list("", ["#002FFF", "#FF0000"])
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
im = ax.imshow(gradient, cmap=gradient_cmap, aspect='auto', extent=[150, median_price_designer['price'].max() * 1.12,
                                                                    -0.5, len(median_price_designer)-0.5])

for i, v in enumerate(median_price_designer['price']):
    ax.text(v + 210, i, str(round(v)), ha='center', va='center', color="black", fontsize=11,
            bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.05'))

plt.subplots_adjust(left=0.4, bottom=0.1)
plt.gca().invert_yaxis()
plt.show()

# Statistical analysis using OLS regression
formula = "price ~ designer"
model = smf.ols(formula=formula, data=digit_designer_data).fit()
print(model.summary())

''' Result of multiple linear regression model using conventional method Ordinary Least Squares (OLS).
    The model includes 196 predictor variables and one dependent variable (price).

    The R-squared value of the model is 0.448, which means that approximately 44.8% of the variance in the dependent
variable can be explained by the predictor variables in the model.
    The adjusted R-squared value is 0.407, which takes into account the number of predictor variables
in the model and adjusts the R-squared value accordingly.
    The F-statistic is 10.79 with a corresponding probability value (Prob) of 7.46e-219.
    This indicates that there is strong evidence that at least one of the predictor variables
significantly associated with the dependent variable.
    The log likelihood value of the model is -23395, which is used to compare models
and assessment of their compliance.
    Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) also help
compare the current model with others.
    It is important to note that the covariance type is unstable, meaning that the model is sensitive to outliers.
    Omnibus Test - A value of 1331.561 indicates that the residuals are not normally distributed,
with a corresponding probability value (Prob(Omnibus)) of 0.000. This violates one of the assumptions
linear regression models and may affect the reliability of the results.
    The Durbin-Watson statistic is a test for the presence of autocorrelation in the residuals. Value 1.817
indicates that there may be some positive autocorrelation in the residuals, but not that much
serious enough to invalidate the results.
    The Jarque-Bera (JB) test is another test of normality, but it takes into account both skewness and kurtosis.
A value of 10710.006 and a probability (Prob(JB)) of 0.00 indicate that the residuals are not normally distributed,
which confirms the results of the Omnibus test.
    A skewness value of 2.092 indicates that the distribution of residuals has a positive
skewness, and the kurtosis value of 11.623 indicates that the distribution has heavy tails
and is more peaked than the normal distribution.
    Condition number (Cond. No.) 146 is a measure of multicollinearity in the model. Higher value
indicates a high degree of multicollinearity, which may make it difficult to interpret individual
coefficients However, a value of 146 suggests that multicollinearity is not severe
problem in this model. '''

# 2.3 Prices and product categories

# Product information: product categories and number of items by category
print(analysis_data["category"].unique())
print(analysis_data["category"].value_counts())

# Or
value_counts_by_category = analysis_data.groupby("category")['price'].agg(["count"]).reset_index()
value_counts_by_category = value_counts_by_category.sort_values('count', ascending=False).reset_index(drop=True)
print(value_counts_by_category)

plt.figure(figsize=(10, 6))
gradient_colors = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
plt.bar(value_counts_by_category['category'], value_counts_by_category['count'],
        color=gradient_colors(value_counts_by_category['count'] / value_counts_by_category['count'].max()))

plt.grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(value_counts_by_category['count']):
    plt.text(i, v + 15, str(v), ha='center', va='center', color="b",
             bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.01'))

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Number of titles by Categories")
plt.xlabel("")
plt.ylabel("")
plt.grid(axis='y', linestyle='--')
plt.ylim(0, value_counts_by_category['count'].max() * 1.1)
plt.subplots_adjust(bottom=0.32, left=0.1)

plt.show()

# Descriptive statistics for price by category
price_stats_by_category = analysis_data.groupby("category")['price'].describe()

# Convert to Markdown
price_stats_by_category_markdown = price_stats_by_category.reset_index().to_markdown(index=False)
print(price_stats_by_category_markdown)

# Or (let's highlight the most significant values for us)
price_stats = analysis_data.groupby("category")["price"].agg(["max", "mean", "median", "min"]).reset_index()
price_stats = price_stats.sort_values('max', ascending=False).reset_index(drop=True)

# Convert to Markdown
price_stats_markdown = price_stats.to_markdown(index=False)
print(price_stats_markdown)

# Visualization of Prices by Categories
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(price_stats['category'], price_stats["max"], label="Max")
ax.bar(price_stats['category'], price_stats["mean"], label="Mean", alpha=0.7)
ax.bar(price_stats['category'], price_stats["median"], label="Median")
ax.bar(price_stats['category'], price_stats["min"], label="Min", alpha=0.7)
plt.ylim(0, price_stats["max"].max() * 1.1)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title("Prices by Categories")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.grid(axis='y', linestyle='--')
for i, v in enumerate(price_stats["max"]):
    plt.text(i, v+100, str(round(v)), ha='center', va='center', color="b",
             bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.1'))
plt.subplots_adjust(bottom=0.32, left=0.1)

plt.show()

# Median price depending on category
median_price = analysis_data.groupby("category")["price"].median().reset_index()
median_sort_price = median_price.sort_values('price', ascending=False).reset_index(drop=True)

# Convert to Markdown
median_sort_price_markdown = median_sort_price.to_markdown(index=False)
print(median_sort_price_markdown)

# Visualization of Median prices by Categories
fig, ax = plt.subplots(figsize=(10, 6))
gradient_colors = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

ax.bar(median_sort_price['category'], median_sort_price['price'],
       color=gradient_colors(median_sort_price['price']/median_sort_price['price'].max()),
       width=0.5, edgecolor='black', linewidth=1, alpha=0.7, align='center',
       capsize=7, ecolor='gray', hatch='///',
       error_kw=dict(elinewidth=1, capsize=3, capthick=1))

ax.set_xticks(range(len(median_sort_price['category'])))
ax.set_xticklabels(median_sort_price['category'], rotation=45, ha='right', fontsize=10)
ax.set_title("Median prices by Categories")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(150, median_sort_price['price'].max() * 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for i, v in enumerate(median_sort_price['price']):
    ax.text(i, v+60, str(round(v)), ha='center', va='center', color="b",
            bbox=dict(facecolor='white', edgecolor='w', boxstyle='round,pad=0.1'))

plt.subplots_adjust(bottom=0.32, left=0.1)

plt.show()

# Statistical analysis using OLS regression
formula = "price ~ C(category)"
model = smf.ols(formula=formula, data=analysis_data).fit()
print(model.summary())

''' Result of multiple linear regression model using conventional method
Ordinary Least Squares (OLS).
    The model includes 16 predictor variables and one dependent variable (price).

    The R-squared value of the model is 0.159, which means that approximately 15.9% of the variance in the dependent
variable can be explained by the predictor variables in the model.
    The adjusted R-squared value is 0.154, which takes into account the number
predictor variables in the model and adjusts the R-squared value accordingly.
    The F-statistic is 34.05 with a corresponding probability value (Prob) of 3.08e-96. This indicates
there is strong evidence that at least one of the predictor variables is significantly
associated with the dependent variable.
    The log likelihood value of the model is -24827, which is used for comparison
models and assessment of their compliance. Akaike Information Criterion (AIC) and Bayesian Information Criterion
criterion (BIC) also help to compare the current model with others.
    The covariance type is unreliable, which means the model is sensitive to outliers and influential observations.
Overall, the model has a low R-squared value, indicating that the predictor variables are not
are strong predictors of the dependent variable. The F-statistic is significant, indicating
that there is a significant relationship between the predictor variables and the dependent variable,
but the effect size is relatively small.
    The omnibus test is a test of the normality assumption of residuals in a linear regression model.
A value of 1299.027 indicates that the residuals are not normally distributed, with a corresponding value
probability (Prob(Omnibus)) equal to 0.000. This violates one of the assumptions of the linear regression model and
may affect the reliability of the results.
    The Durbin-Watson statistic is a test for the presence of autocorrelation in the residuals. Value 1.708
indicates that there may be some positive autocorrelation in the residuals, but not that much
serious enough to invalidate the results.
    The Jarque-Bera (JB) test is another test of normality, but it takes into account both skewness and kurtosis.
A value of 7022.093 and a probability (Prob(JB)) of 0.00 indicate that the residuals are not normally distributed,
which confirms the results of the omnibus test.
    A skewness value of 2.101 indicates that the distribution of residuals is positively skewed,
and a kurtosis value of 9.36 indicates that the distribution is more peaked than normal
distribution.
    The condition number (Cond. No.) of 35.5 is a measure of multicollinearity in the model.
A higher value indicates a greater degree of multicollinearity, which may make it difficult to
interpretation of individual coefficients. However, a value of 35.5 suggests that multicollinearity
is not a major problem in this model. '''

# 2.4 Prices and sizes

# Let's calculate the correlation between price and volume of goods
analysis_data['size'] = analysis_data['width'] * analysis_data["depth"] * analysis_data["height"]
correlation = analysis_data["price"].corr(analysis_data['size'])
print(correlation)

''' In this case, the Pearson correlation coefficient between “price” and “size” is 
indicating a strong positive correlation between the two variables. It means, 
that as the size of an object increases, the price also tends to increase, 
and vice versa. This indicates that the probability observing a correlation as strong 
as 0.8233616402808847 is only due to random chance actually equal to zero. 
Therefore, we can conclude that there is a significant correlation between the variables 
"price" and "size" in the data set.
    Overall, the result indicates a strong positive correlation between “price” and 
“size”, which means that as the size of the property increases, the price also tends 
to increase. '''

# Calculate 'size' and find size outliers
z_scores = zscore(analysis_data['size'])
outliers = (z_scores > 3)
size_outliers_data = analysis_data[outliers]

# Convert to Markdown
size_outliers_data_markdown = size_outliers_data.to_markdown(index=False)
print(size_outliers_data_markdown)

# Remove size outliers
size_outliers_index = analysis_data.index[outliers]
data_without_size_outliers = analysis_data[~analysis_data.index.isin(size_outliers_index)].copy()

# Convert to Markdown
data_without_size_outliers_markdown = data_without_size_outliers.to_markdown(index=False)
print(data_without_size_outliers_markdown)

'''
# The other variant
numeric_columns = ['price', 'depth', 'height', 'width']

# Remove outliers based on z-score
z_scores = zscore(data_with_size_outliers[numeric_columns])
abs_z_scores = np.abs(z_scores)
outlier_indices = np.where(abs_z_scores > 3)[0]
outliers = data_with_size_outliers.iloc[outlier_indices]
print(outliers)

# Remove size outliers
size_outliers_index = outliers.index
data_without_size_outliers = data_with_size_outliers[~data_with_size_outliers.index.isin(size_outliers_index)].copy()
print(data_without_size_outliers)
'''

# Scatterplot with regression line
sns.regplot(x=data_without_size_outliers['size'], y=data_without_size_outliers['price'], line_kws={"color": "red"})
plt.title('Prices by Sizes')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

data_without_size_outliers = data_without_size_outliers[["price", "depth", "height", "width"]]
data_without_size_outliers = data_without_size_outliers.dropna(subset=["depth", "height", "width"])
print(data_without_size_outliers)

# Let's calculate the pairwise correlation between numerical columns
correlation_matrix = data_without_size_outliers.corr()

# Convert to Markdown
correlation_matrix_markdown = correlation_matrix.to_markdown()
print(correlation_matrix_markdown)

# Or
correlation_width_price = pearsonr(data_without_size_outliers["price"], data_without_size_outliers["width"])
correlation_depth_price = pearsonr(data_without_size_outliers["price"], data_without_size_outliers["depth"])
correlation_height_price = pearsonr(data_without_size_outliers["price"], data_without_size_outliers["height"])

# Convert to Markdown
correlation_width_price_markdown = correlation_width_price[0]
correlation_depth_price_markdown = correlation_depth_price[0]
correlation_height_price_markdown = correlation_height_price[0]
print(f"Correlation between Price and Width: {correlation_width_price_markdown}")
print(f"Correlation between Price and Depth: {correlation_depth_price_markdown}")
print(f"Correlation between Price and Height: {correlation_height_price_markdown}")

''' There is a strong positive correlation between “price” and “width” (correlation 
coefficient 0.77), which suggests that as the width of the object increases, the price 
also tends to increase.
    There is a moderate positive correlation between “price” and “depth” (correlation 
coefficient 0.64), which suggests that as the depth of the object increases, the price 
tends to increase.
    There is a moderate positive correlation between “price” and “height” (correlation 
coefficient 0.23), which indicates a slight upward trend in price as the height of 
the property increases. 
    So, "width" and "depth" are important factors in determining the price of an object, 
while “height” has a weaker relationship with price.
    Overall, these results indicate that there is a strong positive linear relationship 
between price and depth and height, as well as a weaker, but still statistically 
significant positive linear relationship between price and width. '''

# Heatmaps
correlation_matrix.index = correlation_matrix.index.map(str)
correlation_matrix.columns = correlation_matrix.columns.map(str)

fig, ax = plt.subplots()
im = ax.imshow(correlation_matrix, cmap='coolwarm')

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(['price', 'depth', 'height', 'width'])))
ax.set_yticks(np.arange(len(['price', 'depth', 'height', 'width'])))
ax.set_xticklabels(['price', 'depth', 'height', 'width'], rotation=45, ha="right")
ax.set_yticklabels(['price', 'depth', 'height', 'width'])

for i in range(len(['price', 'depth', 'height', 'width'])):
    for j in range(len(['price', 'depth', 'height', 'width'])):
        text = ax.text(j, i, '{:.2f}'.format(correlation_matrix.iloc[i, j]), ha="center", va="center",
                       color="w")
plt.title("Correlation matrix between Price and Sizes")
plt.show()

# 3 Hypotheses
''' Based on the EDA and your common sense, choose two hypotheses you
want to test/analyse. For each hypothesis list the null hypothesis and other possible
alternative hypotheses, design tests to distinguish between them, and complete them.
Describe the results. '''

''' Hypothesis 1. The price of furniture created by popular designers is higher than that of furniture created by
designed by less known designers.
    We can define popular designers as those whose names appear in more than 50% of the set
data, and less known designers as those whose names appear in less than 50% of the data set.
    Null hypothesis: there is no significant difference in the price of furniture between popular and less known
designers do not.
    Alternative hypothesis 1: furniture designed by popular designers is more expensive than furniture
designed by less known designers.
    Alternative Hypothesis 2: Furniture designed by less known designers is more expensive than furniture 
designed by popular designers. '''

digit_mask = analysis_data['designer'].str.contains('\d')
designers_data = analysis_data[~digit_mask].reset_index(drop=True)

designers_counts = designers_data['designer'].value_counts()[::-1]

# Calculate the sum of counts
sum_of_counts = designers_counts.sum()

# Calculate the cumulative sum of percentages
cumulative_percentage = designers_counts.cumsum() / sum_of_counts

# Find the designers that contribute to 50% of the total counts
famous_designers = designers_counts[cumulative_percentage > 0.5][::-1]

print(f'Famous Designers contributing to 50% of total counts:')
print(famous_designers)

famous_designers_data = designers_data[designers_data['designer'].isin(famous_designers.index)]
less_known_designers_data = designers_data[~designers_data['designer'].isin(famous_designers.index)]

#1) t-test to compare the average prices of two groups
t_statistic, p_value = ttest_ind(famous_designers_data['price'], less_known_designers_data['price'],
                                 equal_var=False)
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

famous_designers_mean = np.mean(famous_designers_data['price'])
less_known_designers_mean = np.mean(less_known_designers_data['price'])
difference_means = famous_designers_mean - less_known_designers_mean
if difference_means > 0:
    print(''' Alternative hypothesis 1 is confirmed: furniture designed by popular designers, "
        more expensive than furniture designed by less known designers. ''')
else:
    print(''' Alternative hypothesis 2 is confirmed: furniture designed by less known designers is more expensive
        furniture designed by popular designers. ''')

''' Alternative Hypothesis 2 is true for this hypothesis.
    A p value of 0.000 indicates that the probability of observing such a large difference by chance is
in the mean values between the two groups is very small. In other words, the null hypothesis stating
that there is no difference between the average prices of the products developed by the two groups can be rejected
with a high degree of reliability. '''

# 2) Mann-Whitney-Wilcoxon test
statistic, p_value = mannwhitneyu(famous_designers_data['price'], less_known_designers_data['price'])

print(f"Statistic: {statistic:.2f}")
print(f"P-value: {p_value:.3f}")

''' If the Mann-Whitney U test yielded a p-value of 0.000, this means that the test detected
a significant difference in the average price of furniture between well-known and less known designers.
Since the p-value is less than the typical significance level of 0.05, we can reject the null hypothesis
and conclude that there is a statistically significant difference in the median price of furniture between 
the two groups.
    Thus, the Mann-Whitney U test and the difference in average price values (see the first test) support
alternative hypothesis 2. '''

# 3) Bootstrap Test
n_bootstraps = 10000
diffs = np.empty(n_bootstraps)
for i in range(n_bootstraps):
    famous_samples = np.random.choice(famous_designers_data['price'], size=len(famous_designers_data), replace=True)
    less_known_samples = np.random.choice(less_known_designers_data['price'], size=len(less_known_designers_data),
                                          replace=True)
    diffs[i] = np.mean(famous_samples) - np.mean(less_known_samples)

p_value = np.sum(diffs >= 0) / n_bootstraps
print(f"P-value: {p_value:.3f}")

''' So, Bootstrap Test and the difference in average price values (see the first test) support
alternative hypothesis 2. '''

#4) Shuffle test
famous_prices_values = famous_designers_data['price'].values
less_known_prices_values = less_known_designers_data['price'].values

observed_diff = np.mean(famous_prices_values) - np.mean(less_known_prices_values)

concatenated_prices = np.concatenate([famous_prices_values, less_known_prices_values])
np.random.shuffle(concatenated_prices)

shuffled_famous_prices = concatenated_prices[:len(famous_prices_values)]
shuffled_less_known_prices = concatenated_prices[len(famous_prices_values):]

shuffled_diff = np.mean(shuffled_famous_prices) - np.mean(shuffled_less_known_prices)

n_permutations = 1000
null_distribution = np.zeros(n_permutations)
for i in range(n_permutations):
    np.random.shuffle(concatenated_prices)
    shuffled_famous_prices = concatenated_prices[:len(famous_prices_values)]
    shuffled_less_known_prices = concatenated_prices[len(famous_prices_values):]
    null_distribution[i] = np.mean(shuffled_famous_prices) - np.mean(shuffled_less_known_prices)

p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))

print("Observed difference in mean values: {:.2f}".format(observed_diff))
print("P-value: {:.4f}".format(p_value))

''' Based on the observed mean difference of -243.64 and p-value of 0.0000, we can
reject the null hypothesis and conclude that there is a significant difference in the 
price of furniture between popular and less known designers. Moreover, alternative 
hypothesis 2 is supported, according to which furniture created by less known designers 
is more expensive than furniture, designed by popular designers. '''

# 5) Tukey's honestly significant difference (HSD) test
''' The Tukey HSD test is a commonly used multiple comparison test that compares
all possible pairs of group means and calculates a confidence interval for the 
difference in means. If the confidence interval does not include zero, it suggests 
a significant difference between the two in groups. '''

famous_prices = famous_designers_data['price']
less_known_prices = less_known_designers_data['price']

tukey_results = pairwise_tukeyhsd(np.concatenate([famous_prices, less_known_prices]),
                                  np.concatenate([['famous'] * len(famous_prices),
                                                  ['less-known'] * len(less_known_prices)]))

print(tukey_results.summary())

''' The result shows that the average difference between the "famous" group and the "less famous" group
is 243.6383 with a p-value of 0.0. This means that the average price of products developed by well-known
designers, significantly lower than the average price of products designed by less known designers.
    The lower and upper limits of the confidence interval are 140.5739 and 346.7026, respectively.
Since the confidence interval does not contain zero, we can reject the null hypothesis and conclude,
that the difference in average prices between the two groups is statistically significant at the 0.05 
significance level. '''

# 6) ANOVA test - analysis of variance
''' This is a parametric test that can be used to compare averages over
than two groups. In our case, we can use it to compare average prices
for furniture designed by popular designers and lesser-known designers. '''

_, p_value = stats.f_oneway(famous_prices, less_known_prices)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "between popular and less known designers.")
else:
    print("The null hypothesis could not be rejected. Significant difference in the average price of furniture "
          "there is no difference between popular and less known designers")

''' So, the ANOVA test and the difference in average prices (see the first test) support
alternative hypothesis 2. '''

# 7 Wilcoxon signed-rank test
min_length = min(len(famous_prices_values), len(less_known_prices_values))
famous_prices_values = famous_prices_values[:min_length]
less_known_prices_values = less_known_prices_values[:min_length]

statistic, p_value = wilcoxon(famous_prices_values, less_known_prices_values)

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

''' The Wilcoxon signed rank test was conducted for two samples of furniture prices (furniture from famous
designers and furniture from less known designers) to test whether there is a significant difference
in average prices between the two groups. The test resulted in a test statistic of 353068.00 and a p-value of 0.0000.
    The test statistic is the sum of the ranks assigned to the differences between each pair
observations in two samples. The larger the test statistic, the stronger the evidence against the null.
hypothesis which states that there is no difference between the means of two groups.
    The p-value is the probability of observing a test statistic as extreme as
obtained under the null hypothesis. In this case, the p-value is very small (less than 0.001), which means
that the probability of observing such a large test statistic by chance is very small. Thus,
we reject the null hypothesis and conclude that there is a significant difference in average prices between the two
groups of furniture.
    So, the Wilcoxon signed rank test and the difference in mean prices (see first test) support
alternative hypothesis 2. '''

# 8) Wilcoxon rank-sum test
statistic, p_value = ranksums(famous_prices_values, less_known_prices_values)

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

''' Unlike the previous test, the test statistics are based on the ranks of values in two samples, and
not on the differences between values.
    The results of the Wilcoxon rank sum test show that there is a significant difference
between the prices of furniture from popular and less known designers. The test statistics is -8.48,
and the p-value is 0.0000, which indicates that the probability of such a random observation
There is very little difference in the averages. Therefore, we can reject the null hypothesis and
conclude that prices for furniture designed by popular designers differ significantly
from the prices of furniture designed by less known designers.
    So, the Wilcoxon rank sum test and the difference in mean prices (see first test) support
alternative hypothesis 2. '''

# 9) Kruskal-Wallis test
statistic, p_value = kruskal(famous_designers_data['price'], less_known_designers_data['price'])

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

''' The test statistic was calculated as 40.56 and the p-value was found to be 0.0000.
This indicates that there is good reason to reject the null hypothesis.
    So, the Kruskal-Wallis test and the difference in average price values (see the first test) support
alternative hypothesis 2. '''

''' Overall conclusion: tests support alternative hypothesis 2: furniture designed by less known
by designers, more expensive than furniture designed by popular designers. '''

''' Hypothesis 2. Seating Furniture is on average more expensive than storage furniture
(Storage Furniture), i.e. there is a significant difference in average price depending on function
(use) furniture.
    Seating furniture: Sofas & armchairs, Beds, Chairs, Children's furniture, Nursery furniture,
Café furniture, Bar furniture, Outdoor furniture.
    Storage furniture: Tables & desks, Trolleys, Wardrobes, Bookcases & shelving units,
TV & media furniture, Cabinets & cupboards, Room dividers, Sideboards, buffets & console tables,
Chests of drawers & drawer units.
    Null hypothesis: The average price of seating and storage furniture is the same.
    Alternative hypothesis 1: The average price of seating furniture is higher than storage furniture.
    Alternative Hypothesis 2: The average price of storage furniture is higher than that of seating furniture. '''

category_groups = {
    'Seating Furniture': ['Sofas & armchairs', 'Beds', 'Chairs', 'Children\'s furniture', 'Nursery furniture',
                                   'Café furniture', 'Bar furniture', 'Outdoor furniture'],
    'Storage Furniture': ['Tables & desks','Trolleys', 'Wardrobes', 'Bookcases & shelving units',
                          'TV & media furniture', 'Cabinets & cupboards', 'Room dividers',
                          'Sideboards, buffets & console tables', 'Chests of drawers & drawer units']
}

grouped_data = analysis_data.groupby('category')

seating_furniture = grouped_data.get_group('Sofas & armchairs')
for category in category_groups['Seating Furniture'][1:]:
    seating_furniture = pd.concat([seating_furniture, grouped_data.get_group(category)])

storage_furniture = grouped_data.get_group('Tables & desks')
for category in category_groups['Storage Furniture'][1:]:
    storage_furniture = pd.concat([storage_furniture, grouped_data.get_group(category)])

print('Seating Furniture:')
print(seating_furniture)

print('\nStorage Furniture:')
print(storage_furniture)

#1) t-test to compare the average prices of two groups
t_statistic, p_value = ttest_ind(seating_furniture['price'], storage_furniture['price'],
                                 equal_var=False)
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

seating_furniture_mean = np.mean(seating_furniture['price'])
storage_furniture_mean = np.mean(storage_furniture['price'])
difference_means = seating_furniture_mean - storage_furniture_mean
if p_value < 0.05 and difference_means > 0:
    print(''' Alternative hypothesis 1 is confirmed: the average price of seating furniture is higher,
            than for storage. ''')
elif p_value < 0.05 and difference_means <0:
    print(''' Alternative hypothesis 2 is confirmed: the average price of furniture for storage is higher,
            than for sitting. ''')
else:
    print(''' The null hypothesis is confirmed: The average price of seating and storage 
furniture is the same. ''')

''' Since the p-value is very small (p-value > 0.05), we can accept the null hypothesis,
which states that the average price of seating and storage furniture is the same. Hence,
we can't conclude that there is a significant difference in the average price depending on 
the function (use) of the furniture. Therefore, the null hypothesis is true and we 
can't say that the seating furniture on average, it costs different than storage furniture. '''

# 2) Mann-Whitney-Wilcoxon test
statistic, p_value = mannwhitneyu(seating_furniture['price'], storage_furniture['price'])

print(f"Statistic: {statistic:.2f}")
print(f"P-value: {p_value:.3f}")

''' According to the results of the Mann-Whitney U test, the p-value is more than 0.05, which means that 
the null hypothesis is confirmed by the Mann-Whitney U-test. Therefore, we can accept the null hypothesis 
that the average price of seating and storage furniture is the same. '''

# 3) Bootstrap Test
def diff_means(x, y):
    return np.mean(x) - np.mean(y)

n_iterations = 1000
differences = []
for i in range(n_iterations):
    sample_a = resample(seating_furniture['price'], replace=True, n_samples=len(seating_furniture))
    sample_b = resample(storage_furniture['price'], replace=True, n_samples=len(storage_furniture))
    difference = diff_means(sample_a, sample_b)
    differences.append(difference)

alpha = 0.05
lower = np.percentile(differences, alpha/2*100)
upper = np.percentile(differences, (1-alpha/2)*100)

if lower <= 0 <= upper:
    print("There is no fundamental difference in average prices.")
else:
    print("There is a significant difference in average prices.")

''' So, Bootstrap Test and the difference in average price values (see the first test) support
the null hypothesis. '''

# 4) Shuffle test
observed_diff = np.mean(seating_furniture['price']) - np.mean(storage_furniture['price'])

n_iterations = 1000
differences = []
combined_data = pd.concat([seating_furniture, storage_furniture])

for i in range(n_iterations):
    np.random.shuffle(combined_data['price'].values)
    sample_a = combined_data.iloc[:len(seating_furniture)]
    sample_b = combined_data.iloc[len(seating_furniture):]
    difference = np.mean(sample_a['price']) - np.mean(sample_b['price'])
    differences.append(difference)

p_value = (np.abs(differences) >= np.abs(observed_diff)).mean()

print("Observed difference in means: {:.2f}".format(observed_diff))
print("P-value: {:.4f}".format(p_value))

if p_value < 0.05:
    print("There is a significant difference in average prices between seating furniture "
          "and furniture for storage, (p = {:.3f}).".format(p_value))
else:
    print("There is no significant difference in average prices between seating and furniture "
          "for storage, (p = {:.3f}).".format(p_value))

''' The results of the permutation test show that there is a significant difference between the average 
prices of the groups “Furniture for seating” and “Furniture for storage”. P value is more than 0.05, which
supports the null hypothesis of no difference. '''

# 5) Tukey's honestly significant difference (HSD) test
results = pairwise_tukeyhsd(
np.concatenate([seating_furniture['price'], storage_furniture['price']]),
np.concatenate([['Seating Furniture']*len(seating_furniture), ['Storage Furniture']*len(storage_furniture)]),
alpha=0.05)

print(results)

''' The Tukey HSD test results show that there is a significant difference between the means
values of the groups “Furniture for seating” and “Furniture for storage”. P value is more than 0.05, which
supports the null hypothesis of no difference. '''

# 6) ANOVA test - analysis of variance
_, p_value = stats.f_oneway(seating_furniture['price'], storage_furniture['price'])

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")
else:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same")

''' So, the ANOVA test supports the null hypothesis. '''

# 7) Wilcoxon signed-rank test
seating_values = seating_furniture['price'].values
storage_values = storage_furniture['price'].values
min_length = min(len(seating_values), len(storage_values))
seating_values = seating_values[:min_length]
storage_values = storage_values[:min_length]

statistic, p_value = wilcoxon(seating_values, storage_values)

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

if p_value > 0.05:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

''' So, the Wilcoxon signed-rank test supports the null hypothesis. '''

# 8) Wilcoxon rank-sum test
statistic, p_value = ranksums(seating_values, storage_values)

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

if p_value > 0.05:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

''' So, the Wilcoxon rank-sum test supports the null hypothesis. '''

# 9) Kruskal-Wallis test
statistic, p_value = kruskal(seating_furniture['price'], storage_furniture['price'])

print("Test statistic: {:.2f}".format(statistic))
print("P-value: {:.4f}".format(p_value))

if p_value > 0.05:
    print("The null hypothesis could not be rejected. The average price of seating and storage furniture is the same.")
else:
    print("Reject the null hypothesis. There is a significant difference in the average price of furniture "
          "for seating and storage.")

''' So, the Kruskal-Wallis test supports the null hypothesis. '''

# 10) OLS regression model
# Concatenate prices and create labels for categories
X = pd.concat([seating_furniture['price'], storage_furniture['price']])
y = np.concatenate([np.ones(len(seating_furniture)), np.zeros(len(storage_furniture))])

# Add a constant term to the independent variable
X = add_constant(X)

# Fit the OLS model
model = stsm.OLS(y, X)
results = model.fit()

print(results.summary())

if results.pvalues[1] > 0.05:
    print("There is no fundamental difference in average prices.")
else:
    print("Reject the null hypothesis. There is a significant difference in average prices.")

''' The OLS regression model shows that there is a significant difference in the average price between
seating furniture and storage furniture with a p-value of 0.00. Coefficient of the variable “price”
positive, indicating that seating furniture is generally more expensive than outdoor furniture
storage However, the R-squared value is low at 0.022, indicating that the model
does not explain most of the variance in the data. The model intercept test with a p-value
supports the null hypothesis. '''

''' Overall conclusion: tests confirm the null hypothesis: 
The average price of seating and storage furniture is the same. '''

# 4 Machine Learning
''' Train a model to predict the price of furniture.
- Indicate which columns should not be included in the model and why.
- Create a cross-validation pipeline for training and evaluation
models, including (if necessary) steps such as imputation
missing values and normalization.
- Suggest methods to improve model performance.
Describe the results. '''

''' Before training the model, we must exclude unnecessary columns such as 'index', 'item_id',
since they do not affect the price forecast.
    Additionally, we should exclude the "old_price" column since it is highly correlated with
"price" column, and using both functions may cause multicollinearity problems. '''

exclude_cols = ['item_id', 'old_price', 'price', 'name', 'link',
                'short_description', 'sellable_online']
if 'index' in analysis_data.columns:
    analysis_data = analysis_data.drop('index', axis=1)

median_d = analysis_data.groupby(['category'])['depth'].median()
median_h = analysis_data.groupby(['category'])['height'].median()
median_w = analysis_data.groupby(['category'])['width'].median()

median_price = analysis_data.groupby(['category'])['price'].median()
median_dsgn = analysis_data.groupby(['designer'])['price'].median()

analysis_data = analysis_data.set_index(['category'])
analysis_data['depth_1'] = analysis_data['depth'].fillna(median_d)
analysis_data['height_1'] = analysis_data['height'].fillna(median_h)
analysis_data['width_1'] = analysis_data['width'].fillna(median_w)
analysis_data['category'] = median_price
analysis_data = analysis_data.set_index(['designer'])
analysis_data['designer'] = median_dsgn

num_cols = ['depth_1', 'height_1', 'width_1', 'category', 'designer']
cat_cols = ['other_colors']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

X = analysis_data.drop(exclude_cols, axis=1)
y = analysis_data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 1) KNeighborsRegressor
model_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn_regressor', KNeighborsRegressor())
])

model_knn.fit(X_train, y_train)

y_knn_pred = model_knn.predict(X_test)

''' Let's evaluate the performance of the model:
    RMSE is the square root of MSE (mean square error) and is more interpretable
value because it is in the same units as the forecast and actual values. '''

rmse_knn = np.sqrt(np.mean((y_knn_pred - y_test)**2))
print('KNeighborsRegressor RMSE: {:.2f}'.format(rmse_knn))

mse_knn = np.mean((y_knn_pred - y_test)**2)
print('KNeighborsRegressor Mean Squared Error: {:.2f}'.format(mse_knn))

# 2) LinearRegression
model_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_regression.fit(X_train, y_train)

y_regression_pred = model_regression.predict(X_test)

rmse_regression = np.sqrt(np.mean((y_regression_pred - y_test)**2))
print('LinearRegression RMSE: {:.2f}'.format(rmse_regression))

mse_regression = np.mean((y_regression_pred - y_test)**2)
print('LinearRegression Mean Squared Error: {:.2f}'.format(mse_regression))

""" The linear regression model performs slightly better than the K-nearest neighbors 
algorithm with in terms of RMSE and root mean square error, indicating that on average it gives
more accurate forecasts. """

# 3) DecisionTreeRegressor
model_tree_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('tree_regressor', DecisionTreeRegressor(random_state=42))
])

model_tree_regression.fit(X_train, y_train)

y_tree_regression_pred = model_tree_regression.predict(X_test)

rmse_tree_regression = np.sqrt(np.mean((y_tree_regression_pred - y_test)**2))
print('DecisionTreeRegressor RMSE: {:.2f}'.format(rmse_tree_regression))

mse_tree_regression = np.mean((y_tree_regression_pred - y_test)**2)
print('DecisionTreeRegressor Mean Squared Error: {:.2f}'.format(mse_tree_regression))

# Let's configure the DecisionTreeRegressor hyperparameters:
param_grid = {
    'tree_regressor__max_depth': [20, 24, 28, 32, 36, 40, 44],
    'tree_regressor__min_samples_split': [2, 4, 6, 8, 10, 12, 14],
    'tree_regressor__min_samples_leaf': [2, 4, 6, 8, 10, 12, 14]
}

grid_search = GridSearchCV(
    estimator=model_tree_regression,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print('DecisionTreeRegressor Best examples hyperparameters:', grid_search.best_params_)
print('DecisionTreeRegressor Best examples score:', grid_search.best_score_)

# DecisionTreeRegressor with customized hyperparameters
best_params = {
    'max_depth': 20,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'splitter': 'best'
}

model_best_tree_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('best_tree_regressor', DecisionTreeRegressor(random_state=42, **best_params))
])

model_best_tree_regression.fit(X_train, y_train)

y_best_tree_regression_pred = model_best_tree_regression.predict(X_test)

rmse_best_tree_regression = np.sqrt(np.mean((y_best_tree_regression_pred - y_test)**2))
print('DecisionTreeRegressor Best Params RMSE: {:.2f}'.format(rmse_best_tree_regression))

mse_best_tree_regression = np.mean((y_best_tree_regression_pred - y_test)**2)
print('DecisionTreeRegressor Best Params Mean Squared Error: {:.2f}'.format(mse_best_tree_regression))

''' The results show that the decision tree regressor has higher MSE and RMSE than
linear regression, which indicates that the decision tree model has a lower
performance in predicting the target variable (price) compared to linear regression.
This is probably due to the fact that decision trees can overfit the data and become
too complex, resulting in poor performance on new, unseen data.
On the other hand, linear regression is a simpler and more interpretable model that
can work well when there is a linear relationship between the input features and the 
target variable. '''

# 4) ElasticNet
model_elasticnet = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('elasticnet_regressor', ElasticNet())
])

model_elasticnet.fit(X_train, y_train)

y_elasticnet_pred = model_elasticnet.predict(X_test)

rmse_elasticnet = np.sqrt(np.mean((y_elasticnet_pred - y_test)**2))
print('ElasticNet RMSE: {:.2f}'.format(rmse_elasticnet))

mse_elasticnet = np.mean((y_elasticnet_pred - y_test)**2)
print('ElasticNet Mean Squared Error: {:.2f}'.format(mse_elasticnet))

# Let's configure ElasticNet hyperparameters:
param_grid = {
    'elasticnet_regressor__alpha': [0.1, 0.5, 1, 5, 10],
    'elasticnet_regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
}

grid_search = GridSearchCV(
    estimator=model_elasticnet,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print('ElasticNet Best examples hyperparameters:', grid_search.best_params_)
print('ElasticNet Best examples score:', grid_search.best_score_)

# ElasticNet with tuned hyperparameters
best_params = {
    'elasticnet_regressor__alpha': 0.1,
    'elasticnet_regressor__l1_ratio': 0.9
}

model_best_elasticnet = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('best_elasticnet_regressor', ElasticNet())
])

model_best_elasticnet.fit(X_train, y_train)

y_best_elasticnet_pred = model_best_elasticnet.predict(X_test)

rmse_best_elasticnet = np.sqrt(np.mean((y_best_elasticnet_pred - y_test)**2))
print('ElasticNet Besr Params RMSE: {:.2f}'.format(rmse_best_elasticnet))

mse_best_elasticnet = np.mean((y_best_elasticnet_pred - y_test)**2)
print('ElasticNet Besr Params Mean Squared Error: {:.2f}'.format(mse_best_elasticnet))

''' The ElasticNet model performed the worst because it has a more complex approach
to regularization compared to other models used in this analysis, which could lead to
to data retraining. '''

# 5) Linear Regression with Polynomial Features
num_transformer_polynomial = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2))
])

preprocessor_polynomial = ColumnTransformer(transformers=[
    ('num_polynomial', num_transformer_polynomial, num_cols),
    ('cat', cat_transformer, cat_cols)
])

model_polynomial = Pipeline(steps=[
    ('preprocessor_polynomial', preprocessor_polynomial),
    ('regressor_polynomial', LinearRegression())
])

model_polynomial.fit(X_train, y_train)

y_polynomial_pred = model_polynomial.predict(X_test)

rmse_polynomial = np.sqrt(np.mean((y_polynomial_pred - y_test)**2))
print('Linear Regression with Polynomial Features RMSE: {:.2f}'.format(rmse_polynomial))

mse_polynomial = np.mean((y_polynomial_pred - y_test)**2)
print('Linear Regression with Polynomial Features Mean Squared Error: {:.2f}'.format(mse_polynomial))

""" A linear regression model with polynomial functions performs slightly better than
other models in terms of RMSE and root mean square error, indicating that
on average it gives the most accurate forecasts. """

# 6) RandomForestRegressor
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf_regressor', RandomForestRegressor(random_state=42))
])

model_rf.fit(X_train, y_train)

y_rf_pred = model_rf.predict(X_test)

rmse_rf = np.sqrt(np.mean((y_rf_pred - y_test)**2))
print('RandomForestRegressor RMSE: {:.2f}'.format(rmse_rf))

mse_rf = np.mean((y_rf_pred - y_test)**2)
print('RandomForestRegressor Mean Squared Error: {:.2f}'.format(mse_rf))

''' The results show that the random forest model performs worse than the
polynomial regression, in terms of RMSE and root mean square error. '''

# 7) GradientBoostingRegressor
model_gradientboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gradientboost_regressor', GradientBoostingRegressor(random_state=42))
])

model_gradientboost.fit(X_train, y_train)

y_gradientboost_pred = model_gradientboost.predict(X_test)

rmse_gradientboost = np.sqrt(np.mean((y_gradientboost_pred - y_test)**2))
print('GradientBoostingRegressor RMSE: {:.2f}'.format(rmse_gradientboost))

mse_gradientboost = np.mean((y_gradientboost_pred - y_test)**2)
print('GradientBoostingRegressor Mean Squared Error: {:.2f}'.format(mse_gradientboost))

''' The results show that the GradientBoostingRegressor model does not perform as well as
like some other models tested (like LinearRegression and KNeighborsRegressor). '''

# 8) AdaBoostRegressor
model_adaboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('adaboost_regressor', AdaBoostRegressor())
])

model_adaboost.fit(X_train, y_train)

y_adaboost_pred = model_adaboost.predict(X_test)

rmse_adaboost = np.sqrt(np.mean((y_adaboost_pred - y_test)**2))
print('GradientBoostingRegressor RMSE: {:.2f}'.format(rmse_adaboost))

mse_adaboost = np.mean((y_adaboost_pred - y_test)**2)
print('GradientBoostingRegressor Mean Squared Error: {:.2f}'.format(mse_adaboost))

''' Compared to other models trained on the same dataset, AdaBoostRegressor
performs worse than some models such as LinearRegression, PolynomialFeatures,
KNeighborsRegressor and XGBoost Regressor. AdaBoostRegressor also takes significantly 
more time for training than most other models.
    This model is not the best choice for this particular data set. '''

# 9) XGBRegressor
model_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb_regressor', xgb.XGBRegressor())
])

model_xgb.fit(X_train, y_train)

y_xgb_pred = model_xgb.predict(X_test)

rmse_xgb = np.sqrt(np.mean((y_xgb_pred - y_test)**2))
print('XGBRegressor RMSE: {:.2f}'.format(rmse_xgb))

mse_xgb = np.mean((y_xgb_pred - y_test)**2)
print('XGBRegressor Mean Squared Error: {:.2f}'.format(mse_xgb))

# A lower MSE value indicates a better fit of the model to the data.

# 10) BaggingRegressor
model_bagging = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('bagging_regressor', BaggingRegressor(estimator=xgb.XGBRegressor()))
])

model_bagging.fit(X_train, y_train)

y_bagging_pred = model_bagging.predict(X_test)

rmse_bagging = np.sqrt(np.mean((y_bagging_pred - y_test)**2))
print('BaggingRegressor RMSE: {:.2f}'.format(rmse_bagging))

mse_bagging = np.mean((y_bagging_pred - y_test)**2)
print('BaggingRegressor Mean Squared Error: {:.2f}'.format(mse_bagging))

''' The results show that the BaggingRegressor model performs slightly better than 
many others models, in terms of RMSE and root mean square error. '''

# 11) BlendingRegressor
class BlendingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, estimators, meta_estimator):
        self.estimators = estimators
        self.meta_estimator = meta_estimator

    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X, y)
        self.meta_estimator.fit(self._transform_base_preds(X), y)
        return self

    def predict(self, X):
        return self.meta_estimator.predict(self._transform_base_preds(X))

    def _transform_base_preds(self, X):
        base_preds = []
        for est in self.estimators:
            base_preds.append(est.predict(X))
        return np.column_stack(base_preds)


estimators = [
    LinearRegression(),
    RandomForestRegressor(),
    GradientBoostingRegressor()
]

meta_estimator = LinearRegression()

model_blending = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('blending_regressor', BlendingRegressor(estimators, meta_estimator))
])

model_blending.fit(X_train, y_train)

y_blending_pred = model_blending.predict(X_test)

rmse_blending = np.sqrt(np.mean((y_blending_pred - y_test) ** 2))
print('BlendingRegressor RMSE: {:.2f}'.format(rmse_blending))

mse_blending = np.mean((y_blending_pred - y_test) ** 2)
print('BlendingRegressor Mean Squared Error: {:.2f}'.format(mse_blending))

# A lower MSE value indicates a better fit of the model to the data.
# This is the one of the best models.

# 12) HistGradientBoostingRegressor
model_hist_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('hist_gb_regressor', HistGradientBoostingRegressor())
])

model_hist_gb.fit(X_train, y_train)

y_hist_gb_pred = model_hist_gb.predict(X_test)

rmse_hist_gb = np.sqrt(np.mean((y_hist_gb_pred - y_test)**2))
print('HistGradientBoostingRegressor RMSE: {:.2f}'.format(rmse_hist_gb))

mse_hist_gb = np.mean((y_hist_gb_pred - y_test)**2)
print('HistGradientBoostingRegressor Mean Squared Error: {:.2f}'.format(mse_hist_gb))

# A lower MSE value indicates a better fit of the model to the data.

''' Judging by the results, the Linear Regression with PolynomialFeatures showed the best 
results of all tested models with the lowest RMSE and MSE values. '''

''' So, the models with the lowest RMSE and MSE values are:
    PolynomialFeatures (the best result), 
    LinearRegression and BlendingRegressor with the same results, 
    xgb.XGBRegressor '''

# A) Cross Validation for Linear Regression with Polynomial Features
cv_polynomial = KFold(n_splits=5, shuffle=True, random_state=42)
# Or
# cv_polynomial = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
neg_mse_scores_polynomial = cross_val_score(model_polynomial, analysis_data.drop(exclude_cols, axis=1),
                                          analysis_data['price'], cv=cv_polynomial, scoring='neg_mean_squared_error')

print('Cross-Validation RMSE Scores (Polynomial): {:.2f}'.format(np.sqrt(-neg_mse_scores_polynomial.mean())))

mse_scores_polynomial = -neg_mse_scores_polynomial
mse_mean_polynomial = mse_scores_polynomial.mean()

print('Mean RMSE (Polynomial): {:.2f}'.format(mse_mean_polynomial))

# B) Cross Validation for Linear Regression
cv_linear = KFold(n_splits=5, shuffle=True, random_state=42)
# Or
# cv_linear = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
neg_mse_scores_linear = cross_val_score(model_regression, analysis_data.drop(exclude_cols, axis=1),
                                          analysis_data['price'], cv=cv_linear, scoring='neg_mean_squared_error')

print('Cross-Validation RMSE Scores (Linear Regression): {:.2f}'.format(np.sqrt(-neg_mse_scores_linear.mean())))

mse_scores_linear = -neg_mse_scores_linear
mse_mean_linear = mse_scores_linear.mean()

print('Mean RMSE (Linear Regression): {:.2f}'.format(mse_mean_linear))

# C) Cross Validation for BlendingRegressor
cv_blending = KFold(n_splits=5, shuffle=True, random_state=42)
# Or
# cv_blending = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
neg_mse_scores_blending = cross_val_score(model_blending, analysis_data.drop(exclude_cols, axis=1),
                                          analysis_data['price'], cv=cv_blending, scoring='neg_mean_squared_error')

print('Cross-Validation RMSE Scores (BlendingRegressor): {:.2f}'.format(np.sqrt(-neg_mse_scores_blending.mean())))

mse_scores_blending = -neg_mse_scores_blending
mse_mean_blending = mse_scores_blending.mean()

print('Mean RMSE (BlendingRegressor): {:.2f}'.format(mse_mean_blending))

# D) Cross Validation for xgb.XGBRegressor
cv_xgb_regression = KFold(n_splits=5, shuffle=True, random_state=42)
# Or
# cv_xgb_regression = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
neg_mse_scores_xgb_regression = cross_val_score(model_xgb, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_xgb_regression,
                                         scoring='neg_mean_squared_error')
mse_scores_xgb_regression = -neg_mse_scores_xgb_regression
mse_mean_xgb_regression = mse_scores_xgb_regression.mean()

print('Cross-Validation RMSE Scores (xgb.XGBRegressor): {:.2f}'.format(np.sqrt(mse_mean_xgb_regression)))
print('Mean RMSE (xgb.XGBRegressor): {:.2f}'.format(mse_mean_xgb_regression))

# Or
cv_xgb_regression = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores_xgb_regression = []
rmse_scores_xgb_regression = []

for train_index, test_index in cv_xgb_regression.split(analysis_data):
    X_train, X_test = analysis_data.drop(exclude_cols, axis=1).iloc[train_index], \
    analysis_data.drop(exclude_cols, axis=1).iloc[test_index]
    y_train, y_test = analysis_data['price'].iloc[train_index], analysis_data['price'].iloc[test_index]
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    mse_scores_xgb_regression.append(mean_squared_error(y_test, y_pred))
    rmse_scores_xgb_regression.append(np.sqrt(mean_squared_error(y_test, y_pred)))

mse_mean_xgb_regression = np.mean(mse_scores_xgb_regression)
rmse_mean_xgb_regression = np.sqrt(mse_mean_xgb_regression)

print('Cross-Validation RMSE Scores (xgb.XGBRegressor): {:.2f}'.format(rmse_mean_xgb_regression))
print('Mean RMSE (xgb.XGBRegressor): {:.2f}'.format(mse_mean_xgb_regression))

# E) Cross Validation for BaggingRegressor
cv_bagging = KFold(n_splits=5, shuffle=True, random_state=42)
# Or
# cv_bagging = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
neg_mse_scores_bagging = cross_val_score(model_bagging, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_bagging, scoring='neg_mean_squared_error')
mse_scores_bagging = -neg_mse_scores_bagging
mse_mean_bagging = mse_scores_bagging.mean()

print('Cross-Validation RMSE Scores (BaggingRegressor): {:.2f}'.format(np.sqrt(mse_mean_bagging)))
print('Mean RMSE (BaggingRegressor): {:.2f}'.format(mse_mean_bagging))

# Additional Cross-Validations:

# 1) Cross Validation for RandomForestRegressor
cv_rf_regression = KFold(n_splits=5, shuffle=True, random_state=42)
# Or
# cv_rf_regression = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
neg_mse_scores_rf_regression = cross_val_score(model_rf, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_rf_regression, scoring='neg_mean_squared_error')

print('Cross-Validation RMSE Scores (RandomForestRegressor): {:.2f}'.format(np.sqrt(-neg_mse_scores_rf_regression.mean())))

mse_scores_rf_regression = -neg_mse_scores_rf_regression
mse_mean_rf_regression = mse_scores_rf_regression.mean()

print('Mean RMSE (RandomForestRegressor): {:.2f}'.format(mse_mean_rf_regression))

# 2) Cross Validation for HistGradientBoostingRegressor
cv_hist_gb = KFold(n_splits=5, shuffle=True, random_state=42)
# Or
# cv_hist_gb = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
neg_mse_scores_hist_gb = cross_val_score(model_hist_gb, analysis_data.drop(exclude_cols, axis=1),
                                 analysis_data['price'], cv=cv_hist_gb,
                                         scoring='neg_mean_squared_error')
mse_scores_hist_gb = -neg_mse_scores_hist_gb
mse_mean_hist_gb = mse_scores_hist_gb.mean()

print('Cross-Validation RMSE Scores (HistGradientBoostingRegressor): {:.2f}'.format(np.sqrt(mse_mean_hist_gb)))
print('Mean RMSE (HistGradientBoostingRegressor): {:.2f}'.format(mse_mean_hist_gb))

# 1) KNeighborsRegressor
# 2) LinearRegression
# 3) DecisionTreeRegressor with tuned hyperparameters
# 4) ElasticNet with tuned hyperparameters
# 5) Linear Regression with Polynomial Features
# 7) GradientBoostingRegressor
# 8) AdaBoostRegressor
# 12) HistGradientBoostingRegressor

models = [
    ('KNN Regressor', y_knn_pred, 'red'),
    ('Linear Regression', y_regression_pred, 'pink'),
    ('PolynomialFeatures', y_polynomial_pred, 'green'),
    ('DecisionTreeRegressor', y_best_tree_regression_pred, 'blue'),
    ('ElasticNetRegressor', y_best_elasticnet_pred, 'orange'),
    ('GradientBoostingRegressor', y_gradientboost_pred, 'brown'),
    ('AdaBoostRegressor', y_adaboost_pred, 'purple'),
    ('HistGradientBoostingRegressor', y_hist_gb_pred, 'grey')
]

# А) Scatter graphs of predicted and actual prices
fig, axs = plt.subplots(2, 4, figsize=(18, 8))
fig.subplots_adjust(hspace=1.2, wspace=0.8)
axs = axs.ravel()

for i, (model_name, y_pred, color) in enumerate(models):
    min_length = min(len(y_test), len(y_pred))
    sns.scatterplot(x=y_test[:min_length], y=y_pred[:min_length], ax=axs[i], c=color)
    axs[i].plot([0, 3200], [0, 3200], 'k--')
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].set_title(model_name)

# B) Error distribution graphs
fig, ax = plt.subplots(figsize=(18, 8))

for i, (model_name, y_pred, color) in enumerate(models):
    min_length = min(len(y_test), len(y_pred))
    distribution = y_pred[:min_length] - y_test[:min_length]
    sns.kdeplot(distribution, label=model_name, color=color)

ax.set_xlabel('Error')
ax.set_ylabel('Frequency')
ax.set_title('Regression models by error distribution')
ax.legend()

plt.show()

# Four best models:
# 6) RandomForestRegressor
# 9) xgb.XGBRegressor
# 10) BaggingRegressor
# 11) BlendingRegressor

best_models = [
    ('RandomForestRegressor', y_rf_pred, 'green'),
    ('BaggingRegressor', y_bagging_pred, 'purple'),
    ('XGBoost Regressor', y_xgb_pred, 'brown'),
    ('BlendingRegressor', y_blending_pred, 'gray')
]

results = []
for best_model_name, y_pred, color in best_models:
    min_length = min(len(y_test), len(y_pred))
    distribution = y_pred[:min_length] - y_test[:min_length]
    for i in y_test.index[:min_length]:
        results.append((best_model_name, y_test[i], distribution[i]))

results_df = pd.DataFrame(results, columns=['model', 'y_test', 'distribution'])

# С) Error distribution graphs
fig, ax = plt.subplots(figsize=(18, 8))
for i, (best_model_name, y_pred, color) in enumerate(best_models):
    min_length = min(len(y_pred), len(y_test))
    distribution = y_pred[:min_length] - y_test[:min_length]
    sns.kdeplot(distribution, label=best_model_name, color=color)

ax.set_xlabel('Error')
ax.set_ylabel('Frequency')
ax.set_title('Regression models by error distribution')
ax.legend()

plt.show()

# D) regplot error distributions
fig, axs = plt.subplots(1, 3, figsize=(18, 8))
fig.subplots_adjust(hspace=1.2, wspace=1.5)
axs = axs.ravel()

for i, (best_model_name, y_pred, color) in enumerate(best_models[:3]):
    min_length = min(len(y_pred), len(y_test))
    sns.regplot(x=y_test[:min_length], y=y_pred[:min_length]-y_test[:min_length], ax=axs[i], scatter=True, color=color)
    axs[i].set_title(best_model_name)
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].set_ylim([-5000, 5000])

fig.suptitle('Regression models by frequency (ax y) of errors(ax x)')
plt.tight_layout()
plt.show()

# Visualizations of cross-validation

# 1) Visualization of cross-validation of the best model - xgb.XGBRegressor
fig, ax = plt.subplots(figsize=(8, 4))

ax.barh(range(len(rmse_scores_xgb_regression)), rmse_scores_xgb_regression, color='brown')
ax.set_yticks(range(len(rmse_scores_xgb_regression)))
ax.set_yticklabels(['Fold {}'.format(i+1) for i in range(len(rmse_scores_xgb_regression))])
plt.axvline(rmse_mean_xgb_regression, color='y', linestyle='dashed', linewidth=2.5,
            label=f"RMSE: {round(rmse_mean_xgb_regression)}")
for i, v in enumerate(rmse_scores_xgb_regression):
    plt.text(v+7, i, f"{round(v)}", ha='left', va='center', color="black", fontsize=12,
             bbox=dict(facecolor='orange', edgecolor='pink', boxstyle='round4,pad=0.4'))
ax.set_xlim(0, rmse_mean_xgb_regression.max() * 1.225)
plt.xlabel('RMSE')
plt.ylabel('Fold')
plt.title('Cross-validation results of XGBoost Regression')
plt.legend(loc='upper right', fontsize=13)
plt.gca().grid(False)

plt.show()

# 2) Visualization of cross-validation of the BlendingRegressor
fig, ax = plt.subplots(figsize=(8, 4))

ax.barh(range(len(mse_scores_blending)), np.sqrt(mse_scores_blending), color='grey')
ax.set_yticks(range(len(mse_scores_blending)))
ax.set_yticklabels(['Fold {}'.format(i+1) for i in range(len(mse_scores_blending))])
plt.axvline(np.sqrt(mse_mean_blending), color='y', linestyle='dashed', linewidth=2.5,
            label=f"RMSE: {round(np.sqrt(mse_mean_blending))}")

for i, v in enumerate(mse_scores_blending):
    plt.text(np.sqrt(v)+7, i, f"{round(np.sqrt(v))}", ha='left', va='center', color="black", fontsize=12,
             bbox=dict(facecolor='orange', edgecolor='pink', boxstyle='round4,pad=0.4'))

ax.set_xlim(0, np.sqrt(mse_scores_blending).max() * 1.225)
plt.xlabel('RMSE')
plt.ylabel('Fold')
plt.title('Cross-validation results of BlendingRegressor')
plt.legend(loc='upper right', fontsize=13)
plt.gca().grid(False)

plt.show()

# 3) Visualization of cross-validation - BaggingRegressor
fig, ax = plt.subplots(figsize=(8, 4))

ax.barh(range(len(mse_scores_bagging)), np.sqrt(mse_scores_bagging), color='purple')
ax.set_yticks(range(len(mse_scores_bagging)))
ax.set_yticklabels(['Fold {}'.format(i+1) for i in range(len(mse_scores_bagging))])
plt.axvline(np.sqrt(mse_mean_bagging), color='y', linestyle='dashed', linewidth=2.5,
            label=f"RMSE: {round(np.sqrt(mse_mean_bagging))}")
for i, v in enumerate(mse_scores_bagging):
    plt.text(np.sqrt(v)+7, i, f"{round(np.sqrt(v))}", ha='left', va='center', color="black", fontsize=12,
             bbox=dict(facecolor='orange', edgecolor='pink', boxstyle='round4,pad=0.4'))
ax.set_xlim(0, np.sqrt(mse_mean_bagging).max() * 1.225)
plt.xlabel('RMSE')
plt.ylabel('Fold')
plt.title('Cross-validation results of BaggingRegressor')
plt.legend(loc='upper right', fontsize=13)
plt.gca().grid(False)

plt.show()

# Additional visualizations

# 1) Visualization of cross-validation of the RandomForestRegressor
fig, ax = plt.subplots(figsize=(8, 4))

ax.barh(range(len(mse_scores_rf_regression)), np.sqrt(mse_scores_rf_regression), color='green')
ax.set_yticks(range(len(mse_scores_rf_regression)))
ax.set_yticklabels(['Fold {}'.format(i+1) for i in range(len(mse_scores_rf_regression))])
plt.axvline(np.sqrt(mse_mean_rf_regression), color='y', linestyle='dashed', linewidth=2.5,
            label=f"RMSE: {round(np.sqrt(mse_mean_rf_regression))}")
for i, v in enumerate(mse_scores_rf_regression):
    plt.text(np.sqrt(v)+7, i, f"{round(np.sqrt(v))}", ha='left', va='center', color="black", fontsize=12,
             bbox=dict(facecolor='orange', edgecolor='pink', boxstyle='round4,pad=0.4'))
ax.set_xlim(0, np.sqrt(mse_mean_rf_regression).max() * 1.225)
plt.xlabel('RMSE')
plt.ylabel('Fold')
plt.title('Cross-validation results of RandomForestRegressor')
plt.legend(loc='upper right', fontsize=13)
plt.gca().grid(False)

plt.show()

# 2) Visualization of cross-validation of the Linear Regression with Polynomial Features
fig, ax = plt.subplots(figsize=(8, 4))

ax.barh(range(len(mse_scores_polynomial)), np.sqrt(mse_scores_polynomial), color='blue')
ax.set_yticks(range(len(mse_scores_polynomial)))
ax.set_yticklabels(['Fold {}'.format(i+1) for i in range(len(mse_scores_polynomial))])
plt.axvline(np.sqrt(mse_mean_polynomial), color='y', linestyle='dashed', linewidth=2.5,
            label=f"RMSE: {round(np.sqrt(mse_mean_polynomial))}")

for i, v in enumerate(mse_scores_polynomial):
    plt.text(np.sqrt(v)+7, i, f"{round(np.sqrt(v))}", ha='left', va='center', color="black", fontsize=12,
             bbox=dict(facecolor='orange', edgecolor='pink', boxstyle='round4,pad=0.4'))

ax.set_xlim(0, np.sqrt(mse_scores_polynomial).max() * 1.225)
plt.xlabel('RMSE')
plt.ylabel('Fold')
plt.title('Cross-validation results of Linear Regression with Polynomial Features')
plt.legend(loc='upper right', fontsize=13)
plt.gca().grid(False)

plt.show()

''' All models are evaluated using cross-validation with the KFold(5) strategy,
shuffling the data and using negative root mean square error as the evaluation metric.
    On the stage of cross-validation the xgb.XGBRegressor model has the best performance 
with the lowest RMS error and root mean square error among all models. 
    But the resuls on this stage are worse than on the first stage of research.
    Judging by all results, the Linear Regression with PolynomialFeatures showed the best 
results of all tested models with the lowest RMSE and MSE values. '''

''' Overall, the MSE and RMSE values indicate that the models are not very good at 
predicting price. To improve the performance of the furniture price forecasting model, 
I recommend adding the following data into the source table:
    1) the cost of manufacturing products;
    2) the material from which the furniture is made;
    3) furniture rating (according to reviews on the website);
    4) warranty period of furniture;
    5) the number of photographs of furniture on the site; time of order placement;
    6) information about buyers: name (maybe the same person buys certain furniture),
gender, age, education, income level, residence: city / village, number of reviews on 
the site, engagement in social networks, online ordering time, duration of site viewing, 
online method order: website / social networks, from desktop device / from mobile device,
availability of email newsletters;
    7) time of purchase: time of year / month, working day / weekend / holiday, 
time of day;
    8) payment method: credit card / debit card / cash;
    9) cost, method and speed of delivery;
    10) whether there was a sale or not. '''
