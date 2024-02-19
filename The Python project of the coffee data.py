# 1. Which countries are major coffee exporters?

import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import ttest_ind

# Connect to the database file
conn = sqlite3.connect('coffee.db')

# Load the coffee data into a DataFrame
coffee_data = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv')

# Save the DataFrame to the database
coffee_data.to_sql('coffee', conn, if_exists='replace', index=False)

# Group the data by country and compute the total number of bags exported by each country
query = """
    SELECT country_of_origin, SUM(number_of_bags) AS total_bags
    FROM coffee
    GROUP BY country_of_origin
    ORDER BY total_bags DESC
"""

country_data_sum = pd.read_sql_query(query, conn)

# Compute the total number of bags exported by all countries
total_bags = country_data_sum['total_bags'].sum()

# Keep only the top 10 exporters and combine the rest into an 'Other' category
exporters = country_data_sum.head(10)
other_bags = total_bags - exporters['total_bags'].sum()
other_row = pd.DataFrame({'country_of_origin': ['Other'], 'total_bags': [other_bags]})
top_exporters_df = pd.concat([exporters, other_row], ignore_index=True)

# Define colors for the pie chart
colors = ['#0077c8', '#0081a7', '#008d87', '#009970', '#3baf3f', '#a6ce39', '#f1d302', '#f68b1f', '#ed1c24', '#c21f39', '#5f4b8b']

# Define labels and sizes for the pie chart
labels = top_exporters_df.apply(
    lambda row: '{} {:.1f}%'.format(row['country_of_origin'], row['total_bags'] / total_bags * 100),
    axis=1)
sizes = top_exporters_df['total_bags']

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(13, 11.5))

# Draw the pie chart for the number of bags
wedges, _ = ax.pie(sizes, colors=colors, wedgeprops=dict(width=0.5))

# Add labels with country names and percentages
bbox_props = dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
kw = dict(arrowprops=dict(arrowstyle='-'), bbox=bbox_props, zorder=0, va='center')
for i, wedge in enumerate(wedges):
    ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: 'right', 1: 'left'}[int(np.sign(x))]
    connectionstyle = 'angle,angleA=0,angleB={}'.format(ang)
    kw['arrowprops'].update({'connectionstyle': connectionstyle})
    ax.annotate(labels[i], xy=(x, y), xytext=(1.1*np.sign(x), 1.3*y),
                fontsize=10, horizontalalignment=horizontalalignment, **kw)
ax.set_title('Top 10 Coffee Exporters', fontweight='bold', fontsize=16)
legend_labels = []
for i, row in top_exporters_df.iterrows():
    label = '{} {:.1f}% ({:,} bags)'.format(row['country_of_origin'],
                                            row['total_bags'] / total_bags * 100,
                                            row['total_bags'])
    legend_labels.append(label)
plt.legend(wedges, legend_labels, title='Top 10 Coffee Exporters', loc='center left', bbox_to_anchor=(1.3, 0.5))

plt.tight_layout(pad=1.5)

plt.show()

# 2. What are the correlations between different coffee rating metrics?

column_for_matrix = ['body', 'acidity', 'flavor', 'aroma', 'sweetness', 'cupper_points', 'aftertaste', 'balance',
                     'uniformity', 'clean_cup', 'total_cup_points']
corr_matrix = coffee_data[column_for_matrix].corr()
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap='coolwarm')

for i in range(len(column_for_matrix)):
    for j in range(len(column_for_matrix)):
        text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2),
                       ha="center", va="center", color="black")

ax.set_xticks(np.arange(len(column_for_matrix)))
ax.set_yticks(np.arange(len(column_for_matrix)))
ax.set_xticklabels(column_for_matrix)
ax.set_yticklabels(column_for_matrix)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("Correlation Matrix")

plt.show()

''' We see that there are several strong positive correlations.
For example, there is a strong positive correlation between flavor and aftertaste.
(0.896), as well as between flavor and total cup points (0.874).
This suggests that coffees with a higher flavor rating are likely to be rated higher
aftertaste and a higher total number of cups.
    Similarly, there are several strong positive correlations between other pairs of variables,
such as acidity and total cup points (0.797), as well as between balance and aftertaste (0.810), 
between balance and total cup points (total cups) (0.829). On the other hand, there are also 
several weak or negative correlations. For example, between sweetness and all other variables
(correlation coefficients range from 0.22 to 0.55), indicating that sweetness is not strongly
influences estimates of other indicators.
    So,
    body, acidity, flavor, aroma, aftertaste, balance have a very weak positive correlation
with uniformity, clean_cup and sweetness, with all other indicators it has a strong
positive correlation;
    sweetness has a weak positive correlation with total_cup_points, uniformity and clean_cup,
has a very weak positive correlation with all other indicators;
    cupper_points has a very weak positive correlation with uniformity, clean_cup and sweetness,
with body has a weak positive correlation, with all other indicators it has a strong correlation
positive correlation;
    uniformity has a weak positive correlation with total_cup_points, clean_cup and sweetness,
has a very weak positive correlation with all other indicators;
    clean_cup has a weak positive correlation with total_cup_points, uniformity and sweetness,
has a very weak positive correlation with all other indicators;
    total_cup_points have a weak positive correlation with uniformity, clean_cup and sweetness,
has a strong positive correlation with all other indicators.'''

# Find attributes with correlation >= 0.65, ignoring correlations 'sweetness', 'uniformity', 'clean_cup':
corr_mask = np.abs(corr_matrix) >= 0.65
corr_mask = corr_mask.any(axis=1)
selected_columns = corr_mask.loc[corr_mask].index.tolist()

for col in selected_columns:
    if col in ['sweetness', 'uniformity', 'clean_cup']:
        continue
    corr_with_col = corr_matrix[col]
    cols_to_drop = corr_with_col[(corr_with_col <= 0.65) & (corr_with_col >= -0.65)].index.tolist()
    selected_columns = [col for col in selected_columns if col not in cols_to_drop]

print("Selected columns with correlations >= 0.65: ")
print(selected_columns)

# Let's build a correlation matrix between coffee evaluation indicators with correlations >= 0.65
corr_matrix_best_correlations = coffee_data.loc[:, selected_columns].corr()
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix_best_correlations, cmap='coolwarm')

for i in range(len(selected_columns)):
    for j in range(len(selected_columns)):
        text = ax.text(j, i, round(corr_matrix_best_correlations.iloc[i, j], 2),
                       ha="center", va="center", color="black")

ax.set_xticks(np.arange(len(selected_columns)))
ax.set_yticks(np.arange(len(selected_columns)))
ax.set_xticklabels(selected_columns)
ax.set_yticklabels(selected_columns)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("Correlation Matrix with Selected Columns")

plt.show()

# Or using Seaborn
# Let's build a correlation matrix between all coffee evaluation indicators
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Let's build a correlation matrix between coffee evaluation indicators with correlations >= 0.65
sns.heatmap(corr_matrix_best_correlations, annot=True)
plt.show()

#4. How does the bean's color affect the overall grade of coffee?

# 1)
pivot = coffee_data.pivot_table(index=["species"], columns=["color"], values='total_cup_points', aggfunc=np.average)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot.values, cmap='viridis')

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_xticklabels(pivot.columns)
ax.set_yticklabels(pivot.index)
ax.set_xlabel('Color')
ax.set_ylabel('Species')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        text = ax.text(j, i, '{:.2f}'.format(pivot.values[i][j]),
                       ha="center", va="center", color="w")

plt.show()

''' Dark roasted coffee beans generally receive higher overall scores,
than light roasted beans, regardless of type. For example, Robusta and Arabica coffee beans
have higher overall scores with dark roasts than with light roasts. This agrees
with the general consensus that dark roasted coffee is more flavorful and of higher quality,
than light roasted coffee. '''

# Or
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.pcolor(pivot.values, cmap='viridis')

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(pivot.columns))+0.5)
ax.set_yticks(np.arange(len(pivot.index))+0.5)
ax.set_xticklabels(pivot.columns, rotation=45)
ax.set_yticklabels(pivot.index)
ax.set_xlabel('Color')
ax.set_ylabel('Species')

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        text = ax.text(j + 0.5, i + 0.5, '{:.2f}'.format(pivot.values[i, j]),
                       ha='center', va='center', color='w')

plt.show()

# 2)
plt.figure(figsize=(10, 6))

colors = [(0.1, 0.2, 0.4), (0.2, 0.4, 0.6), 'green']

species = pivot.index
n_species = len(species)
bar_width = 0.15

for i, col in enumerate(pivot.columns):
    plt.bar(np.arange(n_species) + i * bar_width, pivot[col], width=bar_width, color=colors[i], alpha=0.8)

plt.xlabel('Species')
plt.ylabel('Average Total Cup Points')
plt.title('Average Total Cup Points by Species and Bean Color')
plt.xticks(np.arange(n_species) + 0.225, species, rotation=45, ha='right')
plt.legend(['Blue-Green', 'Bluish-Green', 'Green',  'None'])

plt.ylim(72, 85)

yticks = np.arange(72, 85, 1)
plt.yticks(yticks, [str(y) for y in yticks])

for i in range(n_species):
    for j in range(len(colors)):
        plt.text(i + j * bar_width, pivot.iloc[i, j] + 0.1, '{:.2f}'.format(pivot.iloc[i, j]), ha='center')

plt.show()

# Or
plt.figure(figsize=(10, 6))

species = pivot.index
n_species = len(species)
bar_width = 0.15
for i, col in enumerate(pivot.columns):
    plt.barh(np.arange(n_species) + i * bar_width, pivot.iloc[:, i], height=bar_width, color=colors[i], alpha=0.8)

plt.xlabel('Average Total Cup Points')
plt.ylabel('Species')
plt.title('Average Total Cup Points by Species and Bean Color')
plt.yticks(np.arange(n_species) + 0.225, species, rotation=45, ha='right')
plt.legend(['Blue-Green', 'Bluish-Green', 'Green',  'None'])

plt.xlim(72, 85)

xticks = np.arange(72, 85, 1)
plt.xticks(xticks, [str(x) for x in xticks])

for i in range(n_species):
    for j in range(len(colors)):
        plt.text(pivot.iloc[i, j] + 0.1, i + j * bar_width, '{:.2f}'.format(pivot.iloc[i, j]), va='center')

plt.show()

# Or using Seaborn

melted = pivot.reset_index().melt(id_vars=['species'], var_name='color', value_name='total_cup_points')

sns.set_palette(colors)

g = sns.catplot(x='species', y='total_cup_points', hue='color', data=melted, kind='bar', height=6, aspect=1.5)

bar_width = 0.2
g.despine(left=True, bottom=True)
g.set_ylabels('Average Total Cup Points')
g.set_xlabels('Species')
g.set_xticklabels(rotation=45, ha='right')
plt.ylim(72, 85)

yticks = np.arange(72, 85, 1)

for i, bar in enumerate(g.ax.patches):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
             '{:.2f}'.format(pivot.iloc[i % n_species, i // n_species]),
             ha='center')

plt.show()

# Or
plt.figure(figsize=(10, 6))

ax = sns.barplot(x='species', y='total_cup_points', hue='color', data=melted, alpha=0.8)

plt.xlabel('Species')
plt.ylabel('Average Total Cup Points')
plt.title('Average Total Cup Points by Species and Bean Color')
plt.xticks(rotation=45, ha='right')
plt.ylim(72, 85)

yticks = np.arange(72, 85, 1)
plt.yticks(yticks, [str(y) for y in yticks])

for i, patch in enumerate(ax.patches):
    height = patch.get_height()
    ax.text(patch.get_x() + patch.get_width() / 2, height + 0.1, '{:.2f}'.format(height), ha='center')

plt.show()

# Or
g = sns.catplot(y='species', x='total_cup_points', hue='color', data=melted, kind='bar', height=6, aspect=1.5, orient='horizontal')
sns.set_palette(colors)
g.despine(bottom=True)
g.set_xlabels('Average Total Cup Points')
g.set_ylabels('Species')
plt.xlim(72, 85)

xticks = np.arange(72, 85, 1)
plt.xticks(xticks, [str(x) for x in xticks])

for i, bar in enumerate(g.ax.patches):
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
             '{:.2f}'.format(pivot.iloc[i % n_species, i // n_species]),
             va='center')

plt.show()

# 3)
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(pivot.columns):
    ax.scatter(pivot.index, pivot[color], color=colors[i], label=color)

ax.legend()
ax.set_xlabel('Species')
ax.set_ylabel('Average total cup points')

plt.show()

# Or using Seaborn
sns.scatterplot(x="species", y="total_cup_points", hue='color', data=melted, palette=colors)
sns.despine()

sns.set_palette(colors)

plt.xlabel('Species')
plt.ylabel('Total cup points')
plt.show()

#5. Does country of origin affect coffee quality?

# The data to analyze
table = pd.pivot_table(coffee_data, index=['country_of_origin'], values=['total_cup_points'], aggfunc=pd.Series.mean)

# Table depending on the country of origin
sort_table = table.sort_values(by='total_cup_points', ascending=False)
print(sort_table)

x = sort_table.index
y = sort_table['total_cup_points']

# Testing the null hypothesis: country of origin has a significant effect on total_cup_points
corr, pval = spearmanr(x, y)

print(f"Spearman's rank correlation coefficient: {corr:.3f}")
print(f"p-value: {pval:.3f}")

if pval < 0.05:
    print("Reject the null hypothesis: country of origin has a significant effect on total_cup_points")
else:
    print("Failed to reject the null hypothesis: country of origin has no significant effect"
          "at total_cup_points.")

# Table of countries of origin by number of representatives
counts = coffee_data['country_of_origin'].value_counts()
print(counts)
counts = counts.reindex(index=sort_table.index)

points = sort_table['total_cup_points']
max_points = points.max()
min_points = points.min()
mean_points = points.mean()

fig, axs = plt.subplots(figsize=(16, 8), ncols=2)

cm = plt.cm.get_cmap('YlGnBu')
normalize = plt.Normalize(vmin=min(counts.values), vmax=max(counts.values))
colors = [cm(normalize(count)) for count in counts.values]

axs[0].barh(list(reversed(counts.index)), list(reversed(counts.values)), color=colors, edgecolor='black', height=0.8)

axs[0].set_xlabel('Number of Coffee Samples')
axs[0].set_title('Coffee Samples by Country of Origin')

country_data_sum = coffee_data.groupby('country_of_origin')['number_of_bags'].sum().reset_index().sort_values(
    by='number_of_bags', ascending=False)

total_bags = country_data_sum['number_of_bags'].sum()

exporters = country_data_sum.head(10)
other_bags = total_bags - exporters['number_of_bags'].sum()
other_row = pd.DataFrame({'country_of_origin': ['Other'], 'number_of_bags': [other_bags]})
top_exporters_df = pd.concat([exporters, other_row], ignore_index=True)

legend_labels = []
for i, row in top_exporters_df.iterrows():
    label = '{} {:.1f}% ({:,} bags)'.format(row['country_of_origin'],
                                            row['number_of_bags'] / total_bags * 100,
                                            row['number_of_bags'])
    legend_labels.append(label)

axs[1].text(0.02, 0.98, '\n'.join(legend_labels), transform=axs[1].transAxes, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8), fontsize=14)
axs[1].set_title('Top 10 Coffee Exporters', loc='left', fontsize=20)
axs[1].axis('off')
plt.subplots_adjust(wspace=4)
plt.tight_layout(pad=4)

fig, ax = plt.subplots(figsize=(16, 8))

cm = plt.cm.get_cmap('YlGnBu')
normalize = plt.Normalize(vmin=min(points.values), vmax=max(points.values))
colors = [cm(normalize(point)) for point in points.values]

ax.barh(points.index, points.values, color=colors, edgecolor='black', height=0.8)
ax.set_xlabel('Total Cup Points')
ax.set_ylabel('')
ax.set_title('Total Cup Points by Country of Origin')
ax.axvline(max_points, color='y', linewidth=2, label='Max')
ax.axvline(min_points, color='g', linewidth=2, label='Min')
ax.axvline(mean_points, color='r', linewidth=2, label='Mean')
ax.set_xlim(70, 90)
ax.legend()
plt.yticks(rotation=0)

plt.gca().invert_yaxis()
plt.subplots_adjust(wspace=4)
plt.tight_layout(pad=4)
plt.show()

""" "total_cup_points" means the overall rating of the coffee on a scale of 0 to 100 based on factors such as:
like aroma, taste, aftertaste, acidity, body, balance and much more. The image shows the average
number of coffee points by country of origin. The height of each bar represents the average
number of points, with the highest average at the top and the lowest at the bottom.
Horizontal lines 77, 80 and 85 are reference lines indicating the range of the average
number of points.
    I got:
    Spearman's rank correlation coefficient: 0.039
    p-value: 0.820
    This means that there is no statistically significant correlation between country of origin and overall
coffee evaluation. A p value of 0.820 is significantly greater than 0.05 and is considered not statistically significant.
    So, country of origin does not have a significant impact on the overall coffee rating. But
I would like to note a few points:
    1) The image in the first problem shows the top 10 coffee exporting countries and the amount 
exported packages. Brazil, Colombia, Ethiopia, Guatemala and Uganda are among the top ten exporting countries and also
have relatively high average overall scores. For example, Mexico has a high rate total_cup_points 
and was ranked based on 236 representatives (first place). Guatemala also has high total_cup_points
indicator and was ranked based on 181 representatives (third place). Brazil, Uganda, Costa Rica, 
Nicaragua, El Salvador have high total_cup_points and were ranked by a large number of 
representatives.
    2) Colombia should be considered the world leader in coffee based on a combination of 
indicators: it was ranked 183 representatives (second place after Mexico), has a high 
total_cup_points - 83 and the largest number of exported packages is 41204.
    3) Honduras is underrated. It is among the top 10 coffee exporting countries and has 
been ranked for 53 representatives (high), but has a low total_cup_points.
    4) Also underrated are Haiti, which has the lowest total_cup_points, but
were assessed by 6 representatives (quite a high figure).
    5) Papua New Guinea is the leader in terms of total_cup_points, but was only rated at 1
representative, while the next highest Ethiopia was ranked with 44 representatives and is included
in the top 10 coffee exporting countries. Therefore, I believe that Ethiopia should be considered 
the country with the most quality coffee.
    6) Vietnam is second in terms of total_cup_points, was ranked by 8 representatives (quite
high), but is not among the top 10 coffee exporting countries. Also USA, China, India and others
countries have a high total_cup_points, were ranked based on a large number of representatives,
but are not among the top 10 coffee exporting countries. """

# 1
sort_table = table.reindex(table.sort_values(by='total_cup_points', ascending=False).index)
print(sort_table)

x = sort_table.index
y = sort_table['total_cup_points']
points = sort_table['total_cup_points']
max_points = points.max()
min_points = points.min()
mean_points = points.mean()

fig, ax = plt.subplots(figsize=(11, 6))

cm = plt.cm.get_cmap('YlGnBu')
normalize = plt.Normalize(vmin=min(points.values), vmax=max(points.values))
colors = [cm(normalize(point)) for point in points.values]

ax.bar(points.index, points.values, color=colors, capsize=10)

ax.axhline(max_points, color='g', linewidth=2, label='Max')
ax.axhline(min_points, color='black', linewidth=2, label='Min')
ax.axhline(mean_points, color='r', linewidth=2, label='Mean')
ax.set_ylim(70, 90)

ax.set_xlabel('Country of Origin')
ax.set_ylabel('Total Cup Points')
ax.set_title('Total Cup Points by Country of Origin')

plt.xticks(rotation=90)
ax.legend()

plt.show()

# 2

fig, ax = plt.subplots(figsize=(11, 6))
cm = plt.cm.get_cmap('YlGnBu')
normalize = plt.Normalize(vmin=min(points.values), vmax=max(points.values))
colors = [cm(normalize(point)) for point in points.values]

ax.barh(points.index, points.values, color=colors, capsize=10)

ax.axvline(max_points, color='g', linewidth=2, label='Max')
ax.axvline(min_points, color='black', linewidth=2, label='Min')
ax.axvline(mean_points, color='r', linewidth=2, label='Mean')
ax.set_xlim(70, 90)

ax.set_ylabel('Country of Origin')
ax.set_xlabel('Total Cup Points')
ax.set_title('Total Cup Points by Country of Origin')

plt.yticks(rotation=0)
ax.legend()
plt.gca().invert_yaxis()
plt.show()

# 3

fig, ax = plt.subplots(figsize=(11, 6))

melted = table.reset_index().melt(id_vars=['country_of_origin'], var_name='color', value_name='cup_points')
ax.scatter(melted['country_of_origin'], melted['cup_points'], marker='s', color='blue', alpha=0.8, s=60, linewidths=1)

ax.axhline(max_points, color='y', linewidth=2, label='Max')
ax.axhline(min_points, color='g', linewidth=2, label='Min')
ax.axhline(mean_points, color='r', linewidth=2, label='Mean')

x_pos = range(len(melted['country_of_origin'].unique()))
plt.vlines(x_pos, ymin=70, ymax=90, linestyles='dotted', colors='grey', alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(melted['country_of_origin'].unique(), rotation=90, fontsize=8)
ax.tick_params(axis='x', which='major', pad=10)

ax.set_ylim(70, 90)

ax.set_xlabel('Country of Origin')
ax.set_ylabel('Total Cup Points')
ax.set_title('Total Cup Points by Country of Origin')

ax.legend()

plt.show()

# Or using Seaborn

# 1
sort_table = table.sort_values(by='total_cup_points', ascending=False)
counts = coffee_data['country_of_origin'].value_counts().reindex(index=sort_table.index)
print(counts)
print(sort_table)

points = sort_table['total_cup_points']
max_points = points.max()
min_points = points.min()
mean_points = points.mean()

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

sns.countplot(y='country_of_origin', data=coffee_data, order=sort_table.index, palette='Blues', edgecolor='black', ax=axs[0])
axs[0].set_xlabel('Number of Coffee Samples')
axs[0].set_title('Coffee Samples by Country of Origin')
plt.gca().invert_yaxis()

sns.barplot(x='total_cup_points', y='country_of_origin', data=sort_table.reset_index(),
            palette='YlOrBr', edgecolor='black', ax=axs[1])
axs[1].set_xlabel('Total Cup Points')
axs[1].set_ylabel('')
axs[1].set_title('Total Cup Points by Country of Origin')
axs[1].axvline(max_points, color='g', linewidth=2, label='Max')
axs[1].axvline(min_points, color='black', linewidth=2, label='Min')
axs[1].axvline(mean_points, color='r', linewidth=2, label='Mean')
axs[1].set_xlim(70, 90)
axs[1].legend()

plt.subplots_adjust(wspace=0.6)
plt.show()

# 2

fig, ax = plt.subplots(figsize=(11, 6))

sns.barplot(x='country_of_origin', y='total_cup_points', data=coffee_data, order=sort_table.index,
            palette='YlOrBr', ax=ax, errorbar=('ci', False))
sns.despine()

max_points = sort_table['total_cup_points'].max()
min_points = sort_table['total_cup_points'].min()
mean_points = sort_table['total_cup_points'].mean()

ax.axhline(max_points, color='g', linewidth=2, label='Max')
ax.axhline(min_points, color='black', linewidth=2, label='Min')
ax.axhline(mean_points, color='r', linewidth=2, label='Mean')

ax.set_ylim(70, 90)
ax.set_xlabel('Country of Origin')
ax.set_ylabel('Total Cup Points')
ax.set_title('Total Cup Points by Country of Origin')

plt.xticks(rotation=90)
ax.legend()

plt.show()

# Or
sort_table = table.reindex(table.sort_values(by='total_cup_points', ascending=False).index)

max_points = sort_table.max()[0]
min_points = sort_table.min()[0]
mean_points = sort_table.mean()[0]

melted = table.reset_index().melt(id_vars=['country_of_origin'], var_name='color', value_name='cup_points')

sns.catplot(data=coffee_data, x='country_of_origin', y='total_cup_points', kind='bar', errorbar=('ci', False))

plt.axhline(max_points, color='y', linewidth=2, label='Max')
plt.axhline(min_points, color='g', linewidth=2, label='Min')
plt.axhline(mean_points, color='r', linewidth=2, label='Mean')

x_pos = range(len(melted['country_of_origin'].unique()))
plt.vlines(x_pos, ymin=70, ymax=90, linestyles='dotted', colors='grey', alpha=0.5)
plt.xticks(x_pos, rotation=90, fontsize=8)

plt.ylim(70, 90)

plt.xlabel('Country of Origin')
plt.title('Total Cup Points by Country of Origin')

plt.legend()

plt.show()

# 3
fig, ax = plt.subplots(figsize=(11, 6))

sns.barplot(x='total_cup_points', y='country_of_origin', data=sort_table.reset_index(), palette='YlOrBr')

ax.axvline(sort_table['total_cup_points'].max(), color='g', linewidth=2, label='Max')
ax.axvline(sort_table['total_cup_points'].min(), color='black', linewidth=2, label='Min')
ax.axvline(sort_table['total_cup_points'].mean(), color='r', linewidth=2, label='Mean')
ax.set_xlim(70, 90)

ax.set_ylabel('Country of Origin')
ax.set_xlabel('Total Cup Points')
ax.set_title('Total Cup Points by Country of Origin')

plt.yticks(rotation=0)
ax.legend()
plt.show()

# 4
fig, ax = plt.subplots(figsize=(11, 6))

sns.scatterplot(data=table, x='country_of_origin', y='total_cup_points', color='blue', alpha=0.8, s=100,
                marker='s', linewidths=1, ax=ax)

ax.axhline(max_points, color='y', linewidth=2, label='Max')
ax.axhline(min_points, color='g', linewidth=2, label='Min')
ax.axhline(mean_points, color='r', linewidth=2, label='Mean')

x_pos = range(len(melted['country_of_origin'].unique()))
plt.vlines(x_pos, ymin=70, ymax=90, linestyles='dotted', colors='grey', alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)

ax.set_ylim(70, 90)

ax.set_xlabel('Country of Origin')
ax.set_ylabel('Total Cup Points')
ax.set_title('Total Cup Points by Country of Origin')

ax.legend()

plt.show()

# Or
fig, ax = plt.subplots(figsize=(10, 8))

sns.stripplot(data=table, x='country_of_origin', y='total_cup_points', s=10, c="purple",
              marker='s')

ax.axhline(max_points, color='y', linewidth=2, label='Max')
ax.axhline(min_points, color='r', linewidth=2, label='Min')
ax.axhline(mean_points, color='g', linewidth=2, label='Mean')

x_pos = range(len(melted['country_of_origin'].unique()))
ax.vlines(x_pos, ymin=70, ymax=90, linestyles='dotted', colors='grey', alpha=0.5)
plt.xticks(x_pos, rotation=90, fontsize=8)

plt.ylim(70, 90)

plt.xlabel('Country of Origin')
plt.title('Total Cup Points by Country of Origin')

plt.legend()

plt.show()

# Or
fig, ax = plt.subplots(figsize=(10, 8))

sns.swarmplot(data=table, x='country_of_origin', y='total_cup_points', s=10,
              marker='s')

ax.axhline(max_points, color='y', linewidth=2, label='Max')
ax.axhline(min_points, color='g', linewidth=2, label='Min')
ax.axhline(mean_points, color='r', linewidth=2, label='Mean')

x_pos = range(len(melted['country_of_origin'].unique()))
ax.vlines(x_pos, ymin=70, ymax=90, linestyles='dotted', colors='grey', alpha=0.5)
plt.xticks(x_pos, rotation=90, fontsize=8)

plt.ylim(70, 90)

plt.xlabel('Country of Origin')
plt.title('Total Cup Points by Country of Origin')

plt.legend()

plt.show()

# Or
sns.catplot(data=table, x='country_of_origin', y='total_cup_points', s=100, kind='strip', marker='s')

plt.axhline(max_points, color='y', linewidth=2, label='Max')
plt.axhline(min_points, color='g', linewidth=2, label='Min')
plt.axhline(mean_points, color='r', linewidth=2, label='Mean')

x_pos = range(len(melted['country_of_origin'].unique()))
plt.vlines(x_pos, ymin=70, ymax=90, linestyles='dotted', colors='grey', alpha=0.5)
plt.xticks(x_pos, rotation=90, fontsize=8)

plt.ylim(70, 90)

plt.xlabel('Country of Origin')
plt.title('Total Cup Points by Country of Origin')

plt.legend()

plt.show()

#6. Does altitude significantly affect the quality of coffee?

# 1
coffee_data = coffee_data[coffee_data.altitude_mean_meters != 110000.0]
coffee_data = coffee_data[coffee_data.altitude_mean_meters != 11000.0]
coffee_data = coffee_data[coffee_data.altitude_mean_meters != 190164.0]
coffee_data.loc[coffee_data['altitude_mean_meters'] == '190164.0 ', 'altitude_mean_meters'] = 'NaN'

table = pd.pivot_table(coffee_data, index=['altitude_mean_meters'], values=['total_cup_points'],
                       aggfunc=np.mean)

# Таблица зависимости оценки от высоты
sort_table = table.sort_values(by='total_cup_points', ascending=False)
print(sort_table)

# Проверка нулевой гипотезы: высота оказывает значительное влияние на total_cup_points
high_altitude = coffee_data[coffee_data['altitude_mean_meters'] > coffee_data['altitude_mean_meters'].median()]
low_altitude = coffee_data[coffee_data['altitude_mean_meters'] < coffee_data['altitude_mean_meters'].median()]

t_stat, p_value = ttest_ind(high_altitude['total_cup_points'], low_altitude['total_cup_points'], equal_var=False)

print("High Altitude, mean total_cup_points: ", high_altitude['total_cup_points'].mean())
print("Low Altitude, mean total_cup_points: ", low_altitude['total_cup_points'].mean())
print("t-statistic: ", t_stat)
print("p-value: ", p_value)

if p_value < 0.05:
    print("Reject the null hypothesis: height has a significant effect on total_cup_points")
else:
    print("Failed to reject the null hypothesis: height has no significant effect on total_cup_points.")

melted = table.reset_index().melt(id_vars=['altitude_mean_meters'], var_name='color', value_name='mean_cup_points')

sns.kdeplot(data=melted, x='altitude_mean_meters', y='mean_cup_points',
            cmap='viridis', fill=True, thresh=0, levels=30)

plt.axhline(y=80, color='red', linestyle='--')
plt.axhline(y=85, color='red', linestyle='--')
plt.axvline(x=500, color='red', linestyle='--')
plt.axvline(x=2000, color='red', linestyle='--')

plt.xlim(-100, 4500)
plt.ylim(70, 90)

points = table['total_cup_points']
max_altitude = points.idxmax()

plt.axvline(x=max_altitude, color='g', linestyle='-', label="Max")

plt.title('Visual distribution of rating depending on altitude above sea level')
plt.legend()

plt.show()

''' I got the result: height has a significant effect on total_cup_points.
On the presented graph we see a cluster of points in the range of marks from 80 to 85 
at the level 500-2000 meters. This graph also shows that at an altitude of more than 
2000 meters the estimates are not very special, I would say they tend to be lower than 
those coffee farmlands, which are located lower down the mountain slope.
The best scores from experts, by the way, focused on the mark of 2075 meters. '''

subset = melted[(melted['mean_cup_points'] >= 80) & (melted['mean_cup_points'] <= 85) &
                (melted['altitude_mean_meters'] >= 500) & (melted['altitude_mean_meters'] <= 2000)]

count_subset = len(subset)

count_total = len(melted)

percentage = count_subset / count_total * 100

print("Percentage of points scored total_cup_points ranging from 80 to 85, altitude_mean_meters "
      "at the level of 500-2000 meters: {:.2f}%".format(percentage))

""" Result:
    Percentage of points scored total_cup_points ranging from 80 to 85, 
    altitude_mean_meters at 500-2000 meters: 66.83% """

subset_below_500 = melted[melted['altitude_mean_meters'] < 500]
subset_above_2000 = melted[melted['altitude_mean_meters'] > 2000]
other_subset_500_2000 = melted[(melted['mean_cup_points'] < 80) & (melted['mean_cup_points'] > 85) &
                             (melted['altitude_mean_meters'] >= 500) & (melted['altitude_mean_meters'] <= 2000)]

count_subset_below_500 = len(subset_below_500)
percentage_below_500 = count_subset_below_500 / count_total * 100

count_subset_above_2000 = len(subset_above_2000)
percentage_above_2000 = count_subset_above_2000 / count_total * 100

labels = ['In range', 'Below_500', 'Above_2000', 'Altitude 500-2000 meters, other total_cup_points']
sizes = [percentage, percentage_below_500, percentage_above_2000,
         100 - percentage - percentage_below_500 - percentage_above_2000]
colors = ['blue', 'green', 'yellow', 'red']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)

plt.title(
    'Percentage of points in range "total_cup_points" from 80 to 85,\n '
    '"altitude_mean_meters" at the level 500-2000 meters')
plt.axis('equal')

plt.show()

# Or
labels = ['In range', 'Below_500', 'Above_2000', '500-2000 meters,\n other total_cup_points']
sizes = [percentage, percentage_below_500, percentage_above_2000,
         100 - percentage - percentage_below_500 - percentage_above_2000]
colors = ['blue', 'green', 'yellow', 'red']

fig, ax = plt.subplots()
rects = ax.bar(labels, sizes, color=colors)

plt.title(
    'Percentage of points in range "total_cup_points" from 80 to 85,\n "altitude_mean_meters" at the level 500-2000 meters')

for i, rect in enumerate(rects):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height, '{:.2f}%'.format(sizes[i]), ha='center', va='bottom')

plt.show()

# Или
subset_below_500_t_below_80 = melted[(melted['altitude_mean_meters'] < 500) & (melted['mean_cup_points'] < 80)]
subset_below_500_t_80_85 = melted[
    (melted['altitude_mean_meters'] < 500) & ((melted['mean_cup_points'] >= 80) & (melted['mean_cup_points'] <= 85))]
subset_below_500_t_above_85 = melted[(melted['altitude_mean_meters'] < 500) & (melted['mean_cup_points'] > 85)]

subset_above_2000_t_below_80 = melted[(melted['altitude_mean_meters'] > 2000) & (melted['mean_cup_points'] < 80)]
subset_above_2000_t_80_85 = melted[
    (melted['altitude_mean_meters'] > 2000) & ((melted['mean_cup_points'] >= 80) & (melted['mean_cup_points'] <= 85))]
subset_above_2000_t_above_85 = melted[(melted['altitude_mean_meters'] > 2000) & (melted['mean_cup_points'] > 85)]

subset_500_2000_t_below_80 = melted[
    ((melted['altitude_mean_meters'] >= 500) & (melted['altitude_mean_meters'] <= 2000)) & (
                melted['mean_cup_points'] < 80)]
subset_500_2000_t_above_85 = melted[
    ((melted['altitude_mean_meters'] >= 500) & (melted['altitude_mean_meters'] <= 2000)) & (
                melted['mean_cup_points'] > 85)]
subset_500_2000_t_80_85 = melted[
    ((melted['altitude_mean_meters'] >= 500) & (melted['altitude_mean_meters'] <= 2000)) & (
                (melted['mean_cup_points'] >= 80) & (melted['mean_cup_points'] <= 85))]


def perc_subs(subs):
    count_subs = len(subs)
    percentage_subs = count_subs / len(melted) * 100
    return percentage_subs


counts = np.array([
    [perc_subs(subset_below_500_t_below_80), perc_subs(subset_500_2000_t_below_80),
     perc_subs(subset_above_2000_t_below_80)],
    [perc_subs(subset_below_500_t_80_85), perc_subs(subset_500_2000_t_80_85), perc_subs(subset_above_2000_t_80_85)],
    [perc_subs(subset_below_500_t_above_85), perc_subs(subset_500_2000_t_above_85),
     perc_subs(subset_above_2000_t_above_85)]
])

x_labels = ['< 500 meters', '500-2000 meters', '> 2000 meters']
y_labels = ['< 80 points', '80-85 points', '> 85 points']

fig, ax = plt.subplots()
im = ax.imshow(counts, cmap='YlGn')

ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        ax.text(j, i, '{:.2f}%'.format(counts[i, j]), ha='center', va='center', color='black')

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('% of samples', rotation=-90, va="bottom")

plt.title(
    'Percentage of points in range "total_cup_points" from 80 to 85,\n "altitude_mean_meters" at the level 500-2000 meters')

plt.show()

# Or
table = pd.pivot_table(coffee_data, index=['altitude_mean_meters'], values=['total_cup_points'],
                       aggfunc=np.mean)

melted = table.reset_index().melt(id_vars=['altitude_mean_meters'], var_name='color', value_name='mean_cup_points')

x = melted['altitude_mean_meters']
y = melted['mean_cup_points']

plt.hexbin(x, y, gridsize=30, cmap='viridis')
plt.colorbar()

plt.axhline(y=80, color='red', linestyle='--')
plt.axhline(y=85, color='red', linestyle='--')
plt.axvline(x=500, color='red', linestyle='--')
plt.axvline(x=2000, color='red', linestyle='--')

plt.xlim(-100, 4500)
plt.ylim(70, 90)

plt.title('Visual distribution of rating depending on altitude above sea level')

plt.show()

# 2
# Table of rating depending on height
sort_table = table.sort_values(by='total_cup_points', ascending=False)
print(sort_table)

sns.histplot(data=coffee_data, x='altitude_mean_meters', y='total_cup_points',
            color='blue')

plt.axhline(y=80, color='red', linestyle='--')
plt.axhline(y=85, color='red', linestyle='--')
plt.axvline(x=500, color='red', linestyle='--')
plt.axvline(x=2000, color='red', linestyle='--')

plt.xlim(-100, 4500)
plt.ylim(70, 90)

points = table['total_cup_points']
max_altitude = points.idxmax()

plt.axvline(x=coffee_data['altitude_mean_meters'].tolist().index(max_altitude),
            color='g', linestyle='-', label="Max")

plt.title('Visual distribution of rating depending on altitude above sea level')
plt.legend()

plt.show()

# Or
fig, ax = plt.subplots()

sns.scatterplot(data=table, x='altitude_mean_meters', y='total_cup_points', alpha=0.3, ax=ax)

ax.axhline(y=80, color='red', linestyle='--')
ax.axhline(y=85, color='red', linestyle='--')
ax.axvline(x=500, color='red', linestyle='--')
ax.axvline(x=2000, color='red', linestyle='--')

ax.set_xlim(-100, 4500)
ax.set_ylim(70, 90)

points = sort_table['total_cup_points']
max_altitude = points.idxmax()

ax.axvline(x=max_altitude, color='g', linestyle='-', label="Max")

ax.set(title='Visual distribution of rating depending on altitude above sea level')
ax.legend()

plt.show()

# Or
sns.relplot(data=table, x='altitude_mean_meters', y='total_cup_points', hue='total_cup_points',
            kind='scatter', palette='YlGnBu', legend=False)

plt.axhline(y=80, color='red', linestyle='--')
plt.axhline(y=85, color='red', linestyle='--')
plt.axvline(x=500, color='red', linestyle='--')
plt.axvline(x=2000, color='red', linestyle='--')

plt.xlim(-100, 4500)
plt.ylim(70, 90)

points = sort_table['total_cup_points']
max_altitude = points.idxmax()

plt.axvline(x=max_altitude, color='g', linestyle='-', label="Max")

plt.title('Visual distribution of rating depending on altitude above sea level')
plt.legend()

plt.show()

# Or
sns.lmplot(data=melted, x='altitude_mean_meters', y='mean_cup_points',
           height=5, aspect=2, scatter_kws={'alpha': 0.3}, fit_reg=False)

plt.axhline(y=80, color='red', linestyle='--')
plt.axhline(y=85, color='red', linestyle='--')
plt.axvline(x=500, color='red', linestyle='--')
plt.axvline(x=2000, color='red', linestyle='--')

plt.xlim(-100, 4500)
plt.ylim(70, 90)

points = sort_table['total_cup_points']
max_altitude = points.idxmax()

plt.axvline(x=max_altitude, color='g', linestyle='-', label="Max")

plt.title('Visual distribution of rating depending on altitude above sea level')
plt.legend()

plt.show()

# Or
fig, ax = plt.subplots()

ax.scatter(table.index, table.values, alpha=0.3)

ax.axhline(y=80, color='red', linestyle='--')
ax.axhline(y=85, color='red', linestyle='--')
ax.axvline(x=500, color='red', linestyle='--')
ax.axvline(x=2000, color='red', linestyle='--')

ax.set_xlim(-100, 4500)
ax.set_ylim(70, 90)

points = sort_table['total_cup_points']
max_altitude = points.idxmax()

ax.axvline(x=max_altitude, color='g', linestyle='-', label="Max")

ax.legend()
ax.set(title='Visual distribution of rating depending on altitude above sea level')

plt.show()

# Or
fig, ax = plt.subplots()

ax.plot(table.index, table.values, alpha=0.3)

ax.axhline(y=80, color='red', linestyle='--')
ax.axhline(y=85, color='red', linestyle='--')
ax.axvline(x=500, color='red', linestyle='--')
ax.axvline(x=2000, color='red', linestyle='--')

ax.set_xlim(-100, 4500)
ax.set_ylim(70, 90)

points = sort_table['total_cup_points']
max_altitude = points.idxmax()

ax.axvline(x=max_altitude, color='g', linestyle='-', label="Max")

ax.legend()
ax.set(title='Visual distribution of rating depending on altitude above sea level')

plt.show()

# 3
fig, ax = plt.subplots()

ax.bar(sort_table.index.astype(int), sort_table['total_cup_points'], alpha=0.7)

ax.axhline(y=80, color='red', linestyle='--')
ax.axhline(y=85, color='red', linestyle='--')
ax.axvline(x=500, color='red', linestyle='--')
ax.axvline(x=2000, color='red', linestyle='--')

ax.set_xlim(-100, 4500)
ax.set_ylim(70, 90)

max_altitude = sort_table['total_cup_points'].idxmax()

ax.axvline(x=max_altitude, color='g', linestyle='-', label="Max")

ax.legend()
ax.set(title='Visual distribution of rating depending on altitude above sea level',
       xlabel='Altitude above sea level, m', ylabel='Average coffee cup rating')

plt.show()

conn.close()
