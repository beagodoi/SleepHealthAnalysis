from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

sns.set_style("white")
sns.set_context("notebook")


# Goals:

    # Use random forrest to predict whos likely to have spleeping disorders

    # Check if certain occupations associated with higher stress or poorer sleep? 

    # Check if there is a non-linear relationship between sleep duration and sleep quality? 

    # Check if stress levels and sleep disorders related? 

    # Check if BMI affect sleep patterns? 

    # Check if there a common patters in sleeping across the different age groups?


#Updating dataset
api = KaggleApi()
api.authenticate()

dataset = "uom190346a/sleep-health-and-lifestyle-dataset"
api.dataset_download_files(dataset, path="./Lifestyle_Analysis", unzip=True)

df = pd.read_csv("Lifestyle_Analysis/Sleep_health_and_lifestyle_dataset.csv")


# Dataset info
print(df.info())

nan_cols = df.isna().sum()
print(nan_cols)


# Removing colmuns that won't be used
df = df[['Gender','Age','Occupation','Sleep Duration','Quality of Sleep',
         'Physical Activity Level','Stress Level','BMI Category','Sleep Disorder']]


# Filling missing values with prediction models
    # Converting categorical to numeric variables
df['Has disorder'] = df['Sleep Disorder'].apply(lambda x: 0 if x == "None" else (1 if x == "Insomnia" else 2))

    # Creating training and testing dataframes
train_df = df[df['Sleep Disorder'].notna()]

test_df = df[df['Sleep Disorder'].isna()]

    #Train classifiers
features = ['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'BMI Category']

df_encoded = train_df.copy()
df_encoded['BMI Category'] = LabelEncoder().fit_transform(df_encoded['BMI Category'])

X = df_encoded[features]
y = df_encoded['Has disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Random Forest
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
print("Random Forest:")
print(classification_report(y_test, classifier.predict(X_test)))


# Prediciton model
test_encoded = test_df.copy()
test_encoded['BMI Category'] = LabelEncoder().fit_transform(test_encoded['BMI Category'])

test_preds = classifier.predict(test_encoded[features])
df.loc[df['Sleep Disorder'].isna(), 'Sleep Disorder'] = ['Insomnia' if pred == 1 else ('Apnea' if pred == 2 else 'None') for pred in test_preds]


# Checking for nan_values
check_nan_cols = df.isna().sum()
print(check_nan_cols)


# Descriptive Analysis
print(df.iloc[:,:-1].describe().T)

    # Bar plot: Distributions of gender, occupation, sleep disorders and BMI categories
fig, axes = plt.subplots(1, 2, figsize=(8, 6))

variaveis = ['Gender', 'Occupation']

for i, var in enumerate(variaveis):
    sns.countplot(data=df, x=var, hue=var, palette='mako', width=0.3, ax=axes[i])
    axes[i].set_title(f'Distribution of {var}')
    axes[1].tick_params(axis='x', rotation=90)

    for ax in axes:
        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
        ax.set_ylabel('Total', fontsize=10)
        ax.set_title(ax.get_title(), fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

        for container in ax.containers:
            ax.bar_label(container, padding=2, fontsize=10)

plt.tight_layout(h_pad=2)   
plt.show()

fig_2, axes_2 = plt.subplots(1, 2, figsize=(8, 6))

variaveis_2 = ['Sleep Disorder', 'BMI Category']

for i_2, var_2 in enumerate(variaveis_2):
    sns.countplot(data=df, x=var_2, hue=var_2, palette='mako', width=0.3, ax=axes_2[i_2])
    axes_2[i_2].set_title(f'Distribution of {var_2}')

    for ax_2 in axes_2:
        ax_2.set_xlabel(ax_2.get_xlabel(), fontsize=10)
        ax_2.set_ylabel('Total', fontsize=10)
        ax_2.set_title(ax_2.get_title(), fontsize=12)
        ax_2.tick_params(axis='both', labelsize=10)
        if ax_2.get_legend() is not None:
            ax_2.get_legend().remove()

        for container_2 in ax_2.containers:
            ax_2.bar_label(container_2, padding=2, fontsize=10)

plt.tight_layout(h_pad=2)   
plt.show()

    # Histogram: Frequency of age, sleep duration, and physical activity level
fig_3, axes_3 = plt.subplots(1, 3, figsize=(10, 5))

variaveis_3 = ['Age', 'Sleep Duration', 'Physical Activity Level']

for i_3, var_3 in enumerate(variaveis_3):
    sns.histplot(df[var_3], kde=True, color='darkblue', ax=axes_3[i_3])
    axes_3[i_3].set_title(f'Distribution of {var_3}')

    for ax_3 in axes_3:
        ax_3.set_xlabel(ax_3.get_xlabel(), fontsize=10)
        ax_3.set_ylabel('Frequency', fontsize=10)
        ax_3.set_title(ax_3.get_title(), fontsize=12)
        ax_3.tick_params(axis='both', labelsize=10)

plt.tight_layout(h_pad=2)    
plt.show()


# Group Comparisons
    # Boxplot: Sleep quality and stress levels by ocupation
variaveis_4 = ['Quality of Sleep', 'Stress Level']

for var_4 in variaveis_4:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Occupation', y=var_4, data=df, hue='Occupation', palette='mako')
    plt.title(f'{var_4} across ocupations', fontsize=12)
    plt.xlabel('Occupations',fontsize=10)
    plt.ylabel(var_4,fontsize=10)
    plt.tick_params(axis='x', rotation=90)
    plt.tight_layout(h_pad=2)
    plt.show()

    # Boxplot: Sleep quality, duration and sleep disorders by BMI category
variaveis_5 = ['Quality of Sleep', 'Sleep Duration', 'Sleep Disorder']

for var_5 in variaveis_5:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='BMI Category', y=var_5, data=df, hue='BMI Category', palette='mako')
    plt.title(f'{var_5} across BMI categories', fontsize=12)
    plt.xlabel('BMI Category',fontsize=10)
    plt.ylabel(var_5,fontsize=10)
    plt.tight_layout(h_pad=2)
    plt.show()


# Correlations
    #Scatterplot: Sleep duration x Sleep quality
sns.lmplot(x='Sleep Duration', y='Quality of Sleep', data=df, palette='mako', height=5, aspect=1.5)
plt.title('Correlation between sleep duration and quality of sleep')
plt.xlabel('Sleep duration')
plt.ylabel('Quality of sleep')
plt.tight_layout()
plt.show()

    #Violin plot: Stress level x Sleep disorder
sns.violinplot(data=df, x='Sleep Disorder', y='Stress Level', hue='Sleep Disorder', palette='mako')
plt.title('Relationship between sleep disorder and stress level')
plt.xlabel('Sleep disorder')
plt.ylabel('Stress level')
plt.tight_layout()
plt.show()

    #Heatmap: Mean of sleep by age group and quality of sleep
df['Age bins'] = pd.cut(df['Age'], bins=[0,10,20,30,40,50,60], labels=["0-10","10-20","20-30","30-40","40-50","50-60"])

heat_df = df.groupby(['Age bins','Quality of Sleep'])['Sleep Duration'].mean().unstack()
sns.heatmap(heat_df, cmap='Blues', annot=True)
plt.title('Mean of sleep duration by age group and quality of sleep')
plt.ylabel('Age group')
plt.xlabel('Sleep duration')
plt.tight_layout()
plt.show()
