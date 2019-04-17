
## Final Project Submission

Please fill out:
* Student name: Matthew Sparr
* Student pace: self paced 
* Scheduled project review date/time: 
* Instructor name: Eli 
* Blog post URL:

## Introduction

For this project I chose a Kaggle dataset for an ongoing competition that can be found at https://www.kaggle.com/c/petfinder-adoption-prediction. This competition involves predicting the speed of adoption for a pet adoption site in Malaysia. Provided are various data fields such as the color of the pet, the age, and the breed. 

Also provided are image data on uploaded photos of the pets that was ran through Google's Vision API and sentiment data that was ran through Google's Natural Language API, on the description given for the pets. 

The goal of this project is to acheieve a decent score on the Kaggle competition test data and hopefully place high on the leaderboard.

![title](petfinder.png)

## Import libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,VotingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import os
import json
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
```

## Grab train and test set

The test set provided does not include the 'AdoptionSpeed' target variable and is only used to make predictions to submit to Kaggle for scoring.


```python
train = pd.read_csv('train/train.csv')
test = pd.read_csv('test/test.csv')
```

## Fill missing values

The 'Name' and 'Description' columns are the only two columns will missing data for both the train and test set. Since both fields are text, they will be filled with a blank space, ' '.


```python
train.isna().sum()
```




    Type                0
    Name             1257
    Age                 0
    Breed1              0
    Breed2              0
    Gender              0
    Color1              0
    Color2              0
    Color3              0
    MaturitySize        0
    FurLength           0
    Vaccinated          0
    Dewormed            0
    Sterilized          0
    Health              0
    Quantity            0
    Fee                 0
    State               0
    RescuerID           0
    VideoAmt            0
    Description        12
    PetID               0
    PhotoAmt            0
    AdoptionSpeed       0
    dtype: int64




```python
test.isna().sum()
```




    Type              0
    Name            303
    Age               0
    Breed1            0
    Breed2            0
    Gender            0
    Color1            0
    Color2            0
    Color3            0
    MaturitySize      0
    FurLength         0
    Vaccinated        0
    Dewormed          0
    Sterilized        0
    Health            0
    Quantity          0
    Fee               0
    State             0
    RescuerID         0
    VideoAmt          0
    Description       2
    PetID             0
    PhotoAmt          0
    dtype: int64




```python
train.Name.fillna(' ', inplace=True)
train.Description.fillna(' ', inplace=True)

test.Name.fillna(' ', inplace=True)
test.Description.fillna(' ', inplace=True)
```

## Explore variables

Below is a basic histogram of all of the variables.


```python
train.hist(figsize=(20,20))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000022794D508D0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000022792C089B0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000022795235128>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002279363C278>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000022799173390>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000022799173748>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227991DAB38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227938B14E0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000227D6777DD8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D64E5128>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D6505438>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D63A5780>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000227D63C7A58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D63C1D68>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D63B30B8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D656D3C8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000227D659C6D8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D65909E8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D6469CF8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000227D6470048>]],
          dtype=object)




![png](final_files/final_16_1.png)


<b>Most of the variables do not have a normal distribution which means we will probably want to standardize them later on. The target variable 'AdoptionSpeed' has a low count of '0' values which could negatively impact training a classifier on the training set.

We can also see that most pets have only one breed and one color as there are many zero values for 'Breed2', 'Color2', and 'Color3'.

<b>Now we can look at some of the value counts of various columns just to get a feel of the distribution of the pets.

## Are dogs or cats more common?


```python
train['Type'].value_counts().rename({1:'Dog',
                                        2:'Cat'}).plot(kind='barh',
                                                       figsize=(15,6))
plt.yticks(fontsize='xx-large')
plt.title('Type Distribution', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Type Distribution')




![png](final_files/final_20_1.png)


<b>Slightly more dogs than cats.

## Do dogs and cats have different adoption rates on average?


```python
train['AdoptionSpeed'][train['Type'] == 1].value_counts().sort_index().plot(kind='barh',
                                                       figsize=(15,6))
plt.yticks(fontsize='xx-large')
plt.ylabel('Adoption Speed')
plt.title('Adoption Speed Distribution (Dogs)', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Adoption Speed Distribution (Dogs)')




![png](final_files/final_23_1.png)



```python
train['AdoptionSpeed'][train['Type'] == 2].value_counts().sort_index().plot(kind='barh',
                                                       figsize=(15,6))
plt.yticks(fontsize='xx-large')
plt.ylabel('Adoption Speed')
plt.title('Adoption Speed Distribution (Cats)', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Adoption Speed Distribution (Cats)')




![png](final_files/final_24_1.png)



```python
pd.DataFrame([train['AdoptionSpeed'][train['Type'] == 1].mean(),train['AdoptionSpeed'][train['Type'] == 2].mean()]).rename({0:'Dogs',
                                        1:'Cats'}).plot(kind='barh',
                                                       figsize=(15,6), legend=None)
plt.yticks(fontsize='xx-large')
plt.xlabel('Adoption Speed')
plt.title('Average Adoption Speed', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Average Adoption Speed')




![png](final_files/final_25_1.png)


<b> The largest number of dogs aren't adopted after 100 days of being listed whereas the largest number of cats are adopted in the first month of being listed. Dogs on average take a longer amount of time to be adopted than cats.

## What breeds are most common?


```python
train['Breed1'].value_counts().head(10).plot(kind='barh',
                                                       figsize=(15,6))
plt.yticks(fontsize='xx-large')
plt.ylabel('Breed ID')
plt.title('Breed Distribution (Breed1)', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Breed Distribution (Breed1)')




![png](final_files/final_28_1.png)


<b> Breed 307 which signifies an unknown breed is the most common primary breed followed by Breed 266 which are domestic shorthair cats.


```python
train['Breed2'].value_counts().head(10).plot(kind='barh',
                                                       figsize=(15,6))
plt.yticks(fontsize='xx-large')
plt.ylabel('Breed ID')
plt.title('Breed Distribution(Breed2)', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Breed Distribution(Breed2)')




![png](final_files/final_30_1.png)


<b> Most pets do not have a second breed but the largest number of the ones that do have an unknown second breed.

## More male or female pets?


```python
train['Gender'][(train['Gender'] == 1) | (train['Gender'] == 2)].value_counts().rename({1:'Male',
                                        2:'Female'}).plot(kind='barh',
                                                       figsize=(15,6))
plt.yticks(fontsize='xx-large')
plt.title('Gender Distribution (excluding groups of pets)', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Gender Distribution (excluding groups of pets)')




![png](final_files/final_33_1.png)


<b> More pets are female.

## 


```python
train['PhotoAmt'].value_counts().sort_index().plot(kind='barh',
                                                       figsize=(20,15))
plt.yticks(fontsize='xx-large')
plt.ylabel('Number of Photos')
plt.title('Amount of Photos Distribution', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Amount of Photos Distribution')




![png](final_files/final_36_1.png)


<b>Most listings that have photos only have 1-5 of them.

## Add image metadata

<b>The image metadata is given by a collection of JSON files with the 'PetID' of the corresponding pet in the name of the file. Some pets have multiple pictures but I will initially just use the first three photos of each pet if available as these are likely the main photos seen by people searching for pets to adopt and thus have the largest effect on drawing in a perspective adoption.
  


```python
for index, row in train.iterrows():  ## First photo
    file = 'train_metadata/' + row['PetID'] + '-1.json'
    if os.path.exists(file):
        data = json.load(open(file, encoding="utf8"))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        train.loc[index, 'vertex_x']= vertex_x
        train.loc[index, 'vertex_y']= vertex_y
        train.loc[index, 'bounding_conf']= bounding_confidence
        train.loc[index, 'bounding_imp']= bounding_importance_frac
        train.loc[index, 'dom_blue']= dominant_blue
        train.loc[index, 'dom_green']= dominant_green
        train.loc[index, 'dom_red']= dominant_red
        train.loc[index, 'pixel_frac']= dominant_pixel_frac
        train.loc[index, 'score']= dominant_score
    else:
        train.loc[index, 'vertex_x']= -1
        train.loc[index, 'vertex_y']= -1
        train.loc[index, 'bounding_conf']= -1
        train.loc[index, 'bounding_imp']= -1
        train.loc[index, 'dom_blue']= -1
        train.loc[index, 'dom_green']= -1
        train.loc[index, 'dom_red']= -1
        train.loc[index, 'pixel_frac']= -1
        train.loc[index, 'score']= -1
```


```python
for index, row in train.iterrows():  ## Second photo
    file = 'train_metadata/' + row['PetID'] + '-2.json'
    if os.path.exists(file):
        data = json.load(open(file, encoding="utf8"))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        try:
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        except:
            dominant_blue = -1
        try:
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        except: 
            dominant_green = -1
        try: 
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        except:
            dominant_red = -1
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        train.loc[index, 'vertex_x2']= vertex_x
        train.loc[index, 'vertex_y2']= vertex_y
        train.loc[index, 'bounding_conf2']= bounding_confidence
        train.loc[index, 'bounding_imp2']= bounding_importance_frac
        train.loc[index, 'dom_blue2']= dominant_blue
        train.loc[index, 'dom_green2']= dominant_green
        train.loc[index, 'dom_red2']= dominant_red
        train.loc[index, 'pixel_frac2']= dominant_pixel_frac
        train.loc[index, 'score2']= dominant_score
    else:
        train.loc[index, 'vertex_x2']= -1
        train.loc[index, 'vertex_y2']= -1
        train.loc[index, 'bounding_conf2']= -1
        train.loc[index, 'bounding_imp2']= -1
        train.loc[index, 'dom_blue2']= -1
        train.loc[index, 'dom_green2']= -1
        train.loc[index, 'dom_red2']= -1
        train.loc[index, 'pixel_frac2']= -1
        train.loc[index, 'score2']= -1
```


```python
for index, row in train.iterrows():  ## Third photo
    file = 'train_metadata/' + row['PetID'] + '-3.json'
    if os.path.exists(file):
        data = json.load(open(file, encoding="utf8"))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        try:
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        except:
            dominant_blue = -1
        try:
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        except: 
            dominant_green = -1
        try: 
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        except:
            dominant_red = -1
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        train.loc[index, 'vertex_x3']= vertex_x
        train.loc[index, 'vertex_y3']= vertex_y
        train.loc[index, 'bounding_conf3']= bounding_confidence
        train.loc[index, 'bounding_imp3']= bounding_importance_frac
        train.loc[index, 'dom_blue3']= dominant_blue
        train.loc[index, 'dom_green3']= dominant_green
        train.loc[index, 'dom_red3']= dominant_red
        train.loc[index, 'pixel_frac3']= dominant_pixel_frac
        train.loc[index, 'score3']= dominant_score
    else:
        train.loc[index, 'vertex_x3']= -1
        train.loc[index, 'vertex_y3']= -1
        train.loc[index, 'bounding_conf3']= -1
        train.loc[index, 'bounding_imp3']= -1
        train.loc[index, 'dom_blue3']= -1
        train.loc[index, 'dom_green3']= -1
        train.loc[index, 'dom_red3']= -1
        train.loc[index, 'pixel_frac3']= -1
        train.loc[index, 'score3']= -1
```


```python
for index, row in test.iterrows():  # First photo
    file = 'test_metadata/' + row['PetID'] + '-1.json'
    if os.path.exists(file):
        data = json.load(open(file, encoding="utf8"))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        test.loc[index, 'vertex_x']= vertex_x
        test.loc[index, 'vertex_y']= vertex_y
        test.loc[index, 'bounding_conf']= bounding_confidence
        test.loc[index, 'bounding_imp']= bounding_importance_frac
        test.loc[index, 'dom_blue']= dominant_blue
        test.loc[index, 'dom_green']= dominant_green
        test.loc[index, 'dom_red']= dominant_red
        test.loc[index, 'pixel_frac']= dominant_pixel_frac
        test.loc[index, 'score']= dominant_score
    else:
        test.loc[index, 'vertex_x']= -1
        test.loc[index, 'vertex_y']= -1
        test.loc[index, 'bounding_conf']= -1
        test.loc[index, 'bounding_imp']= -1
        test.loc[index, 'dom_blue']= -1
        test.loc[index, 'dom_green']= -1
        test.loc[index, 'dom_red']= -1
        test.loc[index, 'pixel_frac']= -1
        test.loc[index, 'score']= -1
```


```python
for index, row in test.iterrows(): # Second photo
    file = 'test_metadata/' + row['PetID'] + '-2.json'
    if os.path.exists(file):
        data = json.load(open(file, encoding="utf8"))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        try:
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        except:
            dominant_blue = -1
        try:
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        except: 
            dominant_green = -1
        try: 
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        except:
            dominant_red = -1
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        test.loc[index, 'vertex_x2']= vertex_x
        test.loc[index, 'vertex_y2']= vertex_y
        test.loc[index, 'bounding_conf2']= bounding_confidence
        test.loc[index, 'bounding_imp2']= bounding_importance_frac
        test.loc[index, 'dom_blue2']= dominant_blue
        test.loc[index, 'dom_green2']= dominant_green
        test.loc[index, 'dom_red2']= dominant_red
        test.loc[index, 'pixel_frac2']= dominant_pixel_frac
        test.loc[index, 'score2']= dominant_score
    else:
        test.loc[index, 'vertex_x2']= -1
        test.loc[index, 'vertex_y2']= -1
        test.loc[index, 'bounding_conf2']= -1
        test.loc[index, 'bounding_imp2']= -1
        test.loc[index, 'dom_blue2']= -1
        test.loc[index, 'dom_green2']= -1
        test.loc[index, 'dom_red2']= -1
        test.loc[index, 'pixel_frac2']= -1
        test.loc[index, 'score2']= -1
```


```python
for index, row in test.iterrows(): # Third photo
    file = 'test_metadata/' + row['PetID'] + '-3.json'
    if os.path.exists(file):
        data = json.load(open(file, encoding="utf8"))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        try:
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        except:
            dominant_blue = -1
        try:
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        except: 
            dominant_green = -1
        try: 
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        except:
            dominant_red = -1
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        test.loc[index, 'vertex_x3']= vertex_x
        test.loc[index, 'vertex_y3']= vertex_y
        test.loc[index, 'bounding_conf3']= bounding_confidence
        test.loc[index, 'bounding_imp3']= bounding_importance_frac
        test.loc[index, 'dom_blue3']= dominant_blue
        test.loc[index, 'dom_green3']= dominant_green
        test.loc[index, 'dom_red3']= dominant_red
        test.loc[index, 'pixel_frac3']= dominant_pixel_frac
        test.loc[index, 'score3']= dominant_score
    else:
        test.loc[index, 'vertex_x3']= -1
        test.loc[index, 'vertex_y3']= -1
        test.loc[index, 'bounding_conf3']= -1
        test.loc[index, 'bounding_imp3']= -1
        test.loc[index, 'dom_blue3']= -1
        test.loc[index, 'dom_green3']= -1
        test.loc[index, 'dom_red3']= -1
        test.loc[index, 'pixel_frac3']= -1
        test.loc[index, 'score3']= -1
```

## Add sentiment data

<b> The sentiment data, similar to the image data, is provided as JSON files with the 'PetID' of the corresponding pet as the file name.
    The relevant values I chose to include from the sentiment data are magnitude and score.


```python
for index, row in train.iterrows():
    file = 'train_sentiment/' + row['PetID'] + '.json'
    if os.path.exists(file):
        data = json.load(open(file, encoding="utf8"))
        mag = data['documentSentiment']['magnitude']
        score = data['documentSentiment']['score']     
        train.loc[index, 'magnitude']= mag
        train.loc[index, 'sentiment_score']= score
    else:
        train.loc[index, 'magnitude']= -1
        train.loc[index, 'sentiment_score']= -1
```


```python
for index, row in test.iterrows():
    file = 'test_sentiment/' + row['PetID'] + '.json'
    if os.path.exists(file):
        data = json.load(open(file, encoding="utf8"))
        mag = data['documentSentiment']['magnitude']
        score = data['documentSentiment']['score']     
        test.loc[index, 'magnitude']= mag
        test.loc[index, 'sentiment_score']= score
    else:
        test.loc[index, 'magnitude']= -1
        test.loc[index, 'sentiment_score']= -1
```

## Save data before adding addtional columns

<b> I will be adding additional columns of data but wanted to save a copy of the train and test sets to compare with later on.


```python
train.to_csv('pre_train.csv')
test.to_csv('pre_test.csv')
```

## Add name and description length

<b> To include a bit more data on 'Description' column and the otherwise unused 'Name' column, I decided to include the length of each as new columns of data. 


```python
train['NameLength'] = train['Name'].map(lambda x: len(str(x))).astype('int')
train['DescLength'] = train['Description'].map(lambda x: len(str(x))).astype('int')
test['NameLength'] = test['Name'].map(lambda x: len(str(x))).astype('int')
test['DescLength'] = test['Description'].map(lambda x: len(str(x))).astype('int')
```


```python
pd.DataFrame([train['DescLength'][train['AdoptionSpeed'] == 0].mean(),
              train['DescLength'][train['AdoptionSpeed'] == 1].mean(),
              train['DescLength'][train['AdoptionSpeed'] == 2].mean(),
              train['DescLength'][train['AdoptionSpeed'] == 3].mean(),
              train['DescLength'][train['AdoptionSpeed'] == 4].mean()]).plot(kind='barh',figsize=(16,5))
plt.yticks(fontsize='xx-large')
plt.ylabel('Adoption Speed')
plt.xlabel('Description Length')
plt.title('Average Description Length', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Average Description Length')




![png](final_files/final_56_1.png)


<b> There average description length trends upward as the adoption speed window increases until it hits level 4, where the average description length then is lower.

## Add dog data

<b> Using data from an AKC website as well as Wikipedia, I assigned a breed group to each dog breed as I suspect that there is a difference in adoptability amongst the dog breed groups. I added these breed groups in Microsoft Excel and generated a csv file 'dog_breeds' using the providing csv of breed labels. Now I just have to add a new column 'Group' to the train and test set. Since this only works for dogs, any cats will just be assigned the group 'Cat'.


```python
dog_data = pd.read_csv('dog_breeds.csv')
```


```python
dog_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>BreedID</th>
      <th>Type</th>
      <th>BreedName</th>
      <th>Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Affenpinscher</td>
      <td>Toy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>Afghan Hound</td>
      <td>Hound</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>Airedale Terrier</td>
      <td>Terrier</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>Akbash</td>
      <td>Working</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>Akita</td>
      <td>Working</td>
    </tr>
  </tbody>
</table>
</div>




```python
for index, row in train.iterrows():
    for i, r in dog_data.iterrows():
        if row['Breed1'] == r['BreedID']:
            train.at[index,'Group'] = r['Group']
            break
```


```python
for index, row in test.iterrows():
    for i, r in dog_data.iterrows():
        if row['Breed1'] == r['BreedID']:
            test.at[index,'Group'] = r['Group']
            break
```


```python
train.Group.isna().sum()
```




    6853




```python
dog_data.columns
```




    Index(['Unnamed: 0', 'BreedID', 'Type', 'BreedName', 'Group'], dtype='object')




```python
for index, row in train.iterrows():
    try:
        breed = row['Breed1']
        group = dog_data[dog_data['BreedID'] == breed]['Group'].values[0]
    except:
        group = 'Cat'
    train.loc[index,'Group'] = group    
```


```python
for index, row in test.iterrows():
    try:
        breed = row['Breed1']
        group = dog_data[dog_data['BreedID'] == breed]['Group'].values[0]
    except:
        group = 'Cat'
    test.loc[index,'Group'] = group 
```


```python
train['Group'][train['Group'] != 'Cat'].value_counts().sort_index().plot(kind='barh',
                                                       figsize=(20,15))
plt.yticks(fontsize='xx-large')
plt.title('Distribution of Dog Groups', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Distribution of Dog Groups')




![png](final_files/final_68_1.png)


<b> It seems that 'Misc' is by far the most common group assigned to the dogs. 


```python
train['Group'][(train['Group'] != 'Cat') & (train['Group'] != 'Misc')].value_counts().sort_index().plot(kind='barh',
                                                       figsize=(20,15))
plt.yticks(fontsize='xx-large')
plt.title('Distribution of Dog Groups', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Distribution of Dog Groups')




![png](final_files/final_70_1.png)


<b> Removing the 'Misc' group we can see the distribution of the other groups much better. From this, 'Sporting' and 'Toy' are the most common with 'Hunting' being the least common.

## Add cat data

<b> Using data from http://www.catbreedslist.com, I decided to include two new variables for the 'Cats' in the dataset. The first is 'Hypo' which is whether or not the cat breed is hypoallergenic. The second is 'Cute' which if the value in this column is 1 then that cat breed is one of the top 10 cutest cat breeds. 


```python
cat_data = pd.read_csv('cat_info.csv')
```


```python
cat_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BreedID</th>
      <th>Type</th>
      <th>BreedName</th>
      <th>Cute</th>
      <th>Hypo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241</td>
      <td>2</td>
      <td>Abyssinian</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>242</td>
      <td>2</td>
      <td>American Curl</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>243</td>
      <td>2</td>
      <td>American Shorthair</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244</td>
      <td>2</td>
      <td>American Wirehair</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>245</td>
      <td>2</td>
      <td>Applehead Siamese</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
for index, row in train.iterrows():
    try:
        breed = row['Breed1']
        cute = cat_data[cat_data['BreedID'] == breed]['Cute'].values[0]
        hypo = cat_data[cat_data['BreedID'] == breed]['Hypo'].values[0]
    except:
        cute = -1
        hypo = -1
    train.loc[index,'Cat_Cute'] = cute
    train.loc[index,'Cat_Hypo'] = hypo
```


```python
for index, row in test.iterrows():
    try:
        breed = row['Breed1']
        cute = cat_data[cat_data['BreedID'] == breed]['Cute'].values[0]
        hypo = cat_data[cat_data['BreedID'] == breed]['Hypo'].values[0]
    except:
        cute = -1
        hypo = -1
    test.loc[index,'Cat_Cute'] = cute
    test.loc[index,'Cat_Hypo'] = hypo
```


```python
pd.DataFrame([train['AdoptionSpeed'][train['Cat_Hypo'] == 0].mean(),train['AdoptionSpeed'][train['Cat_Hypo'] == 1].mean()]).rename({1:'Hypoallergenic', 0:'Non-hypoallergenic'}).plot(kind='barh',figsize=(16,5))
plt.yticks(fontsize='xx-large')
plt.title('Hypoallergenic Adoption Speeds', fontsize='xx-large')
```




    Text(0.5, 1.0, 'Hypoallergenic Adoption Speeds')




![png](final_files/final_78_1.png)


<b> It seems that hypoallergenic cat breeds are adopted more quickly on average than non-hypoallergenic cat breeds.

## Add state data

<b> Using census data found on Wikipedia for the states in Malaysia, I added the population, percentage of urban environment, and population density for each state.
 


```python
state_data = pd.read_csv('state_data.csv')
state_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Population</th>
      <th>StateID</th>
      <th>UrbanPercent</th>
      <th>PopDensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kuala Lumpur</td>
      <td>1627172</td>
      <td>41401</td>
      <td>100.0</td>
      <td>6891</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Labuan</td>
      <td>86908</td>
      <td>41415</td>
      <td>82.3</td>
      <td>950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Johor</td>
      <td>3348283</td>
      <td>41336</td>
      <td>71.9</td>
      <td>174</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kedah</td>
      <td>1890098</td>
      <td>41325</td>
      <td>64.6</td>
      <td>199</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kelantan</td>
      <td>1459994</td>
      <td>41367</td>
      <td>42.4</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
</div>




```python
for index, row in train.iterrows():
    state = row['State']
    urban = state_data[state_data['StateID'] == state]['UrbanPercent'].values[0]
    pop = state_data[state_data['StateID'] == state]['Population'].values[0]
    pop_den = state_data[state_data['StateID'] == state]['PopDensity'].values[0]

    train.loc[index,'UrbanPercent'] = urban
    train.loc[index,'Population'] = pop
    train.loc[index,'PopDensity'] = pop_den
```


```python
for index, row in test.iterrows():
    state = row['State']
    urban = state_data[state_data['StateID'] == state]['UrbanPercent'].values[0]
    pop = state_data[state_data['StateID'] == state]['Population'].values[0]
    pop_den = state_data[state_data['StateID'] == state]['PopDensity'].values[0]

    test.loc[index,'UrbanPercent'] = urban
    test.loc[index,'Population'] = pop
    test.loc[index,'PopDensity'] = pop_den
```

## Save preprocessed data

Saving data at this step to avoid repeating it in the future.


```python
#train.to_csv('processed_train.csv')
#test.to_csv('processed_test.csv')
```

## Import preprocessed data


```python
train = pd.read_csv('processed_train.csv')
test = pd.read_csv('processed_test.csv')
```

## Encode categorical variables


```python
train = pd.get_dummies(train, columns = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                                 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
                                 'State', 'Type', 'Group'
                                ])
test = pd.get_dummies(test, columns = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                                 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
                                 'State', 'Type', 'Group'
                                ])
```

## Make sure train and test have same columns

Encoding variables like 'Breed1' creates many new columns, some of which may only exist in the training set or test set. To remedy this we can make sure each dataset has the same columns and if a column was missing, its values will be fill with 0.


```python
diff_columns = set(train.columns).difference(set(test.columns))
for i in diff_columns:
    test[i] = test.apply(lambda _: 0, axis=1)
diff_columns2 = set(test.columns).difference(set(train.columns))
for i in diff_columns2:
    train[i] = train.apply(lambda _: 0, axis=1)
test = test[train.columns]
```


```python
train.shape
```




    (14993, 453)




```python
test.shape
```




    (3948, 453)



Training set and test set now have the same number of columns.

## Check multicollinearity

To deal with variables that may be highly correlated with eachother, we can grab all of those pairs where the correlation value is above the threshold of 0.85.


```python
corr = train.corr()
indices = np.where(corr > 0.85)
indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
indices
```




    [('bounding_conf', 'bounding_imp'),
     ('bounding_conf', 'pixel_frac'),
     ('dom_blue', 'dom_green'),
     ('dom_green', 'dom_red'),
     ('vertex_y2', 'bounding_conf2'),
     ('vertex_y2', 'bounding_imp2'),
     ('vertex_y2', 'pixel_frac2'),
     ('vertex_y2', 'score2'),
     ('bounding_conf2', 'bounding_imp2'),
     ('bounding_conf2', 'pixel_frac2'),
     ('bounding_conf2', 'score2'),
     ('bounding_imp2', 'pixel_frac2'),
     ('bounding_imp2', 'score2'),
     ('dom_blue2', 'dom_green2'),
     ('dom_blue2', 'dom_red2'),
     ('dom_green2', 'dom_red2'),
     ('pixel_frac2', 'score2'),
     ('vertex_x3', 'vertex_y3'),
     ('vertex_x3', 'bounding_conf3'),
     ('vertex_x3', 'bounding_imp3'),
     ('vertex_x3', 'pixel_frac3'),
     ('vertex_x3', 'score3'),
     ('vertex_y3', 'bounding_conf3'),
     ('vertex_y3', 'bounding_imp3'),
     ('vertex_y3', 'pixel_frac3'),
     ('vertex_y3', 'score3'),
     ('bounding_conf3', 'bounding_imp3'),
     ('bounding_conf3', 'pixel_frac3'),
     ('bounding_conf3', 'score3'),
     ('bounding_imp3', 'pixel_frac3'),
     ('bounding_imp3', 'score3'),
     ('dom_blue3', 'dom_green3'),
     ('dom_blue3', 'dom_red3'),
     ('dom_green3', 'dom_red3'),
     ('pixel_frac3', 'score3'),
     ('Cat_Cute', 'Cat_Hypo'),
     ('Cat_Cute', 'Type_2'),
     ('Cat_Cute', 'Group_Cat'),
     ('Cat_Hypo', 'Type_2'),
     ('Cat_Hypo', 'Group_Cat'),
     ('Population', 'State_41326'),
     ('PopDensity', 'State_41401'),
     ('Breed1_143', 'Breed2_146'),
     ('Breed1_155', 'Breed2_155'),
     ('Breed1_307', 'Group_Misc'),
     ('Type_2', 'Group_Cat')]



Before immediately dropping one of each of the above pairs, I decided to look at the list closely and decided that some in some pairs, dropping one of the variables over the other is better. These are 'Group_Cat' and 'Type_2' because the inclusion of 'Cat_Cute' and 'Cat_Hypo' made them redundant. And also 'State_41326' and 'State_41401' because they correlated highly with 'Population' and 'PopDensity' respectively, but the latter two are more important to keep in the dataset.


```python
drop_list = list(['Group_Cat', 'Type_2', 'State_41326', 'State_41401'])
for i in manual_drop:
    train.drop(i, axis=1, inplace=True)
    test.drop(i, axis=1, inplace=True)
```


```python
for i in indices:
    if (i[0] in drop_list) or (i[1] in drop_list):
        pass
    else:
        try:
            train.drop(i[0], axis=1, inplace=True)
            test.drop(i[0], axis=1, inplace=True)
            drop_list.append(i[0])
        except:
            ## already dropped
            pass
```

Below are all the columns that were dropped to deal with multicollinearity.


```python
drop_list
```




    ['Group_Cat',
     'Type_2',
     'State_41326',
     'State_41401',
     'bounding_conf',
     'dom_blue',
     'dom_green',
     'vertex_y2',
     'bounding_conf2',
     'bounding_imp2',
     'dom_blue2',
     'dom_green2',
     'pixel_frac2',
     'vertex_x3',
     'vertex_y3',
     'bounding_conf3',
     'bounding_imp3',
     'dom_blue3',
     'dom_green3',
     'pixel_frac3',
     'Cat_Cute',
     'Breed1_143',
     'Breed1_155',
     'Breed1_307']



## Set target variable


```python
target = train['AdoptionSpeed'].astype('int')
```

## Drop irrelevent columns

Dropping unnecessary columns as well as target column 'AdoptionSpeed'.


```python
X = train.drop(['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed', 'Unnamed: 0'], axis=1)
X_pred = test.drop(['Name', 'RescuerID', 'Description', 'Unnamed: 0'], axis=1)
```

## Test metric

According to the rules for the Kaggle competition, the results are scored using the quadratic weighted kappa. I will be using the cohen_kappa_score with 'weights' set to quadratic from sklearn.metrics to evaluate my results.

## Set aside validation set

Although the data provided is labeled as 'train', to test classifier performance we need to set aside a validation set.


```python
X_train, X_val, target_train, target_val = train_test_split(X, 
                                                      target, 
                                                      test_size=0.25, 
                                                      random_state=47)
```

## Baseline classifier performance

<b> Now it is time to start testing some classifiers. I will grab a baseline score for a RandomForest, XGBoost, and Adaboost classifier.

### Baseline RandomForest


```python
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, target_train)
```

    C:\Users\Matthew\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:248: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
cohen_kappa_score(target_val, clf_rf.predict(X_val), weights='quadratic')
```




    0.29257257000903714




```python
feature_importances = pd.DataFrame(clf_rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>score</th>
      <td>0.045825</td>
    </tr>
    <tr>
      <th>DescLength</th>
      <td>0.044196</td>
    </tr>
    <tr>
      <th>pixel_frac</th>
      <td>0.043875</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.041879</td>
    </tr>
    <tr>
      <th>dom_red</th>
      <td>0.041559</td>
    </tr>
    <tr>
      <th>magnitude</th>
      <td>0.039455</td>
    </tr>
    <tr>
      <th>score2</th>
      <td>0.036550</td>
    </tr>
    <tr>
      <th>NameLength</th>
      <td>0.034588</td>
    </tr>
    <tr>
      <th>dom_red2</th>
      <td>0.033189</td>
    </tr>
    <tr>
      <th>sentiment_score</th>
      <td>0.033013</td>
    </tr>
    <tr>
      <th>vertex_x</th>
      <td>0.032567</td>
    </tr>
    <tr>
      <th>vertex_y</th>
      <td>0.030376</td>
    </tr>
    <tr>
      <th>dom_red3</th>
      <td>0.028200</td>
    </tr>
    <tr>
      <th>score3</th>
      <td>0.027632</td>
    </tr>
    <tr>
      <th>PhotoAmt</th>
      <td>0.025038</td>
    </tr>
    <tr>
      <th>vertex_x2</th>
      <td>0.023397</td>
    </tr>
    <tr>
      <th>Population</th>
      <td>0.015160</td>
    </tr>
    <tr>
      <th>UrbanPercent</th>
      <td>0.014829</td>
    </tr>
    <tr>
      <th>Quantity</th>
      <td>0.013536</td>
    </tr>
    <tr>
      <th>PopDensity</th>
      <td>0.012904</td>
    </tr>
    <tr>
      <th>Fee</th>
      <td>0.012840</td>
    </tr>
    <tr>
      <th>Group_Misc</th>
      <td>0.010596</td>
    </tr>
    <tr>
      <th>Gender_2</th>
      <td>0.010524</td>
    </tr>
    <tr>
      <th>Color1_1</th>
      <td>0.010236</td>
    </tr>
    <tr>
      <th>Sterilized_2</th>
      <td>0.010147</td>
    </tr>
  </tbody>
</table>
</div>



<b> The top three importance features for the baseline RandomForest classifier are 'score', 'dom_red', and 'DescLength'.

### Baseline XGBoost


```python
clf_xgb = xgb.XGBClassifier()
clf_xgb.fit(X_train, target_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='multi:softprob', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1)




```python
cohen_kappa_score(target_val, clf_xgb.predict(X_val), weights='quadratic')
```




    0.36919141456485527



<b> Using an XGBClassifier improved the cohen score significantly.


```python
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(12,18))
plot_importance(clf_xgb, max_num_features=25, height=0.8, ax=ax)
plt.show()
```


![png](final_files/final_127_0.png)


<b> From the feature importance chart it seems that 'Age', 'DescLength', and 'score' are the top 3 most important features.

### AdaBoost Baseline


```python
clf_ada = AdaBoostClassifier()
clf_ada.fit(X_train, target_train)
```




    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=50, random_state=None)




```python
cohen_kappa_score(target_val, clf_ada.predict(X_val), weights='quadratic')
```




    0.33028851731478026




```python
feature_importances = pd.DataFrame(clf_ada.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.06</td>
    </tr>
    <tr>
      <th>vertex_y</th>
      <td>0.06</td>
    </tr>
    <tr>
      <th>DescLength</th>
      <td>0.06</td>
    </tr>
    <tr>
      <th>Group_Misc</th>
      <td>0.04</td>
    </tr>
    <tr>
      <th>pixel_frac</th>
      <td>0.04</td>
    </tr>
    <tr>
      <th>magnitude</th>
      <td>0.04</td>
    </tr>
    <tr>
      <th>UrbanPercent</th>
      <td>0.04</td>
    </tr>
    <tr>
      <th>Color1_7</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Type_1</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>State_41336</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Sterilized_3</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Sterilized_2</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Dewormed_2</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>FurLength_3</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>FurLength_1</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Color3_5</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Color1_1</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Breed1_179</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Breed1_11</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Gender_1</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Breed1_213</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Breed2_291</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Breed2_247</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Breed2_207</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Breed1_283</th>
      <td>0.02</td>
    </tr>
  </tbody>
</table>
</div>



## Parameter tuning

<b> Now I will try to improve the baseline scores of the three classifiers by using GridSearchCV to find the optimal parameters for each classifier.

### RandomForest Tuning


```python
rf_params = {
    'bootstrap': [True, False],
    'max_depth': [25, 50, 75, 100],
    'max_features': ['auto'],
    'min_samples_leaf': [2, 3, 5, 10],
    'min_samples_split': [5, 10, 15],
    'n_jobs':[-1],
    'n_estimators': [50, 100, 200, 300],
    'random_state' : [47]
}
```


```python
rf_gridsearch = GridSearchCV(estimator = clf_rf, 
                                      param_grid = rf_params, 
                                      cv = 3, 
                                      n_jobs = -1, 
                                      verbose = 1, 
                                      scoring=make_scorer(cohen_kappa_score,weights='quadratic'))
```


```python
rf_gridsearch.fit(X_train, target_train)
```

    Fitting 3 folds for each of 384 candidates, totalling 1152 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:   10.9s
    [Parallel(n_jobs=-1)]: Done 176 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 426 tasks      | elapsed:  2.4min
    [Parallel(n_jobs=-1)]: Done 776 tasks      | elapsed:  5.0min
    [Parallel(n_jobs=-1)]: Done 1152 out of 1152 | elapsed:  8.0min finished
    




    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': [50, 100, 200, 300], 'bootstrap': [True, False], 'n_jobs': [-1], 'min_samples_leaf': [2, 3, 5, 10], 'max_features': ['auto'], 'max_depth': [25, 50, 75, 100], 'min_samples_split': [5, 10, 15], 'random_state': [47]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=make_scorer(cohen_kappa_score, weights=quadratic),
           verbose=1)




```python
rf_gridsearch.best_params_
```




    {'bootstrap': False,
     'max_depth': 25,
     'max_features': 'auto',
     'min_samples_leaf': 2,
     'min_samples_split': 5,
     'n_estimators': 300,
     'n_jobs': -1,
     'random_state': 47}




```python
rf_gridsearch.best_score_
```




    0.343240804127584



### Testing best parameters on validation set


```python
clf_rf_best = RandomForestClassifier(bootstrap=False, max_depth=25, max_features='auto', min_samples_leaf=2,
                                min_samples_split=5, 
                                n_estimators=300, n_jobs=-1, random_state=47)
```


```python
clf_rf_best.fit(X_train, target_train)
```




    RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                max_depth=25, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=2, min_samples_split=5,
                min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
                oob_score=False, random_state=47, verbose=0, warm_start=False)




```python
cohen_kappa_score(target_val, clf_rf_best.predict(X_val),weights='quadratic')
```




    0.3582847219900983



### XGBoost Tuning


```python
xgb_params = {'objective' : ['multi:softmax'],
              'eta' : [0.01],
              'max_depth' : [3, 4, 6],
              'min_child_weight' : [2, 3, 4],
}
```


```python
xgb_gridsearch = GridSearchCV(estimator = clf_xgb, 
                                      param_grid = xgb_params, 
                                      cv = 3, 
                                      n_jobs = -1, 
                                      verbose = 1, 
                                      scoring=make_scorer(cohen_kappa_score,weights='quadratic'))
```


```python
xgb_gridsearch.fit(X_train, target_train)
```

    Fitting 3 folds for each of 9 candidates, totalling 27 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  27 out of  27 | elapsed:  4.3min finished
    




    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='multi:softprob', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_child_weight': [2, 3, 4], 'objective': ['multi:softmax'], 'max_depth': [3, 4, 6], 'eta': [0.01]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=make_scorer(cohen_kappa_score, weights=quadratic),
           verbose=1)




```python
xgb_gridsearch.best_params_
```




    {'eta': 0.01,
     'max_depth': 4,
     'min_child_weight': 4,
     'objective': 'multi:softmax'}




```python
xgb_gridsearch.best_score_
```




    0.34320068021975203



### Testing best parameters on validation set


```python
clf_xgb_best = xgb.XGBClassifier(eta = 0.01, max_depth = 4, min_child_weight = 4, objective = 'multi:softmax')
```


```python
clf_xgb_best.fit(X_train, target_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, eta=0.01, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=4, min_child_weight=4, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='multi:softprob', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
           subsample=1)




```python
cohen_kappa_score(target_val, clf_xgb_best.predict(X_val),weights='quadratic')
```




    0.38268750072423174



### AdaBoost Tuning


```python
ada_params = {'base_estimator': [None, DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=5)],
              'n_estimators': [50, 100, 200, 300]}
```


```python
ada_gridsearch = GridSearchCV(estimator = clf_ada, 
                                      param_grid = ada_params, 
                                      cv = 3, 
                                      n_jobs = -1, 
                                      verbose = 1, 
                                      scoring=make_scorer(cohen_kappa_score,weights='quadratic'))
```


```python
ada_gridsearch.fit(X_train, target_train)
```

    Fitting 3 folds for each of 12 candidates, totalling 36 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  36 out of  36 | elapsed:   57.9s finished
    




    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=50, random_state=None),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'base_estimator': [None, DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weigh...resort=False, random_state=None,
                splitter='best')], 'n_estimators': [50, 100, 200, 300]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=make_scorer(cohen_kappa_score, weights=quadratic),
           verbose=1)




```python
ada_gridsearch.best_params_
```




    {'base_estimator': None, 'n_estimators': 100}




```python
ada_gridsearch.best_score_
```




    0.30248923825346563



### Testing best parameters on validation set


```python
clf_ada_best = AdaBoostClassifier(base_estimator=None, n_estimators=100)
```


```python
clf_ada_best.fit(X_train, target_train)
```




    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=100, random_state=None)




```python
cohen_kappa_score(target_val, clf_ada_best.predict(X_val),weights='quadratic')
```




    0.3467641968995947



## Combining tuned classifiers with VotingClassifier

<b> Since all three classifiers have decent, comparable performances, I will combined all three into one final ensemble classifier using VotingClassifer with soft voting.


```python
clf_vot = VotingClassifier(estimators=[('RF',clf_rf_best),('XGB',clf_xgb_best),('ADA',clf_ada_best)],voting='soft')
```


```python
clf_vot.fit(X_train, target_train)
```




    VotingClassifier(estimators=[('RF', RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                max_depth=25, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=2, min_samples_split=5,
                min_wei...='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=100, random_state=None))],
             flatten_transform=None, n_jobs=None, voting='soft', weights=None)




```python
cohen_kappa_score(target_val, clf_vot.predict(X_val),weights='quadratic')
```




    0.3863173663782099



To visualize the differences in predictions of the three base classifiers and the ensemble classifier, we can look at bar charts of each 'AdoptionSpeed' prediction for the classifiers below.


```python
probas = [c.fit(X_train, target_train).predict(X_val) for c in (clf_rf_best, clf_xgb_best, clf_ada_best, clf_vot)]
```


```python
class_0 = list()
class_1 = list()
class_2 = list()
class_3 = list()
class_4 = list()

for i in probas:
    class_0.append(np.array(np.unique(i, return_counts=True))[1][0])
    class_1.append(np.array(np.unique(i, return_counts=True))[1][1])
    class_2.append(np.array(np.unique(i, return_counts=True))[1][2])
    class_3.append(np.array(np.unique(i, return_counts=True))[1][3])
    class_4.append(np.array(np.unique(i, return_counts=True))[1][4])
```


```python
N = 4  # number of groups
ind = np.arange(N)  # group positions
width = 0.5 # bar width

ax1 = plt.subplot2grid((2,6), (0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

# bars for base classifiers
p0 = ax1.bar(ind + width, np.hstack(([class_0[:-1], [0]])), width, color='green', alpha=0.5, edgecolor='k')
p1 = ax2.bar(ind + width, np.hstack(([class_1[:-1], [0]])), width, color='blue',alpha=0.5, edgecolor='k')
p2 = ax3.bar(ind + width, np.hstack(([class_2[:-1], [0]])), width, color='red',alpha=0.5, edgecolor='k')
p3 = ax4.bar(ind + width, np.hstack(([class_3[:-1], [0]])), width, color='orange',alpha=0.5, edgecolor='k')
p4 = ax5.bar(ind + width, np.hstack(([class_4[:-1], [0]])), width, color='purple',alpha=0.5, edgecolor='k')

# bars for voting classifier
p5 = ax1.bar(ind + width, [0, 0, 0, class_0[-1]], width,color='green', edgecolor='k')
p6 = ax2.bar(ind + width, [0, 0, 0, class_1[-1]], width,color='blue', edgecolor='k')
p7 = ax3.bar(ind + width, [0, 0, 0, class_2[-1]], width,color='red', edgecolor='k')
p8 = ax4.bar(ind + width, [0, 0, 0, class_3[-1]], width,color='orange', edgecolor='k')
p9 = ax5.bar(ind + width, [0, 0, 0, class_4[-1]], width,color='purple', edgecolor='k')

# plot annotations
ax1.set_xticks(ind + width)
ax1.set_ylabel('Number of predictions')
ax1.set_xticklabels(['RandomForest',
                    'XGBoost',
                    'AdaBoost',
                    'VotingClassifier'],
                   rotation=40,
                   ha='right')
ax2.set_xticks(ind + width)
ax2.set_xticklabels(['RandomForest',
                    'XGBoost',
                    'AdaBoost',
                    'VotingClassifier'],
                   rotation=40,
                   ha='right')
ax3.set_xticks(ind + width)
ax3.set_xticklabels(['RandomForest',
                    'XGBoost',
                    'AdaBoost',
                    'VotingClassifier'],
                   rotation=40,
                   ha='right')
ax4.set_xticks(ind + width)
ax4.set_ylabel('Number of predictions')
ax4.set_xticklabels(['RandomForest',
                    'XGBoost',
                    'AdaBoost',
                    'VotingClassifier'],
                   rotation=40,
                   ha='right')
ax5.set_xticks(ind + width)
ax5.set_xticklabels(['RandomForest',
                    'XGBoost',
                    'AdaBoost',
                    'VotingClassifier'],
                   rotation=40,
                   ha='right')
ax1.set_title('Adoption Speed 0')
ax2.set_title('Adoption Speed 1')
ax3.set_title('Adoption Speed 2')
ax4.set_title('Adoption Speed 3')
ax5.set_title('Adoption Speed 4')

plt.rcParams["figure.figsize"] = [20,20]
plt.show()
```


![png](final_files/final_173_0.png)


From the above chart we can see how the voting classifier averages out the predictions of the three base classifiers to better predict 'AdoptionSpeed'. One thing to note is that for 'Adoption Speed 0', the RandomForest and XGBoost classifiers predicted far less cases of this value than the AdaBoost classifier. This ultimately did not raise the average generated by the VotingClassifier, but it is still a significantly outlier when comparing the charts side by side.

## Fitting classifier to whole train data

Now that we have our final classifier, we can fit it to the entire training set.


```python
clf_vot_final = VotingClassifier(estimators=[('RF',clf_rf_best),('XGB',clf_xgb_best),('ADA',clf_ada_best)],voting='soft')
```


```python
clf_vot_final.fit(X, target)
```




    VotingClassifier(estimators=[('RF', RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                max_depth=25, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=2, min_samples_split=5,
                min_wei...='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=100, random_state=None))],
             flatten_transform=None, n_jobs=None, voting='soft', weights=None)



## Predictions on test data

With the final classifier trained on all the data, we can now make predictions based on the given test data from the Kaggle competition.


```python
test_pred = clf_vot_final.predict(X_pred.drop(['AdoptionSpeed','PetID'], axis=1))
```

Now we can view the distribution of predictions on the test data.


```python
plt.rcParams["figure.figsize"] = [7,7]
pd.DataFrame(test_pred).hist()
plt.title('Adoption Speed Predictions on Test Data')
plt.ylabel('Number of prediction')
plt.xticks(np.arange(5))
```




    ([<matplotlib.axis.XTick at 0x2291f2bd4a8>,
      <matplotlib.axis.XTick at 0x22971b59e10>,
      <matplotlib.axis.XTick at 0x22971b59b38>,
      <matplotlib.axis.XTick at 0x2291f2cdeb8>,
      <matplotlib.axis.XTick at 0x2291f2a5630>],
     <a list of 5 Text xticklabel objects>)




![png](final_files/final_183_1.png)



```python
pd.DataFrame(test_pred)[0].value_counts()
```




    4    1788
    2    1257
    1     684
    3     219
    Name: 0, dtype: int64



Somewhat surprising, there are no predictions of an 'AdoptionSpeed' of 0 for any of the test data. In the training data, there were significantly fewer cases of the lowest 'AdoptionSpeed' which may be why it's possible for the test set to have zero occurances. However, it still seems unusual for that kind of imbalance and could be investigated further.

## Saving predictions on test data

Saving the predictions to a seperate CSV file will allow me to upload it to the Kaggle competition to receive a scoring.


```python
pred['PetID'] = X_pred['PetID']
pred['AdoptionSpeed'] = test_pred
pred.set_index('PetID').to_csv("submission.csv", index=True)
```

## Scoring results on Kaggle

The prediction submitted received a score of <b>0.333</b> on the Kaggle competition. This placed us in about the 50th percentile of all of the competitors.

## Summary

The current high score on the Kaggle competition is 0.452 so there is certainly room for improvement. However, the final classifier of this project still scored decently well given this was my first participation in a Kaggle competition. The final classifier was a definite improvement over the baseline classifiers and even the tuned classifiers so choosing to ensemble them using a VotingClassifier was a good choice. 

## Future work

<b>With more time I would focus on the following:

1) Class imbalance
    - The lowest 'AdoptionSpeed' had a very low occurence in the training data and no occurence in the test predictions. This seems unusual and could be investigate further. I would want to look into the confusion matrix of predictions on the training data to see how well or how not so well the classifier predicts an 'AdoptionSpeed' of 0. 
    - I would consider using SMOTE to balance the classes of 'AdoptionSpeed' better.
2) Removing data
    - I added a few of my own columns to the dataset. In hindsight, some of this additional data could have just added noise to the dataset. I would test removing some of the added columns of data.
    - I would also test not using as much of the image data as I did - maybe only the first photo for each pet.
3) Further investigation of image and sentiment data
    - For both the image and sentiment data, I simply used the variables provided without much research into the Google APIs behind them. I would like to learn more about how these APIs work and what the values they generate signify.
    - I would also like to possibly try running either the images or descriptions through my own computer vision or NLP algorithm.
4) Utilize other classifiers
    - Most of the highest scoring kernals on the Kaggle competition use LightGBM for their predictions. I would have liked to learn how to utilize that as a classifier to improve my score.
    - I could also include more base classifiers in my VotingClassifier as well as trying out other ensemble methods.
