# Exploring Data Missingness 
# Ref. http://datascience.ibm.com/blog/missing-data-
#conundrum-exploration-and-imputation-techniques/
# https://github.com/ResidentMario/missingno
  
import pandas as pd    
import missingno as msno
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

### READ IN RAW DATA 

print( "\nReading data from disk ...")
properties = pd.read_csv('./properties_2016.csv')
train = pd.read_csv('./train_2016_v2.csv')
sample = pd.read_csv('./sample_submission.csv')

### Analyse the Dimensions of our Datasets.
print("Training Size:" + str(train.shape))
print("Property Size:" + str(properties.shape))
print("Sample Size:" + str(sample.shape))

merged_df = pd.merge(train,properties)  
missingdata_df = merged_df.columns[merged_df.isnull().any()].tolist()  

### The nullity matrix 
#gives you a data-dense display which lets you quickly visually pick out the 
#missing data patterns in the dataset. Also, the sparkline on the right gives 
#you a summary of the general shape of the data completeness and an indicator 
#of the rows with maximum and minimum rows.
msno.matrix(merged_df[missingdata_df]) 

### The missingno bar chart 
#is a visualization of the data nullity. We log transformed the data on the 
#y-axis to better visualize features with very large missing values.
msno.bar(merged_df[missingdata_df], color="blue", log=True, figsize=(30,18))

###The correlation heatmap 
#describes the degree of nullity relationship between the different features. 
#The range of this nullity correlation is from -1 to 1 (-1 ≤ R ≤ 1). 
#Features with no missing value are excluded in the heatmap. 
#If the nullity correlation is very close to zero (-0.05 < R < 0.05), no value 
#will be displayed. Also, a perfect positive nullity correlation (R=1) indicates 
#when the first feature and the second feature both have corresponding missing values 
#while a perfect negative nullity correlation (R=-1) means that one of the features is 
#missing and the second is not missing.
msno.heatmap(merged_df[missingdata_df], figsize=(20,20))

### More fully correlate variable completion 
#The dendrogram reveals trends deeper than the pairwise 
#ones visible in the correlation heatmap.

msno.dendrogram(merged_df[missingdata_df], orientation='left')

### Quadtree nullity distribution 
#
#msno.geoplot(merged_df[missingdata_df], x='longitude', y='latitude')#, by='regionidzip',histogram=True)


