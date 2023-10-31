#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
import math

import librosa
import librosa.display

import warnings
warnings.filterwarnings('ignore')


# In[4]:


path='C:\\Music Genre Classification\\Data'
genres=os.listdir(os.path.join(path,'genres_original\\'))
print(genres)


# In[5]:


y,sr= librosa.load(os.path.join(path,'genres_original','classical','classical.00001.wav'))
print("y=",y,'\n')
print("Sample rate",sr,'\n')
#sample rate- number of samples of audio collected to represent the audio digitally
#more the sample rates, better the quality.
#the sample rate 22050 hz is considered good.

#y represents recored amplitude of the samples.y is 2-d array, y[0] represents amplitudes and y[1] represents number of channels
print("Duration :",y.shape[0]/sr)


# In[6]:


#trim the silence
audio,_=librosa.effects.trim(y)
print("y: ",y,'\n')
print('Duration :',audio.shape[0]/sr)
#no silence in the beginning or end


# In[7]:


plt.figure(figsize=(16,6))
librosa.display.waveshow(y=audio,sr=sr,color='r')
plt.title('Classical-1')


# In[8]:


for i in genres:
    aud,sr=librosa.load(os.path.join(path,'genres_original',i,f'{i}.00001.wav'))
    plt.figure(figsize=(12,4))
    librosa.display.waveshow(y=aud,sr=sr,color='r')
    plt.title(f'{i}')


# In[9]:


n_fft=2048 #default value recommended, n_fft represents the number of samples that will be converted at once.
hop_length=512 
win_length=2048 #window using which samples are converted.

for i in genres:
    aud,sr=librosa.load(os.path.join(path,'genres_original',i,f'{i}.00001.wav'))
    aud_ft= np.abs(librosa.stft(aud, n_fft = n_fft, hop_length = hop_length,win_length=win_length))
#     print(np.shape(aud_ft)) #(1025,1302) 
    plt.figure(figsize=(12,4))
    plt.plot(aud_ft[:400,:])#viewing only upto 400 Hz
    plt.title(f'{i}')


# In[10]:


music_stft = np.abs(librosa.stft(audio,n_fft=n_fft,hop_length= hop_length))
plt.figure(figsize = (16,6))
plt.plot(music_stft); 


# In[11]:


# Converting from amplitute(Linear scale) to decibels, a log scale
music_stft_decibels = librosa.amplitude_to_db(music_stft, ref= np.max) 

# Plotting the spectogram 
plt.figure(figsize=(16,6))
librosa.display.specshow(music_stft_decibels, sr = sr, hop_length= hop_length, x_axis='time', y_axis='log', cmap = "cool");
plt.colorbar(); 


# In[13]:


# Mel spectogram 

metal_sample = "metal/metal.00032.wav"
y, sr = librosa.load(os.path.join(path,"genres_original",metal_sample))
y,_ = librosa.effects.trim(y)
    
S = librosa.feature.melspectrogram(y, sr = sr)
S_DB = librosa.amplitude_to_db(S, ref= np.max)

plt.figure(figsize = (16,6))


librosa.display.specshow(S_DB, sr = sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')
plt.colorbar();
plt.title("Metal Mel Spectogram", fontsize = "23")


# In[14]:


n_mels=128
mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
librosa.display.specshow(mel, sr = sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')


# In[15]:


zero_crossing_rate=librosa.zero_crossings(y) # has a boolean output
sum(zero_crossing_rate)


# In[16]:


harmonics,percussive=librosa.effects.hpss(y)
plt.figure(figsize=(15,5))
plt.plot(harmonics);
plt.figure(figsize=(15,5))
plt.plot(percussive);


# In[17]:


spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0] #returns spectral centroid per frame.
print('Centroids:', spectral_centroids, '\n')
print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')

# Computing the time variable for visualization
frames = range(len(spectral_centroids))

# Converts frame counts to time (seconds)
t = librosa.frames_to_time(frames)

print('frames:', frames, '\n')
print('t:', t)

# Function that normalizes the Sound Data
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
plt.figure(figsize = (16, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.2, color = 'red');
plt.plot(t, normalize(spectral_centroids), color='blue')


# In[18]:


spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0]
plt.figure(figsize = (16, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.2, color = 'red');
plt.plot(t, normalize(spectral_rolloff), color='blue')


# In[19]:


mfccs = librosa.feature.mfcc(y, sr=sr)
print('mfccs shape:', mfccs.shape)
plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
#the data is in small range,thus scaling
mfccs = sklearn.preprocessing.scale(mfccs,axis=1)
plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');


# In[20]:


chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=hop_length)
print('Chromogram shape:', chromagram.shape)
plt.figure(figsize=(16, 6))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length,cmap='coolwarm')


# In[21]:


data=pd.read_csv(os.path.join(path,'features_30_sec.csv'))
data.head()


# In[22]:


data.columns


# In[23]:


from sklearn import preprocessing
data=data.iloc[0:,2:]
Y=data.loc[:,'label']
X=data.loc[:,data.columns!='label']

cols=X.columns
min_max_scaler=preprocessing.MinMaxScaler()
scaled_X=min_max_scaler.fit_transform(X) #the column names are removed
X=pd.DataFrame(scaled_X,columns=cols)


# In[24]:


from sklearn.decomposition import PCA
n=10
pca=PCA(n_components=n)
pc=pca.fit_transform(X)
col_names=[f'PC{i}' for i in range(1,n+1)]
data_X=pd.DataFrame(data=pc,columns=col_names)

final_df=pd.concat([data_X,Y],axis=1)
final_df.head()
# pca.explained_variance_ratio_ 
# sum of variances on each pca(noticed it decreases quite exponentially with n_components)


# In[25]:


plt.figure(figsize = (16, 9))
sns.scatterplot(x = "PC1", y = "PC2", data = final_df, hue = "label", alpha = 0.7,
               s = 100);

plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)


# In[26]:


plt.figure(figsize = (16, 9))
sns.scatterplot(x = "PC1", y = "PC10", data = final_df, hue = "label", alpha = 0.7,
               s = 100);

plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 10", fontsize = 15)


# In[29]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


# In[30]:


#reading the 3 s csv
data= pd.read_csv(os.path.join(path,'features_3_sec.csv'))
data.head()


# In[31]:


data.columns


# In[32]:


data=data.iloc[0:,2:]
Y=data.loc[:,'label']
X=data.loc[:,data.columns!='label']

cols=X.columns

scaler=preprocessing.MinMaxScaler()
scaled_X=scaler.fit_transform(X)

X=pd.DataFrame(scaled_X,columns=cols)


# In[33]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=21)


# In[34]:


def model_assess(model,title):
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')


# In[35]:


# Naive Bayes
nb = GaussianNB()
model_assess(nb, "Naive Bayes")

# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=5000, random_state=0)
model_assess(sgd, "Stochastic Gradient Descent")

# KNN
knn = KNeighborsClassifier(n_neighbors=19)
model_assess(knn, "KNN")

# Decission trees
tree = DecisionTreeClassifier()
model_assess(tree, "Decission trees")

# Random Forest
rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_assess(rforest, "Random Forest")

# Support Vector Machine
svm = SVC(decision_function_shape="ovo")
model_assess(svm, "Support Vector Machine")

# Logistic Regression
lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_assess(lg, "Logistic Regression")

# Neural Nets
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)
model_assess(nn, "Neural Nets")


# In[36]:


le=LabelEncoder()
labels=le.fit_transform(Y)
X_train,X_test,y_train,y_test=train_test_split(X,labels,test_size=0.3,random_state=21)


# Cross Gradient Booster
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
xgb.fit(X_train,y_train)
preds=xgb.predict(X_test)
print('Accuracy',"xgb", ':', round(accuracy_score(y_test, preds), 5), '\n')


# Cross Gradient Booster (Random Forest)
xgbrf = XGBRFClassifier(objective= 'multi:softmax')
xgbrf.fit(X_train,y_train)
preds=xgbrf.predict(X_test)
print('Accuracy',"xgbrf", ':', round(accuracy_score(y_test, preds), 5), '\n')


# In[37]:


xgb1=XGBClassifier(n_estimators=1400,learning_rate=0.03)
xgb1.fit(X_train,y_train)
preds=xgb1.predict(X_test)
print('Accuracy',"xgb1", ':', round(accuracy_score(y_test, preds), 5), '\n')

#the accuracy doesn't increase much


# In[38]:


# Libraries

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

# Read data
data = pd.read_csv(f'{path}/features_30_sec.csv', index_col='filename')

# Extract labels
labels = data[['label']]

# Drop labels from original dataframe
data = data.drop(columns=['length','label'])
data.head()

# Scale the data
data_scaled=preprocessing.scale(data)
print('Scaled data type:', type(data_scaled))
# data_scaled


# In[39]:


# Cosine similarity
similarity = cosine_similarity(data_scaled)
print("Similarity shape:", similarity.shape)

# Convert into a dataframe and then set the row index and column names as labels
sim_df_labels = pd.DataFrame(similarity)
sim_df_names = sim_df_labels.set_index(labels.index)
sim_df_names.columns = labels.index

sim_df_names.head()


# In[40]:


def find_similar_songs(name):
    # Find songs most similar to another song
    series = sim_df_names[name].sort_values(ascending = False)
    
    # Remove cosine similarity == 1 (songs will always have the best match with themselves)
    series = series.drop(name)
    
    # Display the 5 top matches 
    print("\n*******\nSimilar songs to ", name)
    print(series.head(5))


# In[41]:


find_similar_songs('metal.00002.wav') 


# In[ ]:




