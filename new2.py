#!/usr/bin/env python
# coding: utf-8

# #### **Machine Learning havig three types of Recommender System**
# ##### **1. Content Based Recommender System <br>2. Collaborative filtiring<br>3. Hybrid<br>In this Project We are using Content Based Recommender System**

# #### **Project Flow**<br> **1. Loading Data**<br> **2. Data Preprocessing**<br> **3. Model Building**<br> **4. Website Development** <br> **5. Deployment**

# In[6]:


# from pyforest import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot,plot
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas.core.reshape.merge import merge
from ast import literal_eval


# **Loading Movies Data**

# In[7]:


movies=pd.read_csv('./tmdb_5000_movies.csv')
credits=pd.read_csv('./tmdb_5000_credits.csv')
movies.head(2)


# In[8]:


credits.head()


# In[9]:


movies.head()


# ##### Merge two dataframe on the Basis of Title

# In[10]:


movies=movies.merge(credits,on='title')
movies


# In[11]:


movies.head(2).shape


# In[12]:


movies.head(2)
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[13]:


movies.columns


# In[14]:


movies.isnull().sum()


# **We Take only those Columns that we need in the Dataset and remove other Columns**

# In[15]:


#note that by taking onlly what we need from data qualize droping null columns and data we don't need 
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew' ]]


# In[15]:





# In[16]:


movies.head(2)


# #### Data Preprocessing

# In[17]:


movies.isnull().sum()


# In[18]:


movies.duplicated().sum()


# In[19]:


movies.iloc[0].genres


# In[20]:


## This function evaluates an expression node or a string consisting of a Python literal or container display.
import ast
def convert(text):
    L=[]
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


# In[21]:


movies.dropna(inplace=True)


# In[22]:


movies['genres']=movies['genres'].apply(convert)


# In[23]:


movies.head()


# In[24]:


movies['keywords']=movies['keywords'].apply(convert)


# In[25]:


movies.head()


# In[26]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[27]:


movies['cast']=movies['cast'].apply(convert3)


# In[28]:


movies.head()


# In[29]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[30]:


movies


# In[31]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[32]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[33]:


movies


# In[34]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[35]:


movies.head()


# **Removing Space in Every Words**

# In[36]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[37]:


# movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[38]:


movies.head()


# **We Make  new tags Column in Dataset and add all the Column in it**

# In[39]:


# movies['overview'] = movies['overview'].apply(lambda x : x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()


# In[40]:


movies.head()


# In[41]:


new = movies.drop(columns=['keywords','crew'])
# new = movies.drop(columns=['title', 'tagline', 'status', 'homepage', 
#                                         'keywords','crew','vote_count', 'vote_average',
#                                         'tagline', 'spoken_languages', 'runtime',
#                                         'popularity', 'production_companies', 'budget',
#                                         'production_countries', 'release_date', 'revenue',
#                                         'title', 'original_language'])

new.columns


# In[41]:





# ##### **Convert tags Column into String**

# In[42]:


movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies.head()
new['tags']=new['tags'].apply(lambda x:x)


# **Convert tags Column to Lower**

# In[46]:


new.head()


# In[47]:


new['tags'][0]


# In[49]:


new.to_csv('final_data.csv', index=False)
final_data = pd.read_csv(r'/Users/grab/Documents/temp/shashu/27 nov/final_data.csv')
final_data.head()


# In[51]:


import pandas as pd

pd.read_csv('/Users/grab/Documents/temp/shashu/27 nov/final_data.csv')


# In[52]:


import pandas as pd

pd.read_csv('/Users/grab/Documents/temp/shashu/27 nov/final_data.csv')


# In[ ]:





# In[54]:


import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ff = pd.read_csv(r'/Users/grab/Documents/temp/shashu/27 nov/final_data.csv')
ff


# In[56]:


def get_data():
    movie_data = pd.read_csv(r'/Users/grab/Documents/temp/shashu/27 nov/final_data.csv')
    ##Convert the title of all the movies to lowercase letters.
    movie_data['original_title'] = movie_data['original_title'].str.lower()
    ##Return the dataset as the function’s result.
    return movie_data


# In[57]:


def combine_data(data):
    ##Drop the attributes not required for feature extraction.
    data_recommend = data.drop(columns=['id', 'original_title','plot'])
    ##Combine the two columns cast and genres into one single column.
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(
                                                                        lambda x: ','.join(x.dropna().astype(str)),axis=1)
    #We have a combined column with cast and genres values present in it 
    # so we remove the cast and genres columns existing separately from our dataset.
    data_recommend = data_recommend.drop(columns=[ 'cast','genres'])
    return data_recommend


# In[58]:


def transform_data(data_combine, data_plot):
    ##Make an object for CountVectorizer and initiate to remove English stopwords using the stop_words parameter.
    count = CountVectorizer(stop_words='english')
    #Fit the CountVectorizer object count onto the value returned by combine_data()
    # combined column values of cast and genres. After this, we get a sparse matrix
    count_matrix = count.fit_transform(data_combine['combine'])
    # Make an object for TfidfVectorizer and initiate to remove English stopwords using the stop_words parameter.
    tfidf = TfidfVectorizer(stop_words='english')
    #Fit the TfidfVectorizer object tfdif onto the column plot that we get from get_data(). After this, we get a sparse matrix 
    tfidf_matrix = tfidf.fit_transform(data_plot['plot'])
    #We combine the two sparse matrices we get by CountVectorizer and TfidfVectorizer into a single sparse matrix.
    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
    #We now apply Cosine Similarity on our combined sparse matrix.
    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)
    # Return the cosine similarity matrix generated 
    return cosine_sim


# In[60]:


def recommend_movies(title, data, combine, transform):
    #Create a Pandas Series with indices of all the movies present in our dataset.
    indices = pd.Series(data.index, index = data['original_title'])
    #Get the index of the input movie that is passed onto our recommend_movies() function in the title parameter.
    index = indices[title]
    #Here we store the Cosine Values of each movie with respect to our input movie.
    sim_scores = list(enumerate(transform[index]))
    #After getting the cosine values we sort them in reverse order.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #We need the top 20 movies with respect to our input movie.
    sim_scores = sim_scores[1:21]
    #In these lines, we store the movie indices with their respective columns.
    movie_indices = [i[0] for i in sim_scores]
    #We create a Pandas DataFrame with Movie_Id, Name, Genres as the columns.
    
    #We store all the 20 movies similar to our input movie
    movie_id = data['id'].iloc[movie_indices]
    movie_title = data['original_title'].iloc[movie_indices]
    movie_genres = data['genres'].iloc[movie_indices]
    
    #Return the Pandas DataFrame with the top 20 movie recommendations.
    recommendation_data = pd.DataFrame(columns=['Id','Name','Genres'])

    recommendation_data['Id'] = movie_id
    recommendation_data['Name'] = movie_title
    recommendation_data['Genres'] = movie_genres

    return recommendation_data


# In[61]:


def results(movie_name):
    '''
    convert the movie_name to lower case as all the movies is in lower case in our dataset.
    We do this as a precautionary measure. If a user types a movie name in lower case and upper case letters together then 
    it won't be a problem as our function will still return the results.
    '''
    movie_name = movie_name.lower()
    '''
    We store the values returned by get_data(), combine_data() and transform_data().
    '''
    find_movie = get_data()
    combine_result = combine_data(find_movie)
    transform_result = transform_data(combine_result,find_movie)
    '''
    Check whether the input movie is present in our dataset.
    If not found in our dataset then we return that the movie is not found.
    '''
    
    '''
    If our movie is present in the dataset then we call our recommend_movies() function 
    and pass the return values of get_data(), combine_data() and transform_data() along with the movie name 
    as the function's parameter.
    '''
    if movie_name not in find_movie['original_title'].unique():
        return 'Movie not in Database'
    
    else:
        recommendations = recommend_movies(movie_name, find_movie, combine_result, transform_result)
        return recommendations.to_dict('records')


# In[62]:


name = input('What is the name of movie?\n') 
results(name)

