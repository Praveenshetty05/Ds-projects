#!/usr/bin/env python
# coding: utf-8

# In[103]:


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
warnings.filterwarnings("ignore")


# # performing EDA(exploratory data analysis)

# # BOOKS

# In[104]:


try:
    books = pd.read_csv("D:\\DS new project\\Books.csv", encoding='ISO-8859-1')
except Exception as e:
    print(f"An error occurred: {e}")


# In[105]:


books


# In[106]:


books.info()


# In[107]:


books.describe()


# In[108]:


#checking of duplicate values


# In[109]:


books.loc[books.duplicated()]


# In[110]:


top_books = books['Book-Title'].value_counts().head(10)


# In[111]:


top_books


# In[112]:


top_books.index


# In[113]:


sns.barplot(x=top_books.values, y=top_books.index, palette='muted')


# In[114]:


books['Year-Of-Publication'].value_counts()


# In[115]:


books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')


# In[116]:


books.info()


# In[117]:


plt.figure(figsize=(15,7))
sns.countplot(y='Book-Author',data=books,order=pd.value_counts(books['Book-Author']).iloc[:10].index)
plt.title('Top 10 Authors')


# In[118]:


plt.figure(figsize=(15,7))
sns.countplot(y='Publisher',data=books,order=pd.value_counts(books['Publisher']).iloc[:10].index)
plt.title('Top 10 Publishers')


# In[119]:


books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('str')
a = list(books['Year-Of-Publication'].unique())
a = set(a)
a = list(a)
a = [x for x in a if x is not None]
a.sort()
print(a)


# In[120]:


books['Book-Author'].fillna('other',inplace=True)


# In[121]:


books.loc[books.Publisher.isnull(),:]


# In[122]:


books.Publisher.fillna('other',inplace=True)


# In[123]:


books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1,inplace=True)


# In[124]:


books.isna().sum()


# In[125]:


books.isna().sum()


# # USERS

# In[126]:


users = pd.read_csv("D:\\DS new project\\Users.csv",encoding='latin-1')


# In[127]:


users


# In[128]:


# shape of the data

users.shape


# In[129]:


users.info()


# In[130]:


users.describe()


# In[131]:


# Checking duplicated values

users.loc[users.duplicated()]


# In[132]:


# Checking Missing values

cols = users.columns 
colours = ['blue', 'yellow'] # yellow is missing. blue is not missing.
sns.heatmap(users[cols].isnull(), cmap=sns.color_palette(colours))


# In[133]:


# Checking Null values

users.loc[users['Age'].isnull()]


# In[134]:


mean = users['Age'].mean()
print(mean)


# In[135]:


# mean impuation on null values

users['Age'] = users['Age'].fillna(users['Age'].mean())
users.isnull().sum()


# In[136]:


users.head()


# OUT LAYERS

# In[137]:


# distplot

sns.distplot(users['Age'])


# In[138]:


# Histogram

users['Age'].hist()
plt.show()


# In[139]:


# Boxplot 

plt.boxplot(users['Age'])
plt.xlabel('Boxplot')
plt.ylabel('Age')
plt.show()


# In[140]:


# Locations

print(users.Location.unique())


# In[141]:


# Location data is not suitable to interpret the information

for i in users:
    users['Country'] = users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')  


# In[142]:


users.Country.nunique()


# In[143]:


# Dropping the Location

users.drop('Location',axis=1,inplace=True)


# In[144]:


users.head()


# In[145]:


users.isnull().sum()


# In[146]:


users['Country']=users['Country'].astype('str')


# In[147]:


a = list(users.Country.unique())
a = set(a)
a = list(a)
a = [x for x in a if x is not None]
a.sort()
print(a)


# In[148]:


users['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


# In[149]:


print(users.Country.nunique())


# In[150]:


plt.figure(figsize=(15,7))
sns.countplot(y='Country', data=users, order=pd.value_counts(users['Country']).iloc[:10].index)
plt.title('Count of users Country wise')


# In[151]:


users.isna().sum()


# RATINGS

# In[152]:


ratings =pd.read_csv("D:\\DS new project\\Ratings.csv",encoding='latin-1')


# In[153]:


ratings


# In[154]:


ratings['User-ID'].value_counts()


# In[155]:


ratings['User-ID'].unique().shape


# In[156]:


x = ratings['User-ID'].value_counts() > 200
x[x]


# In[157]:


y = x[x].index
y


# In[158]:


ratings = ratings[ratings['User-ID'].isin(y)]


# In[159]:


ratings


# In[160]:


plt.figure(figsize=(10,6), dpi=100)
ratings['Book-Rating'].value_counts().plot(kind='bar')
plt.title('Ratings Frequency',  fontsize = 16, fontweight = 'bold')
plt.show()


# CONSOLIDATING OF DATASET

# In[161]:


ratings_with_books = ratings.merge(books,on = 'ISBN')
ratings_with_books


# In[162]:


num_rating = ratings_with_books.groupby('Book-Title')['Book-Rating'].count().reset_index()


# In[163]:


num_rating.head()


# In[164]:


num_rating.rename(columns={'Book-Rating':'num_of_rating'},inplace=True)


# In[165]:


num_rating.head()


# In[166]:


final_ratings=ratings_with_books.merge(num_rating,on = 'Book-Title')


# In[167]:


final_ratings.head()


# In[168]:


final_ratings.shape


# In[169]:


final_ratings.drop_duplicates(['User-ID','Book-Title'],inplace=True)


# In[170]:


final_ratings


# In[171]:


final_ratings = final_ratings.rename({'User-ID' : 'userid','Book-Title' : 'booktitle','Book-Rating' : 'bookrating'},axis=1)


# In[172]:


final_ratings.drop_duplicates(['userid','booktitle'],inplace=True)


# In[173]:


final_ratings


# In[ ]:





# # MODEL BUILDING

# #Collaborative Filtering
# 

# In[174]:


# Now let us create the pivot table

pivot_table = final_ratings.pivot_table(index='userid',
                                   columns='booktitle',
                                   values='bookrating')


# In[175]:


pivot_table 


# In[176]:


# Filling Null values

pivot_table.fillna(0, inplace=True)


# In[177]:


pivot_table


# In[178]:


# Calculating Cosine Similarity between Users

user_sim = 1 - pairwise_distances(pivot_table.values,metric='cosine')
user_sim


# In[179]:


#Store the results in a dataframe

user_sim_df = pd.DataFrame(user_sim)
user_sim_df


# In[180]:


user_sim_df.index = final_ratings.userid.unique()
user_sim_df.columns = final_ratings.userid.unique()


# In[181]:


user_sim_df


# In[182]:


user_sim_df.iloc[0:15, 0:15]


# In[183]:


# Filling Diagonal values to prevent self similarity

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:15, 0:15]


# In[184]:


# Most Similar Users

user_sim_df.idxmax(axis=1)[0:15]


# In[185]:


print(user_sim_df.max(axis=1).sort_values(ascending=False).head(10))


# In[186]:


user_sim_df.sort_values((44728),ascending=False).iloc[0:5,0:15]


# In[187]:


final_ratings[(final_ratings['userid']==13552)|(final_ratings['userid']==183995)].head(10)


# In[188]:


user1 = final_ratings[(final_ratings['userid']==13552)]
user1


# In[189]:


user2 = final_ratings[(final_ratings['userid']==183995)]
user2


# In[190]:


pd.merge(user1,user2,on='booktitle',how='outer')


# In[191]:


book_read_by_user1 = list(set(user1['booktitle']))
book_read_by_user2 = list(set(user2['booktitle']))

for book_name in book_read_by_user1:
    if book_name not in book_read_by_user2:
        print("Recommendation : ", book_name)


# In[192]:


book_read_by_user1 = list(set(user1['booktitle']))
book_read_by_user2 = list(set(user2['booktitle']))

for book_name in book_read_by_user2:
     if book_name not in book_read_by_user1:
        print("Recommendation : ", book_name)  


# In[193]:


top_n = 5
most_similar_users_ids = {}

for user_id_val in user_sim_df.columns:
    
    # Sort the user IDs by similarity score in descending order
    similar_ids = user_sim_df[user_id_val].sort_values(ascending=False).index.tolist()
    
    # Remove the user's own ID from the list
    similar_ids.remove(user_id_val)
    
    # Store the top N  similar user IDs in the dictionary
    most_similar_users_ids[user_id_val] = similar_ids[:top_n]
    
most_similar_users_ids


# In[194]:


def get_top_n_similar_users(userid, topn=5):
    
    # Sort the user IDs by similarity score in descending order
    similar_ids = user_sim_df[userid].sort_values(ascending=False).index.tolist()
    
    # Remove the user's own ID from the list
    similar_ids.remove(userid)
    
    # Return the top N similar user IDs
    return similar_ids[:topn]

# Example
userid = 26535  
topn = 5  

similar_users = get_top_n_similar_users(userid, topn)

print("Top", topn, "similar users for user", userid, ":", similar_users)


# In[195]:


def get_top_rated_books_for_user(userid, topn=5):
    
    # Filter the final_ratings DataFrame for the given user
    user_ratings = final_ratings[final_ratings['userid'] == userid]
    
    # Sort the user's ratings by book rating in descending order
    user_top_rated_books = user_ratings.sort_values(by='bookrating', ascending=False).head()
    
    return user_top_rated_books

# Example
userid = 43806  
topn = 5  

users_top_rated_books = get_top_rated_books_for_user(userid, topn)

print("Users Top", topn, "rated books for user", userid, ":")
print(users_top_rated_books)


# In[196]:


def recommend_books_to_user(userid, topn=5):
    
    # Get the most similar users
    similar_users = get_top_n_similar_users(userid, topn)
    
    recommended_books = []
    
    for sim_user in similar_users:
        
        # Filter books rated by the similar user
        sim_user_ratings = final_ratings[final_ratings['userid'] == sim_user]
        
        # Find the top-rated books by the similar user
        top_rated_books = sim_user_ratings.sort_values(by='bookrating', ascending=False).head(topn)
        
        # Get the titles of the top-rated books
        new_recommendations = top_rated_books['booktitle'].tolist()
        
        # Append all new recommendations to the list
        recommended_books.extend(new_recommendations)
    
    # Remove duplicates and limit to the specified number of recommendations
    recommended_books = list(set(recommended_books))[:topn]
    
    return recommended_books

# Example
userid = 12538 
top_n = 3  

recommended_books = recommend_books_to_user(userid, topn=top_n)

print("Book recommendations for user", userid, ":", recommended_books)


# In[197]:


recommend_books_to_user(3363,7)


# # MODEL EVALUATION

# In[198]:


#First, let us evaluate the model using Precision, Recall & F1-Scores


# In[199]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

train_data, test_data = train_test_split(final_ratings, test_size=0.2, random_state=42)

# Rebuilding the pivot table for training data
train_pivot_table = train_data.pivot_table(index='userid',
                                           columns='booktitle',
                                           values='bookrating')
train_pivot_table.fillna(0, inplace=True)

# Function to get recommendations for a user based on the trained model
def recommend_books_to_user_eval(userid, topn=5):
    similar_users = get_top_n_similar_users(userid, topn)
    
    recommended_books = []
    
    for sim_user in similar_users:
        sim_user_ratings = train_data[train_data['userid'] == sim_user]
        top_rated_books = sim_user_ratings.sort_values(by='bookrating', ascending=False).head(topn)
        new_recommendations = top_rated_books['booktitle'].tolist()
        recommended_books.extend(new_recommendations)
    recommended_books = list(set(recommended_books))[:topn]
    
    return recommended_books

# Evaluate the recommendation system on the test data
precision_scores = []
recall_scores = []
f1_scores = []

for userid in test_data['userid'].unique():
    
    # Get actual books rated by the user in the test set
    actual_books = test_data[test_data['userid'] == userid]['booktitle'].tolist()
    
    # Get recommended books using the recommendation function
    recommended_books = recommend_books_to_user_eval(userid, topn=5)
    
    # Check if both actual and recommended books lists have the same length
    if len(actual_books) == len(recommended_books):
        
        # Calculate precision, recall, and F1-score
        precision = precision_score(actual_books, recommended_books, average='micro')
        recall = recall_score(actual_books, recommended_books, average='micro')
        f1 = f1_score(actual_books, recommended_books, average='micro')
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

# Calculate the average scores
average_precision = sum(precision_scores) / len(precision_scores)
average_recall = sum(recall_scores) / len(recall_scores)
average_f1 = sum(f1_scores) / len(f1_scores)

print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1-score:", average_f1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




