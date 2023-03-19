#Building a simple book recommender system using the following csv dataset
#http://www2.informatik.uni-freiburg.de/~cziegler/BX/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

books = pd.read_csv(r'C:\\Users\HH\Downloads\BX-CSV-Dump (3)\BX-Books.csv',low_memory=False, sep=';', on_bad_lines='skip', encoding="latin-1")
books.columns = ['ISBN', 'BookTitle', 'BookAuthor', 'YearOfPublication', 'Publisher', 'ImageUrlS', 'ImageUrlM', 'ImageUrlL']

users = pd.read_csv(r'C:\\Users\HH\Downloads\BX-CSV-Dump (3)\BX-Users.csv', sep=';', on_bad_lines='skip', encoding="latin-1")
users.columns = ['UserID', 'Location', 'Age']

ratings = pd.read_csv(r'C:\\Users\HH\Downloads\BX-CSV-Dump (3)\BX-Book-Ratings.csv', sep=';', on_bad_lines='skip', encoding="latin-1")
ratings.columns = ['UserID', 'ISBN', 'BookRating']



#The books dataset
print(books.nunique())
print(books.shape)
print(list(books.columns))
#Dropping URLs columns that don't seem to be important
books.drop(['ImageUrlS', 'ImageUrlM', 'ImageUrlL'],axis=1,inplace=True)
pd.set_option('display.max_columns', None)
print(books.head())
print('\n')
#Looking for duplicates
duplicated=books.duplicated().sum()
print('Total duplicated rows=',duplicated)
print('\n')
print(books.dtypes)
print('\n')

print(books.YearOfPublication.unique())
print('\n')
#Publishers 'DK Publishing Inc' and 'Gallimard' incorrectly listed under YearOfPublication
#Making corrections and setting the datatype for YearOfPublication as int
with pd.option_context("display.max_colwidth", None):
    print(books.loc[books.YearOfPublication=='DK Publishing Inc',:])
books.loc[books.ISBN=='078946697X','YearOfPublication']=2000
books.loc[books.ISBN=='078946697X','BookAuthor']='Michael Teitelbaum'
books.loc[books.ISBN=='078946697X','Publisher']='DK Publishing Inc'
books.loc[books.ISBN=='078946697X','BookTitle']='DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'

books.loc[books.ISBN=='0789466953','YearOfPublication']=2000
books.loc[books.ISBN=='0789466953','BookAuthor']='James Buckley'
books.loc[books.ISBN=='0789466953','Publisher']='DK Publishing Inc'
books.loc[books.ISBN=='0789466953','BookTitle']='DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
print('\n')

#Rechecking my corrections
with pd.option_context("display.max_colwidth", None):
    print(books.loc[books.ISBN=='078946697X',:])
    print('\n')
    print(books.loc[books.ISBN=='0789466953',:])
print('\n')

with pd.option_context("display.max_colwidth", None):
    print(books.loc[books.YearOfPublication=='Gallimard',:])
books.loc[books.ISBN=='2070426769','YearOfPublication']=2003
books.loc[books.ISBN=='2070426769','BookAuthor']='Jean-Marie Gustave Le Clézio'
books.loc[books.ISBN=='2070426769','Publisher']='Gallimard'
books.loc[books.ISBN=='2070426769','BookTitle']='Peuple du ciel, suivi de Les Bergers'         
print('\n')

#Rechecking my corrections
print(books.loc[books.ISBN=='2070426769',:])
print('\n')
 
#Correcting the dtype of YearOfPublication
books.YearOfPublication=pd.to_numeric(books.YearOfPublication)
print(books.dtypes)
print('\n')



#The Users dataset
print(users.nunique())
print(users.shape)
print(list(users.columns))
print(users.head())
print('\n')
#Looking for duplicates
duplicated=users.duplicated().sum()
print('Total duplicated rows=',duplicated)
print('\n')
print(users.dtypes)
print('\n')

#Checking the age distribution
users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('system2.png', bbox_inches='tight')
plt.show()
#Most of the reviewers are in their 20's and 30's


#The Ratings dataset
print(ratings.nunique())
print(ratings.shape)
print(list(ratings.columns))
print(ratings.head())
print('\n')
#Looking for duplicates
duplicated=ratings.duplicated().sum()
print('Total duplicated rows=',duplicated)
print('\n')


#Checking the rating distribution
plt.rc("font", size=10)
ratings.BookRating.value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('system1.png', bbox_inches='tight')
plt.show()


#Separating explicit ratings represented by 1-10 from implicit ratings represented by 0
#Using only explicit ratings
ratings_explicit=ratings[ratings.BookRating !=0]
ratings_implicit=ratings[ratings.BookRating ==0]

#Checking shapes
print(ratings.shape)
print(ratings_explicit.shape)
print(ratings_implicit.shape)
print('\n')


sns.countplot(data=ratings_explicit, x='BookRating')
plt.show()
#Higher ratings are more common. Rating 8 is represented the most often

#Popularity based Recommendation system
#Based on book rating counts for different books 
rating_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['BookRating'].count())
sorted=rating_count.sort_values('BookRating', ascending=False).head(10)
top10=sorted.merge(books,left_index=True,right_on='ISBN')
print(top10)
pd.set_option('display.max_columns', None)
print('Top10 recommended books:\n', top10)


print('********************************')


#Collaborative Filtering using k-Nearest Neighbors (kNN) to find clusters of similar users based on common book ratings
#Making predictions based on the average rating of top k nearest neighbors
#Looking for the popular books from the original dataset
combined_book_rating = pd.merge(ratings_explicit, books, on='ISBN')
print(combined_book_rating.head())
print('\n')

#Creating a new column (TotalRatingCount)using groupby by book ISBN
combined_book_rating = combined_book_rating.dropna(axis = 0, subset = ['ISBN'])

book_ratingCount =(combined_book_rating.groupby(by =['ISBN'])['BookRating'].count().reset_index().rename(columns ={'BookRating':'TotalRatingCount'})[['ISBN','TotalRatingCount']])
#book_ratingCount =(combined_book_rating.groupby(by =['BookTitle'])['BookRating'].count().reset_index().rename(columns ={'BookRating':'TotalRatingCount'})[['BookTitle','TotalRatingCount']])
print(book_ratingCount.head())
print('\n')

#Merge to find out which books are popular 
rating_with_totalRatingCount = combined_book_rating.merge(book_ratingCount, left_on = 'ISBN', right_on = 'ISBN', how = 'left')
print(rating_with_totalRatingCount.head())
print('\n')

#Statistics
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_ratingCount['TotalRatingCount'].describe())
print('\n')
print(book_ratingCount['TotalRatingCount'].quantile(np.arange(.9, 1, .01)))
print('\n')

#Approximately 1% of the books received 24 or more ratings.
#271 360 books in the dataset can be limited to the top 1%, i.e. 2713 unique books
popularity_threshold = 24
rating_popular_book = rating_with_totalRatingCount.query('TotalRatingCount >= @popularity_threshold')
rating_popular_book.head()
print('\n')

#Filtering only to the users in the US
##Merging user data with the rating data and total rating count data
combined = rating_popular_book.merge(users, left_on = 'UserID', right_on = 'UserID', how = 'left')

us_user_rating = combined[combined['Location'].str.contains("usa")]
us_user_rating=us_user_rating.drop('Age', axis=1)
print(us_user_rating.head())
print('\n')

#Implementing kNN
#Fitting the model
from scipy.sparse import csr_matrix

#Converting to a 2D matrix. Filling the missing values with zeros
#Transforming the ratings of the matrix into a scipy sparse matrix
us_user_rating = us_user_rating.drop_duplicates(['UserID', 'BookTitle'])
us_user_rating_pivot = us_user_rating.pivot(index = 'BookTitle', columns = 'UserID', values = 'BookRating').fillna(0)
us_user_rating_matrix = csr_matrix(us_user_rating_pivot.values)

from sklearn.neighbors import NearestNeighbors

#Algorithm computing the nearest neighbors will calculate the cosine similarity between rating vectors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_user_rating_matrix)

#Testing the model
#Measuring distance between instances. Classifying an instance by finding its nearest neighbors
#Picking the most popular class among the neighbors
query_index = np.random.choice(us_user_rating_pivot.shape[0])
distances,indices = model_knn.kneighbors(us_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(us_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, us_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))

print('\n')




