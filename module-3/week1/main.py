import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data_csv(path):
    data_indexed = pd.read_csv(path)
    return data_indexed


def rating_group(rating):
    if rating >= 7.5:
        return 'Good'
    elif rating >= 6.0:
        return 'Average'
    else:
        return 'Bad'


if __name__ == "__main__":
    data = read_data_csv('data/IMDB-Movie-Data.csv')
    data.head()
    # data.info()

    # print(data.describe())
    genre = data['Genre']
    # print(genre)
    # print(data[['Genre']])
    some_cols = data[['Title', 'Genre', 'Actors', 'Director', 'Rating']]

    # print(data.iloc[10:15]
    #       [['Title', 'Genre', 'Actors', 'Director', 'Rating']])

    search = data[(data['Year'] >= 2010) &
                  (data['Year'] <= 2015) &
                  (data['Rating'] < 6.0) &
                  (data['Revenue (Millions)'] > data['Revenue (Millions)'].quantile(0.95))]
    # print(search)

    mean_rating = data.groupby('Director')[['Rating']].mean().head()
    # print(mean_rating)
    Sorting_Operations = data.groupby('Director')[['Rating']].mean().sort_values(
        ['Rating'], ascending=False).head()

    # print(Sorting_Operations)

    # print(data.isnull().sum())

    # delete data is null
    # data.drop('Metascore', axis=1).head()
    # data.dropna()

    # Calculate the mean revenue
    revenue_mean = data['Revenue (Millions)'].mean()

    # Print the mean revenue
    # print("The mean revenue is:", revenue_mean)

    # Fill the null values in 'Revenue (Millions)' with the mean revenue
    data['Revenue (Millions)'].fillna(revenue_mean, inplace=True)
    # print(data)

    data['Rating_category'] = data['Rating'].apply(rating_group)

    # Displaying the first 5 rows with Title, Director, Rating, and Rating_category
    print(data[['Title', 'Director', 'Rating', 'Rating_category']].head(5))
