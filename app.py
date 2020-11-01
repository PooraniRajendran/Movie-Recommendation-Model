# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:31:54 2020

@author: poora
"""
import pandas as pd
from flask import Flask,render_template,request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__)

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

def similarity_matrix(dataset):
    cv=CountVectorizer()
    count_mat=cv.fit_transform(dataset['combined'])
    similarity=cosine_similarity(count_mat)
    return similarity    
    
@app.route('/recommendations', methods=['POST'])
def recommendations():
    dataset=pd.read_csv('data.csv')
    similarity=similarity_matrix(dataset)
    movie_name=request.form['movie']
    movie_name=movie_name.lower()
    
    movie_index=dataset[dataset['movie_title']==movie_name].index.values
    
    if movie_index.size > 0:
        movie_index_num=movie_index[0]
        similarity_value_index={similarity_value:similarity_index for similarity_index,similarity_value in enumerate(similarity[movie_index_num])}
    
        sorted_sim_vals=sorted(similarity_value_index.keys(),reverse=True)
    
        top_10_movies=[]
        for i in range(1,11):
            m =similarity_value_index.get(sorted_sim_vals[i])
            top_10_movies.append(dataset['movie_title'][m])
    
        return render_template('index.html',movies=top_10_movies,reply_message='Recommendations for the movie "{}" are'.format(movie_name))
    else:
        return render_template('index.html',reply_message='Sorry, Movie not available in database')

if __name__ == "__main__":
    app.run(debug=True)
    