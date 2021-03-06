import sys
import pandas as pd
import numpy as np
import os

base, test = sys.argv[1:3] #argument


def svd_factorization(matrix, k):
    from scipy.sparse.linalg import svds

    U, sigma, Vt = svds(matrix, k = k)
    sigma = np.diag(sigma)
    svd_matrix = np.dot(np.dot(U, sigma), Vt)
    svd_matrix = pd.DataFrame(svd_matrix, index = matrix.index, columns = matrix.columns)
    return svd_matrix



base_name = base
base = pd.read_csv(base, sep='\t', names=['u_id', 'i_id', 'ratings', 'ts'], header=None, index_col='i_id')
test = pd.read_csv(test, sep='\t', names=['u_id', 'i_id', 'ratings', 'ts'], header=None, index_col='u_id')

base = base.iloc[:,[0,1,2]]
test = test.iloc[:,[0,1,2]]

def makeCF(threshold, imputation, k):
    movie_user_rating = base.pivot_table('ratings', index='i_id', columns='u_id').reindex(
                                       range(1,test['i_id'].max()+1), axis='index')


    pre_use_preference = movie_user_rating.copy()
    pre_use_preference[~np.isnan(pre_use_preference)] = 1.0
    pre_use_preference.fillna(0.0, inplace = True)

    svd_pre_use_preference = svd_factorization(pre_use_preference, k)

    movie_user_rating[(np.isnan(movie_user_rating)) & (abs(svd_pre_use_preference) <=threshold)] = imputation #uninteresting

    movie_user_rating.fillna(2.0, inplace = True) # unrated item


    from sklearn.metrics.pairwise import cosine_similarity
    item_based_CF = cosine_similarity(movie_user_rating)

    item_based_collabor = pd.DataFrame(data = item_based_CF,
                                       index = movie_user_rating.index,
                                       columns = movie_user_rating.index)
    return movie_user_rating, item_based_collabor


def set_new_rating(CF, u_id, i_id, n):
    s_items = CF[i_id].sort_values(ascending=False) #similar items
    u_ratings = base[base['u_id'].isin([u_id])]
    cnts = [0,0,0,0,0]
    i = 0
    for s_item in s_items.index:
        if u_ratings.ratings.get(s_item) != None:
            cnts[u_ratings.ratings.get(s_item)-1] += n-i
            i += 1
        if i == n: #????????? ????????? n??? ?????? ??????
            break
    return cnts.index(max(cnts))+1 #?????? ??????


idx = 0
trues = 0
n = 20
k = 3
new_base, CF = makeCF(0.4, 0, k)
with open(base_name + '_prediction.txt', 'w') as f:
    for row in test.iterrows():
        #'''
        rating = set_new_rating(CF, row[0], row[1]['i_id'], n)
        
        if rating == row[1]['ratings']:
                trues += 1
        '''
        if idx <= 100:
            
            rating = set_new_rating(CF, row[0], row[1]['i_id'], n)
        
            if rating == row[1]['ratings']:
                trues += 1
            #print(row[0], row[1]['i_id'], rating, row[1]['ratings'], rating == row[1]['ratings']) 
        else:
            break
        '''
        idx += 1
        f.write(str(row[0]) + '\t' + str(row[1]['i_id']) + '\t' + str(rating) + '\n')
print(n,' accuracy : ', round(trues / idx * 100,2), '%')


