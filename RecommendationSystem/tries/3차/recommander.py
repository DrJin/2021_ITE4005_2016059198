import sys
import pandas as pd
import numpy as np
import os

base, test = sys.argv[1:3] #argument


def svd_factorization(matrix):
    mean = np.mean(matrix.to_numpy(), axis = 1)
    matrix_mean = matrix - mean.reshape(-1,1)
    from scipy.sparse.linalg import svds

    U, sigma, Vt = svds(matrix, k = 12)
    sigma = np.diag(sigma)
    svd_matrix = np.dot(np.dot(U, sigma), Vt) + mean.reshape(-1,1)
    svd_matrix = pd.DataFrame(svd_matrix, index = matrix.index, columns = matrix.columns)
    return svd_matrix



base_name = base
base = pd.read_csv(base, sep='\t', names=['u_id', 'i_id', 'ratings', 'ts'], header=None, index_col='i_id')
test = pd.read_csv(test, sep='\t', names=['u_id', 'i_id', 'ratings', 'ts'], header=None, index_col='u_id')

base = base.iloc[:,[0,1,2]]
test = test.iloc[:,[0,1,2]]

movie_user_rating = base.pivot_table('ratings', index='i_id', columns='u_id').reindex(
                                       index=range(1,test['i_id'].max()+1),
                                       columns=range(1,test['i_id'].max()+1))


#print(movie_user_rating)
#svd_movie_user_rating = svd_factorization(movie_user_rating)
#print(svd_movie_user_rating)

pre_use_preference = movie_user_rating.copy()
pre_use_preference[~np.isnan(pre_use_preference)] = 1.0
pre_use_preference.fillna(0.0, inplace = True)

#print(movie_user_rating)

svd_pre_use_preference = svd_factorization(pre_use_preference)
#print(svd_pre_use_preference)

movie_user_rating[(np.isnan(movie_user_rating)) & (abs(svd_pre_use_preference) <=0.1)] = 1.0 #uninteresting

#print(movie_user_rating)

movie_user_rating.fillna(2.0, inplace = True) # unrated item

#print(movie_user_rating)


from sklearn.metrics.pairwise import cosine_similarity
item_based_CF = cosine_similarity(movie_user_rating)

#print(item_based_CF)

item_based_collabor = pd.DataFrame(data = item_based_CF,
                                   index = movie_user_rating.index,
                                   columns = movie_user_rating.index)

#print(item_based_collabor)

def set_new_rating(u_id, i_id, n):
    s_items = item_based_collabor[i_id].sort_values(ascending=False) #similar items
    #print(s_items[1:6].index.values)
    u_ratings = base[base['u_id'].isin([u_id])]
    cnts = [0,0,0,0,0]
    for s_item in s_items.index:
        if u_ratings.ratings.get(s_item) != None:
            #print('u_id',str(u_id),'i_id : ', str(i_id), 'ratings : ', str(u_ratings.ratings.get(s_item)))
            #return u_ratings.ratings.get(s_item)
            cnts[u_ratings.ratings.get(s_item)-1] += 1
        if sum(cnts) == n: #유사작 상위권 n개 작품 참고
            break
    #print(cnts)
    return cnts.index(max(cnts))+1 #최대 개수
    #return  sum([cnts[i] * (i+1) for i in range(5)]) // sum(cnts)# 단순 평균
    #return round(np.dot(np.array(range(1,6)), np.array(cnts)) / n) #가중 평균

'''   
idx = 0
for row in test.iterrows():
    if idx == 30:
        break
    #print(row[0], row[1]['i_id'], row[1]['ratings'])
    print(row[0], row[1]['i_id'], set_new_rating(row[0], row[1]['i_id']), row[1]['ratings'],
          set_new_rating(row[0], row[1]['i_id']) == row[1]['ratings'])
    idx += 1
'''


idx = 0
trues = 0
with open(base_name + '_prediction.txt', 'w') as f:
    for row in test.iterrows():
        n = 100
        rating = set_new_rating(row[0], row[1]['i_id'], n)
        
        if rating == row[1]['ratings']:
                trues += 1
        '''
        if idx <= 50:
            print(row[0], row[1]['i_id'], rating, row[1]['ratings'],
                  rating == row[1]['ratings']) 
        #else:
        #    break
        '''
        idx += 1
        f.write(str(row[0]) + '\t' + str(row[1]['i_id']) + '\t' + str(rating) + '\n')
print(n,' accuracy : ', trues / idx * 100, '%')
#print(test.iloc[0])
#print(test.iloc[0].name)



