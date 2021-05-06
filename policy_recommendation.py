import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

policy_data=pd.read_csv("C:/erickeagle/recommendation system/policy insurance/policy.csv")

policy_data

rating_data=pd.read_csv("C:/erickeagle/recommendation system/policy insurance/ratings.csv")

rating_data=rating_data[['user_id', 'policy_id', 'rating', 'timestamp']]

Mean = rating_data.groupby(by="user_id",as_index=False)['rating'].mean()
rating_avg = pd.merge(rating_data,Mean,on='user_id')
rating_avg['avg_rating']=rating_avg['rating_x']-rating_avg['rating_y']
final=pd.pivot_table(rating_avg,values='avg_rating',index='user_id',columns='policy_id')


check_table = pd.pivot_table(rating_avg,values='rating_x',index='user_id',columns='policy_id')
final_policy = final.fillna(final.mean(axis=0))

final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)

b = cosine_similarity(final_user)
np.fill_diagonal(b, 0 )
similarity_with_user = pd.DataFrame(b,index=final_user.index)
similarity_with_user.columns=final_user.index

cosine = cosine_similarity(final_policy)
np.fill_diagonal(cosine, 0 )
similarity_with_policy =pd.DataFrame(cosine,index=final_policy.index)
similarity_with_policy.columns=final_user.index


def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df
sim_user_30_u = find_n_neighbours(similarity_with_user,30)
sim_user_30_m = find_n_neighbours(similarity_with_policy,30)


def get_user_similar_policys( user1, user2 ):
    common_policys = rating_avg[rating_avg.user_id == user1].merge(
    rating_avg[rating_avg.user_id == user2],
    on = "policy_id",
    how = "inner" )
    return common_policys.merge( policy_data, on = 'policy_id' )

a = get_user_similar_policys(1,25)

a = a.loc[ : , ['rating_x_x','rating_x_y','policy_title']]



def User_item_score(user,item):
    a = sim_user_30_m[sim_user_30_m.index==user].values
    b = a.squeeze().tolist()
    c = final_policy.loc[:,item]
    d = c[c.index.isin(b)]
    f = d[d.notnull()]
    avg_user = Mean.loc[Mean['user_id'] == user,'rating'].values[0]
    index = f.index.values.squeeze().tolist()
    corr = similarity_with_policy.loc[user,index]
    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['avg_score','correlation']
    fin['score']=fin.apply(lambda x:x['avg_score'] * x['correlation'],axis=1)
    nume = fin['score'].sum()
    deno = fin['correlation'].sum()
    final_score = avg_user + (nume/deno)
    return final_score



score = User_item_score(1,10)
print("score (u,i) is",score)


rating_avg = rating_avg.astype({"policy_id": str})
Policy_user = rating_avg.groupby(by = 'user_id')['policy_id'].apply(lambda x:','.join(x))


def User_item_score1(user):
    Policy_seen_by_user = check_table.columns[check_table[check_table.index==user].notna().any()].tolist()
    a = sim_user_30_m[sim_user_30_m.index==user].values
    b = a.squeeze().tolist()
    d = Policy_user[Policy_user.index.isin(b)]
    l = ','.join(d.values)
    Policy_seen_by_similar_users = l.split(',')
    Policys_under_consideration = list(set(Policy_seen_by_similar_users)-set(list(map(str, Policy_seen_by_user))))
    Policys_under_consideration = list(map(int, Policys_under_consideration))
    score = []
    for item in Policys_under_consideration:
        c = final_policy.loc[:,item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = Mean.loc[Mean['user_id'] == user,'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = similarity_with_policy.loc[user,index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['avg_score','correlation']
        fin['score']=fin.apply(lambda x:x['avg_score'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume/deno)
        score.append(final_score)
    data = pd.DataFrame({'policy_id':Policys_under_consideration,'score':score})
    top_5_recommendation = data.sort_values(by='score',ascending=False).head(5)
    Policy_Name = top_5_recommendation.merge(policy_data, how='inner', on='policy_id')
    Policy_Names = Policy_Name.policy_title.values.tolist()
    return Policy_Names


user = int(input("Enter the user id to whom you want to recommend : "))
predicted_policys = User_item_score1(user)

for i in predicted_policys:
    print(i)
