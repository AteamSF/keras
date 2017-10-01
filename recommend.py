#Recommendation Model using Keras. Backend Theano!

from __future__ import division, print_function
from theano.sandbox import cuda
import utils
from imp import reload
from utils import *
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

#path = "data/ml-20m/" 
path = "data/ml-small/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
batch_size=64

ratings = pd.read_csv(path+'ratings.csv')
ratings.head()
len(ratings)

movie_names = pd.read_csv(path+'movies.csv').set_index('movieId')['title'].to_dict()

#movie_names
users = ratings.userId.unique()
movies = ratings.movieId.unique()

#users
userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}

#userid2idx
ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])
ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])
ratings.head()

user_min, user_max, movie_min, movie_max = (ratings.userId.min(), ratings.userId.max(), ratings.movieId.min(), ratings.movieId.max())

#print(user_min, user_max, movie_min, movie_max)

np.random.seed = 42
msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]

g=ratings.groupby('userId')['rating'].count()
topUsers=g.sort_values(ascending=False)[:15]

g=ratings.groupby('movieId')['rating'].count()
topMovies=g.sort_values(ascending=False)[:15]

top_r = ratings.join(topUsers, rsuffix='_r', how='inner', on='userId')
top_r = top_r.join(topMovies, rsuffix='_r', how='inner', on='movieId')

#print(pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum))

def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)

n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()

#print(n_users, n_movies)

n_factors = 50

user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)

#print(m)

x = merge([u, m], mode='concat')
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(70, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(1)(x)
nn = Model([user_in, movie_in], x)
nn.compile(Adam(0.001), loss='mse')

print(nn.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=1, validation_data=([val.userId, val.movieId], val.rating)))

pre = nn.predict([trn.userId, trn.movieId])

#type(pre)
pre[:10]

#type(trn)
trn[:10]
