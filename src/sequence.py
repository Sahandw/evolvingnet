from nltk import ngrams
from log import Log
from collections import Counter
import numpy as np
from graph import Stat
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge , LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier , MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA


class Sequence:
	def __init__(self):
		self.log = Log()

	def __create_k_grams__(self , k = 5):

		kgram_set = set()
		kgram_count = {}
		s = 0
		user_seq = self.log.get_user_seq()
		for user in user_seq:
			kgrams = ngrams(user_seq[user] , k)
			l = []
			for gram in kgrams:
				if gram[0] == 'r' or gram[0] == 's' or gram[0] == 'j':
					kgram_set.add(gram)
					l.append(gram)


			kgram_count[user] = Counter(l)

		return sorted(list(kgram_set)) , kgram_count


	def __create_cumulative_design_matrix__(self):
		kgram_list , kgram_count = self.__create_k_grams__()

		n = len(kgram_count.keys())
		X = np.zeros((n , len(kgram_list)))
		y = np.zeros(n)
		st = Stat()
		pr = st.user_pagerank()[self.log.end_date]
		sorted_ids = sorted(kgram_count.keys() , key = lambda x: int(x))
		for i , user in enumerate(sorted_ids):
			for j, gram in enumerate(kgram_list):

				X[i,j] = kgram_count[user][gram]
				y[i]= pr[user]

		return X , y

	def freq_pattern(self):
		patterns = {}
		kgram_list, kgram_count = self.__create_k_grams__()
		for gram in kgram_list:
			patterns[gram] = 0
		X , y = self.__create_cumulative_design_matrix__()
		for element in X:
			for i in range(len(element)):
				patterns[kgram_list[i]] += element[i]
		print patterns

	def clean_data(self,X,y, thres = 8e-5):
		n = 0
		for i in range(X.shape[0]):
			if y[i] > thres:
				n += 1
		cleanedX = []
		cleanedy = []
		for i in range(X.shape[0]):
			if y[i] > thres:
				cleanedX.append(X[i, :])
				cleanedy.append(y[i])
		print n
		return np.array(cleanedX),np.array(cleanedy)
	def __create_training_test_set(self):
		X, y = self.__create_cumulative_design_matrix__()
		X , y = self.clean_data(X,y)
		X = self.dimentionality_reduction(X)
		return train_test_split(X,y,test_size=0.2 ,shuffle=True)


	def predict_pagerank(self):
		# X , y = self.__create_cumulative_design_matrix__()
		# print cross_val_score(model , X, y , cv = 10 , scoring='neg_mean_absolute_error' )
		X, testX , y, testy = self.__create_training_test_set()
		print X.shape , y.shape, testX.shape, testy.shape
		model = Ridge()
		#model = KernelRidge()
		model.fit(X,y)
		pred = model.predict(testX)
		print mean_absolute_error(testy, pred )
		print np.dot(testy , pred) / (np.linalg.norm(testy) * np.linalg.norm(pred))
		print np.min(y) , np.max(y)
		print stats.pearsonr(pred, testy)
		print stats.spearmanr(pred,testy)
		plt.plot([i for i in range(len(testy))], pred , 'r.')
		plt.plot([i for i in range(len(testy))], testy, 'b.')
		plt.show()


	def classification_data(self):
		X, y = self.__create_cumulative_design_matrix__()
		X, y = self.clean_data(X, y)
		per = []
		for i in range(1, 100, 25):
			per.append(np.percentile(y, i))
		for i in range(len(y)):
			if y[i] > per[len(per) - 1]:
				y[i] = len(per) - 1
				continue
			for j in range(len(per)):
				if y[i] < per[j]:
					y[i] = j
					break
		print per
		X = self.dimentionality_reduction(X)
		return train_test_split(X, y, test_size=0.2, shuffle=True)

	def dimentionality_reduction(self , X , y = None):
		model = PCA(n_components=40)
		model.fit(X)
		return model.transform(X)

	def classify_pr(self):
		X, testX, y, testy = self.classification_data()
		model = RandomForestClassifier(n_estimators=400 , max_depth= 15,
									   min_samples_split = 5)
		# model = LogisticRegression(multi_class='')
		#model = MLPClassifier((10,10,20,10))
		model.fit(X,y)
		pred = model.predict(testX)
		pred2 = model.predict(X)
		print np.mean(np.array(pred2) == np.array(y))
		print np.mean(np.array(pred) == np.array(testy))

		# xg_train = xgb.DMatrix(X, label=y)
		# xg_test = xgb.DMatrix(testX, label=testy)
		# # setup parameters for xgboost
		# param = {}
		# # use softmax multi-class classification
		# param['objective'] = 'multi:softmax'
		# # scale weight of positive examples
		# param['eta'] = 0.1
		# param['max_depth'] = 6
		# param['silent'] = 1
		# param['nthread'] = 4
		# param['num_class'] = 4
		#
		# watchlist = [(xg_train, 'train'), (xg_test, 'test')]
		# num_round = 5
		# bst = xgb.train(param, xg_train, num_round, watchlist)
		# # get prediction
		# pred = bst.predict(xg_test)
		# print np.mean(np.array(pred) == np.array(testy))



s = Sequence()
s.classify_pr()
s.predict_pagerank()