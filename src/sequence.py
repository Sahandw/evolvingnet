from nltk import ngrams
from log import Log
from collections import Counter
import numpy as np
from graph import Stat
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from scipy import stats


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

		return  sorted(list(kgram_set)) , kgram_count


	def __create_cumulative_design_matrix__(self):
		kgram_list , kgram_count = self.__create_k_grams__(k = 5)
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
		kgram_list, kgram_count = self.__create_k_grams__(k=5)
		for gram in kgram_list:
			patterns[gram] = 0
		X , y = self.__create_cumulative_design_matrix__()
		for element in X:
			for i in range(len(element)):
				patterns[kgram_list[i]] += element[i]
		print patterns

	def __create_training_test_set(self):
		X, y = self.__create_cumulative_design_matrix__()
		return train_test_split(X,y,train_size=0.8 ,shuffle=False)


	def predict_pagerank(self):
		# X , y = self.__create_cumulative_design_matrix__()
		# print cross_val_score(model , X, y , cv = 10 , scoring='neg_mean_absolute_error' )
		X, testX , y, testy = self.__create_training_test_set()
		print X.shape , y.shape, testX.shape, testy.shape
		model = Ridge()
		model.fit(X,y)
		pred = model.predict(testX)
		newpred = []
		newtest = []
		for i in range(len(pred)):
			if testy[i] > 0.0001:
				newpred.append(pred[i])
				newtest.append(testy[i])
		print(len(newpred), len(newtest))

		print mean_absolute_error(testy, pred )
		print np.dot(testy , pred) / (np.linalg.norm(testy) * np.linalg.norm(pred))
		print np.min(y) , np.max(y)
		plt.plot([i for i in range(len(testy))], pred , 'r.')
		plt.plot([i for i in range(len(testy))], testy, 'b.')


		#print(stats.spearmanr(np.argsort(newpred) , np.argsort(newtest)))
		plt.show()

		print mean_absolute_error(newtest, newpred)
		print np.dot(newtest, newpred) / (np.linalg.norm(newtest) * np.linalg.norm(newpred))
		print np.min(newtest), np.max(newtest)
		plt.plot([i for i in range(len(newtest))], newpred, 'r.')
		plt.plot([i for i in range(len(newtest))], newtest, 'b.')
		plt.show()

s = Sequence()
s.predict_pagerank()