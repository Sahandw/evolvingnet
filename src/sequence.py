from nltk import ngrams
from log import Log
from collections import Counter
import numpy as np
from graph import Stat
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge , LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error ,explained_variance_score
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier , MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from datetime import timedelta

from scipy import stats
import tensorly as tl
from tensorly.decomposition import tucker ,parafac,partial_tucker
import os
import pickle
project_folder = os.path.dirname(__file__).split("src")[0]


class Sequence:
	def __init__(self):
		self.log = Log()

	def user_pagerank(self):
		return pickle.load(open(project_folder \
					+ 'data/pickles/weekly_pr.p', 'rb'))

	def get_weekly_ranks(self):
		return pickle.load(open(project_folder \
								+ 'data/pickles/weekly_ranks.p', 'rb'))
	def create_k_grams(self , k = 5):

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
		kgram_list , kgram_count = self.create_k_grams()

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
		kgram_list, kgram_count = self.create_k_grams()
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
	def __create_training_test_set(self, X = None , y = None):
		if X is None and y is None:
			X, y = self.__create_cumulative_design_matrix__()
		X , y = self.clean_data(X,y)
		return train_test_split(X,y,test_size=0.2 ,shuffle=True)


	def predict_pagerank(self):
		# X , y = self.__create_cumulative_design_matrix__()
		# print cross_val_score(model , X, y , cv = 10 , scoring='neg_mean_absolute_error' )
		X, testX , y, testy = self.__create_training_test_set()
		X, testX = self.dimentionality_reduction(X, testX)
		print X.shape , y.shape, testX.shape, testy.shape
		model = Ridge()
		#model = KernelRidge(kernel='rbf')
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
		print stats.pearsonr(pred, testy)
		print stats.spearmanr(pred,testy)
		print 'ex' , explained_variance_score(pred,testy)
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


		plt.plot([i for i in range (len(testy))], stats.rankdata(pred,'min')
				 					- stats.rankdata(testy,'min') )
		print stats.spearmanr(stats.rankdata(pred,'min') , stats.rankdata(testy,'min'))
		plt.show()

		print mean_absolute_error(stats.rankdata(pred,'min'), stats.rankdata(testy,'min'))

	def classification_data(self ,  X = None , y = None):
		if X is None and y is None:
			X, y = self.__create_cumulative_design_matrix__()
		X, y = self.clean_data(X, y)
		per = []
		for i in range(1, 100, 25):
			per.append(np.percentile(y, i))
		for i in range(len(y)):
			if y[i] >= per[len(per) - 1]:
				y[i] = len(per) - 1
				continue
			for j in range(len(per)):
				if y[i] < per[j]:
					y[i] = j
					break
		print per
		return train_test_split(X, y, test_size=0.2, shuffle=True)

	def dimentionality_reduction(self , X , testX):
		model = PCA(n_components=40)
		model.fit(X)
		return model.transform(X) , model.transform(testX)

	def classify_pr(self):
		X, testX, y, testy = self.classification_data()
		X , testX = self.dimentionality_reduction(X,testX)
		model = RandomForestClassifier(n_estimators=400 , max_depth= 15,
									   min_samples_split=20)
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


	def create_weekly_sequences(self , k = 5):
		seq = self.log.get_user_date_seq()
		kgram_set = set()
		kgram_count = {}
		for date , user_seq in seq.iteritems():
			kgram_count[date] = {}
			for user in user_seq:
				kgrams = ngrams(user_seq[user] , k)
				l = []
				for gram in kgrams:
					if (gram[0] == 'r' or gram[0] == 's' or gram[0] == 'j' ) \
							and 'e' not in gram:
						kgram_set.add(gram)
						l.append(gram)
				if l != []:
					kgram_count[date][user] = Counter(l)

		pickle.dump(sorted(list(kgram_set)), open(project_folder \
					+ 'data/pickles/kgram_list'+str(k)+'.p', 'wb'))
		pickle.dump(kgram_count, open(project_folder \
				+ 'data/pickles/kgram_count'+str(k)+'.p', 'wb'))
		self.kgram_count = kgram_count
		self.kgram_list = sorted(list(kgram_set))
		return sorted(list(kgram_set)) , kgram_count

	def get_weekly_seq(self , k = 5):
		return pickle.load(open(project_folder \
			+ 'data/pickles/kgram_list' + str(k) + '.p', 'wb')) , \
			pickle.load(open(project_folder \
				+ 'data/pickles/kgram_count' + str(k) + '.p', 'wb'))

	def date_week(self ,date, start):
		return (date - start).days/7

	def create_tensor(self , start , end):
		prank = self.get_weekly_ranks()
		kgram_list, freq = self.create_weekly_sequences()
		N = len(prank[end]) # num of users
		P = len(kgram_list) # num of kgram patterns
		W = ((end - start).days)/7 + 1
		t = np.zeros((N, P , W))
		for date in freq:
			if date >= start and date <= end:
				dw = self.date_week(date,start)
				for user in freq[date]:
					id = int(user) - 1
					for p in freq[date][user]:
						pd = kgram_list.index(p)
						t[id,pd,dw] = freq[date][user][p]
		return tl.tensor(t)

	def decompose(self , tensor):
		return parafac(tensor,rank = 30)

	def td_create_dataset(self,tensor ):
		X = tl.to_numpy(self.decompose(tensor)[0])
		pr = self.user_pagerank()[self.log.end_date]
		y = np.zeros(X.shape[0])
		for i in range(X.shape[0]):
			y[i] = pr[str(i + 1)]

		X,testX,y,testy = train_test_split(X,y)
		model = Ridge()
		model.fit(X,y)
		pred_train = model.predict(X)
		pred = model.predict(testX)

		print stats.spearmanr(y,pred_train)
		print stats.pearsonr(y,pred_train)

		print stats.spearmanr(testy,pred)
		print stats.pearsonr(testy,pred)

		print mean_absolute_error(testy,pred)
		print mean_absolute_error(y , pred_train)

		print explained_variance_score(testy,pred)
		print explained_variance_score(y, pred_train)

		return stats.spearmanr(testy,pred) ,\
			   explained_variance_score(testy,pred)


	def cumulative_prediction(self,start , end):
		kgram_list, freq = self.kgram_list,self.kgram_count
		prank = self.user_pagerank()
		N = len(prank[end]) # num of users
		P = len(kgram_list) # number of patterns
		X = np.zeros((N,P))
		y = np.zeros(N)
		final_pr = prank[self.log.end_date]
		for i in range(N):
			y[i] = final_pr[str(i + 1)]
		for date in freq:
			if date <= end and date >= start:
				for user in freq[date]:
					id = int(user) - 1
					for p in freq[date][user]:
						pd = kgram_list.index(p)
						X[id , pd] += freq[date][user][p]
		X,testX,y,testy = self.__create_training_test_set(X,y)
		model = Ridge()
		model.fit(X,y)
		pred = model.predict(testX)
		return stats.spearmanr(pred,testy)[0],\
			   explained_variance_score(pred,testy)



	def cumulative_classification(self, start , end):
		kgram_list, freq = self.kgram_list, self.kgram_count
		prank = self.user_pagerank()
		N = len(prank[end])  # num of users
		P = len(kgram_list)  # number of patterns
		X = np.zeros((N, P))
		y = np.zeros(N)
		final_pr = prank[self.log.end_date]
		for i in range(N):
			y[i] = final_pr[str(i + 1)]
		for date in freq:
			if date <= end and date >= start:
				for user in freq[date]:
					id = int(user) - 1
					for p in freq[date][user]:
						pd = kgram_list.index(p)
						X[id, pd] += freq[date][user][p]

		X,testX,y,testy = self.classification_data(X,y)
		for l in y:
			if l!= 0 and l!= 1 and l !=2 and l!= 3:
				print l
		X, testX = self.dimentionality_reduction(X, testX)
		model = RandomForestClassifier(n_estimators=400, max_depth=15,
									   min_samples_split=20)
		# model = LogisticRegression(multi_class='')
		# model = MLPClassifier((10,10,20,10))
		model.fit(X, y)
		pred = model.predict(testX)
		pred2 = model.predict(X)
		return np.mean(np.array(pred2) == np.array(y))\
			, np.mean(np.array(pred) == np.array(testy))



#
# s = Sequence()
# s.predict_pagerank()
# s.classify_pr()




# #
# s = Sequence()
# # print s.create_weekly_sequences()[1]\
# # 			[s.log.start_date + timedelta(7 * 12)]['1']
#
# # cumulative prediction
# freq = s.log.event_frequency('msg','week')
# s.create_weekly_sequences()
# l = []
# k = []
# d = []
# idx = 0
# date = s.log.start_date
# while date < s.log.end_date :
# 	r = s.cumulative_prediction(date,date + timedelta(7))
# 	l.append(r[0])
# 	k.append(r[1])
# 	d.append((date - s.log.start_date).days / 7)
# 	print date ,r[0],r[1]
# 	date += timedelta(7)
# 	idx += 1
#
# plt.plot(d,l)
# plt.ylim([0,0.002])
# plt.show()
#
# plt.plot(d,k)
# plt.ylim([0,0.001])
# plt.show()
#
#
#
#
# # cumulative classification
#
# s.create_weekly_sequences()
# freq = s.log.event_frequency('msg','week')
# l = []
# k = []
# d = []
# date = s.log.start_date
# while date < s.log.end_date:
# 	r = s.cumulative_classification(date,date+timedelta(7))
# 	l.append(r[0])
# 	k.append(r[1])
# 	d.append((date - s.log.start_date).days / 7)
# 	print date ,r[0],r[1]
# 	date += timedelta(7)
#
# print freq
# print d
#
# plt.plot(d,l)
# plt.ylim([0.3,1])
# plt.show()
#
# plt.plot(d,k)
# plt.ylim([0.3,1])
# plt.show()
#








# t = s.create_tensor(s.log.start_date + timedelta(0),
# 					s.log.end_date - timedelta(7*25) )
# s.td_create_dataset(t)

# d = []
# l = []
# k = []
# date = s.log.start_date + timedelta(7*5)
# while date <= s.log.end_date:
# 	t = s.create_tensor(s.log.start_date , date)
# 	res = s.td_create_dataset(t)
# 	l.append(res[0][0])
# 	k.append(res[1])
# 	d.append((date - s.log.start_date).days)
# 	date += timedelta(14)
#
#
# plt.plot(d,l)
# plt.show()
#
# plt.plot(d,k)
# plt.show()
#
#
# plt.plot(d)

# t = s.create_tensor(s.log.start_date,s.log.end_date)
# s.td_create_dataset(t)