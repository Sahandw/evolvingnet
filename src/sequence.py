from nltk import ngrams
from log import Log
from collections import Counter
import numpy as np
from graph import Stat
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge , LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , explained_variance_score, confusion_matrix
from scipy import stats
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPClassifier , MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from datetime import timedelta
from sklearn.neighbors import KNeighborsClassifier


from scipy import stats
#import tensorly as tl
#from tensorly.decomposition import tucker ,parafac,partial_tucker
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


	def __create_cumulative_design_matrix__(self , k = 5):
		kgram_list , kgram_count = self.create_k_grams(k)

		n = len(kgram_count.keys())
		X = np.zeros((n , len(kgram_list)))
		y = np.zeros(n)
		st = Stat()
		pr = st.user_pagerank()[self.log.end_date]
		sorted_ids = sorted(kgram_count.keys() , key = lambda x: int(x))
		for i , user in enumerate(sorted_ids):
			for j, gram in enumerate(kgram_list):
				if user not in pr:
					print user
					continue
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

	def clean_data(self,X,y, thres = 9e-5):
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
	def __create_training_test_set(self, X = None , y = None , k =5):
		if X is None and y is None:
			X, y = self.__create_cumulative_design_matrix__(k)
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

	def classification_data(self ,  X = None , y = None , k = 5 ):
		if X is None and y is None:
			X, y = self.__create_cumulative_design_matrix__(k)
		X, y = self.clean_data(X, y)
		per = []
		for i in range(15, 100, 15):
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

	def dimentionality_reduction(self , X , testX = []):
		model = PCA(n_components=10)
		model.fit(X)
		if testX == []:
			return model.transform(X)
		return model.transform(X) , model.transform(testX)


	def classify_pr(self , k = 5):
		X, testX, y, testy = self.classification_data(k=k)
		sumX, sumtestX = np.sum(X,axis=1).reshape(-1,1),\
						 np.sum(testX,axis = 1).reshape(-1,1)
		#X , testX = self.dimentionality_reduction(X,testX)
		model = RandomForestClassifier(n_estimators=500 ,max_depth=30
									   , min_samples_split=7)

		X, y = np.concatenate((X,testX) , axis= 0),\
				np.concatenate((y,testy) , axis = 0)
		sX  = np.concatenate((sumX,sumtestX) , axis=0 )

		error =  cross_val_score(model, X, y ,cv= 4, scoring='neg_mean_absolute_error').mean()
		print error
		# print cross_val_score(model, X,y , cv = 4 , scoring='f1_weighted')
		# print cross_val_score(model, sX, y , cv = 4 , scoring= 'neg_mean_absolute_error').mean()
		# print cross_val_score(model, sX,y , cv = 4 , scoring='f1_weighted')

		return - error

		#model = MLPClassifier((100,))
		# model.fit(X,y)
		# pred = model.predict(testX)
		# pred2 = model.predict(X)
		# print np.mean(np.array(pred2) == np.array(y))
		# print np.mean(np.array(pred) == np.array(testy))
		# print np.mean(np.abs(np.array(pred) - np.array(testy)))
		# print confusion_matrix(pred , testy)
		# miss = 0
		# for i in range(len(pred)):
		# 	miss += np.abs( pred[i] - testy[i])
		# print  miss
		# model.fit(sumX,y)
		# pred = model.predict(sumtestX)
		# pred2 = model.predict(sumX)
		# print np.mean(np.array(pred2) == np.array(y))
		# print np.mean(np.array(pred) == np.array(testy))
		# print np.mean(np.abs(np.array(pred) - np.array(testy)))
		# print confusion_matrix(pred, testy)
		# miss = 0
		# for i in range(len(pred)):
		# 	miss += np.abs(pred[i] - testy[i])
		# print  miss


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

		return dcg_score(testy,pred) ,\
			   explained_variance_score(testy,pred)

	def kendall(self , model,  testX , testy):
		pred = model.predict(testX)
		return stats.kendalltau(pred ,testy)[0]


	def cumulative_prediction(self,start , end):
		kgram_list, freq = self.kgram_list,self.kgram_count
		prank = self.user_pagerank()
		N = len(prank[end]) # num of users
		P = len(kgram_list) # number of patterns
		X = np.zeros((N,P))
		y = np.zeros(N)
		final_pr = prank[self.log.end_date]
		for i in range(N):
			if str(i + 1) in final_pr:
				y[i] = final_pr[str(i + 1)]
		for date in freq:
			if date <= end and date >= start:
				for user in freq[date]:
					id = int(user) - 1
					for p in freq[date][user]:
						pd = kgram_list.index(p)
						X[id , pd] += freq[date][user][p]
		X,testX,y,testy = self.__create_training_test_set(X,y)
		wholeX, wholey = np.concatenate((X,testX) , axis = 0)\
			, np.concatenate((y, testy) , axis = 0)
		wholeX , wholey = self.clean_data(wholeX, wholey)
		sumwholeX = np.sum(wholeX, axis = 1).reshape(-1,1)
		wholeX = self.dimentionality_reduction(wholeX)
		#model = Ridge(alpha= 1.5)
		# model = KernelRidge(kernel='rbf', degree=10)
		model = RandomForestRegressor(n_estimators=100)
		# return stats.spearmanr(pred,testy)[0],\
		# 	   explained_variance_score(pred,testy)
		if X.shape[0]< 10: return 0 , 0
		# return stats.kendalltau(pred2 , testy ) [0],\
		# 		stats.kendalltau(pred,testy)[0]

		return cross_val_score(model , wholeX,wholey , cv = 5
							   , scoring=self.kendall ).mean() ,\
			cross_val_score(model, sumwholeX , wholey , cv = 5,
							scoring=self.kendall).mean()

		# return stats.spearmanr(pred,testy)[0],\
		# 		stats.kendalltau(pred,testy)[0]

	def cumulative_classification(self, start , end):
		kgram_list, freq = self.kgram_list, self.kgram_count
		prank = self.user_pagerank()
		N = len(prank[end])  # num of users
		P = len(kgram_list)  # number of patterns
		X = np.zeros((N, P))
		y = np.zeros(N)
		final_pr = prank[self.log.end_date]
		for i in range(N):
			if str(i+1) in final_pr:
				y[i] = final_pr[str(i + 1)]
		for date in freq:
			if date <= end and date >= start:
				for user in freq[date]:
					id = int(user) - 1
					for p in freq[date][user]:
						pd = kgram_list.index(p)
						X[id, pd] += freq[date][user][p]

		X,testX,y,testy = self.classification_data(X,y)
		sumX, sumtestX = np.sum(X, axis=1).reshape(-1, 1), \
						 np.sum(testX, axis=1).reshape(-1, 1)
		# X , testX = self.dimentionality_reduction(X,testX)
		model = RandomForestClassifier(n_estimators=500, max_depth=20
									   , min_samples_split=7)

		X, y = np.concatenate((X, testX), axis=0), \
			   np.concatenate((y, testy), axis=0)
		sX = np.concatenate((sumX, sumtestX), axis=0)

		if X.shape[0] < 10: return 0,0
		return -cross_val_score(model, X, y, cv=4,
					scoring='neg_mean_absolute_error').mean(), \
			   -cross_val_score(model, sX, y, cv=4,
					scoring='neg_mean_absolute_error').mean()


#
# # # cumulative prediction
def run_prediction():
	freq = s.log.event_frequency('msg','week')[1:]
	s.create_weekly_sequences()
	l = []
	k = []
	d = []
	idx = 0
	date = s.log.start_date
	while date < s.log.end_date :
		r = s.cumulative_prediction(date,date + timedelta(7))
		l.append(r[0])
		k.append(r[1])
		d.append((date - s.log.start_date).days / 7)
		print date ,r[0],r[1]
		date += timedelta(7)
		idx += 1

	plt.plot(d,l,'r')
	plt.plot(d,k,'g')
	plt.show()

	print stats.spearmanr(freq, l)


#
# cumulative classification
def run_classification( c = 0):
	s.create_weekly_sequences(k = 3)
	freq = s.log.event_frequency('msg','week')
	l = []
	k = []
	d = []
	date = s.log.start_date
	i = 0
	while date < s.log.end_date:
		if c == 0:
			r = s.cumulative_classification(date,date+timedelta(7))
		if c == 1:
			r = s.cumulative_classification(s.log.start_date \
											+ timedelta(7), date + timedelta(7))
		l.append(r[0])
		k.append(r[1])
		d.append((date - s.log.start_date).days / 7)
		print date ,r[0],r[1]  , freq[i]
		i += 1
		date += timedelta(7)

	num_of_zeros = 0
	for el in l:
		if el == 0:
			num_of_zeros += 1
	l = l[num_of_zeros:]
	d = d[num_of_zeros:]
	k = k[num_of_zeros:]

	plt.plot(d,l , 'r')
	plt.ylabel('Absolute mean error')
	plt.xlabel('Week since inception')
	plt.plot(d,k, 'g')
	plt.show()



s = Sequence()
# run_prediction()
# run_classification(c = 1)
# s.predict_pagerank()
#s.classify_pr(k = 3)

def effect_of_k():
	d = []
	for k in range(1,12):
		d.append(s.classify_pr(k = k))

	plt.plot([i for i in range(1,12)] , d , 'b')
	plt.margins(0.01)
	plt.ylabel('Absolute Mean Error')
	plt.xlabel('Different values of K for K-grams')
	plt.show()

# effect_of_k()
