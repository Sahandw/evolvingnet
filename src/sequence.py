from nltk import ngrams
from log import Log
from collections import Counter

class Sequence:
	def __init__(self):
		self.log = Log()

	def create_k_grams(self , n = 5):
		kgram_count = {}
		user_seq = self.log.get_user_seq()
		for user in user_seq:
			fivegrams = ngrams(user_seq[user] , n)
			l = []
			for grams in fivegrams:
				l.append(grams)
			kgram_count[user] = Counter(l)
		print kgram_count['2']



s = Sequence()
s.create_k_grams()