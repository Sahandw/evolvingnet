from nltk import ngrams
from log import Log
from collections import Counter

class Sequence:
	def __init__(self):
		self.log = Log()

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

		return  kgram_set , kgram_count









s = Sequence()
print s.create_k_grams()