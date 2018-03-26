import matplotlib.pyplot as plt
from datetime import timedelta
from log import Log
from graph import Graph_Creator , Stat
from sequence import Sequence
import graph_tool.draw as gtdraw
import os
import numpy as np

from scipy import stats

from collections import Counter









project_folder = os.path.dirname(__file__).split("src")[0]

class Plot:
	def __init__(self):
		self.log = Log()


	def event_frequency(self, type = 'msg' , duration = 'week'):
		data = []
		interval = timedelta(7)
		if duration =='day' : interval= timedelta(1)
		event_snapshot = self.log.get_event_snapshot(duration , type)


		date = self.log.start_date
		while date <= self.log.end_date:
			data.append(len(event_snapshot[date]))
			date += interval

		# for key in sorted(event_snapshot.keys()):
		# 	print key , event_snapshot[key]
		# print len(event_snapshot.keys())

		plt.plot(range(1, len(data) + 1), data)
		#plt.ylim([0,1000])
		plt.show()


	def create_graph_visualizations(self , dur = 'week'):
		gc = Graph_Creator(dur)
		d = 'weekly'
		if dur == 'day':
			d = 'daily'
		graphs = gc.create_culumative_graphs()
		sorted_dates = sorted(graphs.iterkeys())
		for i ,date in enumerate(sorted_dates):
			pos = gtdraw.sfdp_layout(graphs[date])
			gtdraw.graph_draw(graphs[date] , pos = pos ,
					  output= project_folder + 'outputs/' + d + '/' + str(i) + '.png')


		# Can I visualize just the nodes with more than something degree for better
		# visualization?


	def join_date_final_pagerank(self):
		x = self.log.join_date_since_inception()
		st = Stat()
		pr = st.pagerank()[self.log.end_date]
		graph = st.graphs[self.log.end_date ]
		#print pr , graph
		id = graph.vertex_properties['id']
		join_since = []
		ult_pgr = []
		for v in graph.vertices():
			if v.out_degree() > 1:
				ult_pgr.append(pr[v])
				join_since.append(x[id[v]])

		#
		# join_since = join_since[1:]
		# ult_pgr = ult_pgr[1:]

		plt.plot(join_since , ult_pgr , 'r.')
		plt.show()

		# stat tests demo
		thr = max(ult_pgr) - min(ult_pgr)
		thr /= 2
		first = []
		second = []
		for i in range(len(ult_pgr)):
			if ult_pgr[i] > thr:
				first.append(join_since[i])
			else:
				second.append(join_since[i])


		# join = []
		# ult = []
		# for i in range(len(ult_pgr)):
		# 	#if ult_pgr[i] > 0.0001:
		# 	join.append(join_since[i])
		# 	ult.append(ult_pgr[i])

		# print len(join)
		# print len(join_since)
		# plt.plot(join, ult, 'r.')
		# plt.show()

		print(stats.ttest_ind(first, second, equal_var=False))
		print(stats.pearsonr(join_since,ult_pgr))
		print(stats.spearmanr(join_since,ult_pgr))


	def pr_distr(self):
		st = Stat()
		pr = st.pagerank()[self.log.end_date]
		graph = st.graphs[self.log.end_date]
		pr_list = []
		for v in graph.vertices():
			pr_list.append(float("{0:.5f}".format(pr[v])))
		count = Counter(pr_list)
		x = []
		y = []
		for key in sorted(count.iterkeys()):
			x.append(key)
			y.append(count[key])
		plt.plot(x,y, 'r.')
		plt.ylim([0,200])
		plt.show()

		plt.plot(np.log(x), np.log(y), 'r.')
		plt.ylim([0, np.log(200)])
		plt.show()


		cdf =[y[0]]
		for i in range(1 ,len(y)):
			cdf.append(cdf[i - 1] + y[i])

		plt.plot(x, cdf , 'r.')
		plt.show()

		plt.plot(np.log(x), np.log(cdf), 'r.')
		plt.show()

		plt.plot(np.log(x), 1 - np.log(np.array(cdf)), 'r.')
		plt.show()

	def join_final_date_avg_pagerank(self):
		x = self.log.join_date_since_inception()
		st = Stat()
		pr = st.pagerank()[self.log.end_date]
		graph = st.graphs[self.log.end_date]
		id = graph.vertex_properties['id']
		join_since = []
		ult_pgr = []
		for v in graph.vertices():
			if v.out_degree() > 0:
				ult_pgr.append(pr[v])
				join_since.append(x[id[v]])


		avg_pr = []
		for i in range(1 , 217):
			s = []
			for j in range(len(join_since)):
				if i == j: s.append(ult_pgr[j])
			avg_pr.append(np.average(s))

		print(stats.pearsonr([i for i in range(len(avg_pr))], avg_pr))
		print(stats.spearmanr([i for i in range(len(avg_pr))], avg_pr))

		plt.plot([i for i in range(1, 217)], avg_pr , 'r.')
		plt.show()


		avg_pr = []
		i = 0
		while i < 217:
			s = []
			for j in range(len(join_since)):
				if j >= i and j < i + 7:
					s.append(ult_pgr[j])
			avg_pr.append(np.average(s))
			i += 7


		plt.plot([i for i in range(len(avg_pr))], avg_pr, 'r.')
		plt.show()

		print(stats.pearsonr([i for i in range(len(avg_pr))], avg_pr))
		print(stats.spearmanr([i for i in range(len(avg_pr))], avg_pr))


	def deck_pagerank(self):
		x = self.log.join_date_since_inception()
		st = Stat()
		pr = st.pagerank()[self.log.end_date ]
		graph = st.graphs[self.log.end_date  ]
		id = graph.vertex_properties['id']

		pr_dict = {}
		for v in graph.vertices():
			pr_dict[int(id[v])] = pr[v]

		sorted_pr = sorted(pr_dict.values())
		sorted_id = sorted(pr_dict , key = pr_dict.get)


		samples = []
		sample_day_joined = []
		samples_ids = []

		i = 0
		while ( i < len(sorted_id)):

			idx = int(np.random.uniform() * 20)
			it = 0
			while (sorted_pr[idx + i] < 0.0001 and it < 20):
				idx = int(np.random.uniform() * 20)
				it += 1
			samples.append(sorted_pr[idx + i])
			sample_day_joined.append(x[str(sorted_id[idx + i])])
			samples_ids.append(idx + i)
			i+=20

		print(sample_day_joined)
		print(samples)
		print(samples_ids)


		plt.plot([i for i in range(len(samples))], sample_day_joined , 'r.')
		plt.show()

	def ten_top(self):
		x = self.log.join_date_since_inception()
		st = Stat()
		pr = st.pagerank()[self.log.end_date]
		graph = st.graphs[self.log.end_date]
		id = graph.vertex_properties['id']
		id_invert = {}
		for v in graph.vertices():
			id_invert[id[v]] = int(v)
		print id_invert

		pr_dict = {}
		for v in graph.vertices():
			pr_dict[int(id[v])] = pr[v]

		sorted_pr = sorted(pr_dict.values())
		sorted_id = sorted(pr_dict, key=pr_dict.get)

		K = 10

		samples_pr = sorted_pr[20:30]
		samples_id = sorted_id[20:30]
		samples_join = []
		for idx in samples_id:
			samples_join.append(x[str(idx)])

		print samples_pr
		print samples_join
		print samples_id


		pr = st.pagerank()
		date = self.log.start_date
		pr_ev = []
		date_idx = []
		for i in range(K):
			pr_ev.append([])
		while( date < self.log.end_date):
			p = pr[date]

			for i in range(K):
				try:
					pr_ev[i].append(p[str(id_invert[str(samples_id[i])])])
				except Exception:
					pr_ev[i].append(0)
			date += timedelta(14)
			date_idx.append((date - self.log.start_date).days)




		for i in range(K):
			plt.plot(date_idx , pr_ev[i])
			#print pr_ev[i]
		#plt.ylim([0,0.0002])
		plt.show()


	def ten_top_rank(self):
		x = self.log.join_date_since_inception()
		st = Stat()
		pr = st.pagerank()[self.log.end_date]
		graph = st.graphs[self.log.end_date]
		id = graph.vertex_properties['id']
		id_invert = {}
		for v in graph.vertices():
			id_invert[id[v]] = int(v)
		print id_invert

		pr_dict = {}
		for v in graph.vertices():
			pr_dict[int(id[v])] = pr[v]

		sorted_pr = sorted(pr_dict.values())
		sorted_id = sorted(pr_dict, key=pr_dict.get)

		K = 10

		# samples_pr = sorted_pr[-K:]
		# samples_id = sorted_id[-K:]
		samples_pr = sorted_pr[1800 - K: 1800]
		samples_id = sorted_id[1800 - K: 1800]
		# samples_pr = sorted_pr[1000 - K: 1000]
		# samples_id = sorted_id[1000 - K: 1000]
		# samples_pr = sorted_pr[800 - K: 800]
		# samples_id = sorted_id[800 - K: 800]

		samples_join = []
		for idx in samples_id:
			samples_join.append(x[str(idx)])


		print samples_pr
		print samples_join
		print samples_id

		rank = st.get_users_rank()
		date_idx = []
		pr_ev = []
		for i in range(K):
			pr_ev.append([])
		date = self.log.start_date
		while date < self.log.end_date:
			date_idx.append((date - self.log.start_date).days)
			for i in range(K):
				try:
					pr_ev[i].append(
						rank[date].index(str(samples_id[i])) + 1)
				except:
					pr_ev[i].append(0)
			date +=  timedelta(1)

		for i in range(K):
			l = []
			date_idx_temp = []
			for j in range(len(pr_ev[i])):
				if pr_ev[i][j] != 0:
					l.append(pr_ev[i][j])
					date_idx_temp.append(date_idx[j])
			plt.plot(date_idx_temp, l)
		plt.show()


	def msg_sent_received_user_dstr(self):
		seq = self.log.get_user_seq()
		s_count = []
		r_count = []
		for user in seq:
			c = Counter(seq[user])
			s_count.append(c['s'])
			r_count.append(c['r'])

		sc = Counter(s_count)
		rc = Counter(r_count)
		sx = []
		sfreq = []
		rx = []
		rfreq = []
		rcfreq = []
		scfreq = []
		for x,freq in sc.iteritems():
			sx.append(x)
			sfreq.append(freq)
		for x, freq in rc.iteritems():
			rx.append(x)
			rfreq.append(freq )

		plt.plot(np.log(sx),np.log(sfreq) , 'r.')
		plt.show()
		plt.plot(np.log(rx),np.log(rfreq), 'r.')
		plt.show()

		for i in range(len(sfreq)):
			if i == 0:
				scfreq.append( sfreq[i])
			else:
				scfreq.append(sfreq[i] + scfreq[i-1])

		for i in range(len(rfreq)):
			if i == 0:
				rcfreq.append(rfreq[i])
			else:
				rcfreq.append(rfreq[i] + rcfreq[i-1])

		plt.plot(np.log(sx), 1 - np.log(scfreq), 'r.')
		plt.show()
		plt.plot(np.log(rx), 1 - np.log(rcfreq), 'r.')
		plt.show()

	def pr_new_msgs_received(self):
		spcorr = {}
		corr = {}
		user_new_msg = {}
		new_weekly_msgs = self.log.get_new_messages_received()
		st = Stat()
		rank = st.get_users_rank()
		event_frequency = self.log.event_frequency('msg','week')
		for date in rank:
			wrank = []
			wnr = []
			for i, user in enumerate(rank[date]):
				if i + 1 not in user_new_msg:
					user_new_msg[i + 1] = 0
				wrank.append(i + 1)
				if user in new_weekly_msgs[date]:
					wnr.append(len(new_weekly_msgs[date][user]))
					user_new_msg[i+1] += len(new_weekly_msgs[date][user])
				else:
					wnr.append(0)
			spcorr[date] = stats.spearmanr(wnr,wrank)
			corr[date] = stats.pearsonr(wnr,wrank)



		x = []
		y = []
		for date in sorted(corr.iterkeys()):
			x.append((date - self.log.start_date).days)
			y.append(corr[date][0] )

		plt.plot(x, y, 'r.')
		plt.show()

		x = []
		y = []
		for date in sorted(spcorr.iterkeys()):
			x.append((date - self.log.start_date).days)
			y.append(spcorr[date][0])


		plt.plot(x, y, 'r.')
		plt.show()

		x ,y = [] , []
		for user in user_new_msg:
			x.append(user)
			y.append(user_new_msg[user])
		plt.scatter(x,y)
		plt.show()





	def pr_msg_received(self , type = 'r'):
		spcorr = {}
		corr = {}
		st = Stat()
		rank = st.get_users_rank()
		seq = self.log.get_user_date_seq()
		for date in seq.iterkeys():
			weekly_rank = []
			weekly_r_count = []
			for user in seq[date].iterkeys():
				if date != self.log.end_date:
					if user in rank[date]:
						weekly_rank.append(rank[date].index(user) + 1)
						weekly_r_count.append(Counter(
							seq[date + timedelta(7)][user])[type])
			if weekly_r_count != [] and weekly_rank != []:
				corr[date] = stats.pearsonr(weekly_rank , weekly_r_count)
				spcorr[date] = stats.spearmanr(weekly_rank, weekly_r_count)

		x = []
		y = []
		for date in sorted(corr.iterkeys()):
			x.append((date - self.log.start_date).days)
			y.append(corr[date][0])

		plt.plot(x,y , 'r.')
		plt.show()

		x = []
		y = []
		for date in sorted(spcorr.iterkeys()):
			x.append((date - self.log.start_date).days)
			y.append(spcorr[date][0])

		plt.plot(x, y, 'r.')
		plt.show()



	def top_ten_change_in_pr(self):
		x = self.log.join_date_since_inception()
		st = Stat()
		pr = st.pagerank()[self.log.end_date]
		graph = st.graphs[self.log.end_date]
		id = graph.vertex_properties['id']
		id_invert = {}
		for v in graph.vertices():
			id_invert[id[v]] = int(v)
		print id_invert

		pr_dict = {}
		for v in graph.vertices():
			pr_dict[int(id[v])] = pr[v]

		sorted_pr = sorted(pr_dict.values())
		sorted_id = sorted(pr_dict, key=pr_dict.get)

		K = 1899

		samples_pr = sorted_pr[:]
		samples_id = sorted_id[:]
		# samples_pr = sorted_pr[-K:]
		# samples_id = sorted_id[-K:]
		# samples_pr = sorted_pr[1800 - K: 1800]
		# samples_id = sorted_id[1800 - K: 1800]
		# samples_pr = sorted_pr[1000 - K: 1000]
		# samples_id = sorted_id[1000 - K: 1000]
		# samples_pr = sorted_pr[800 - K: 800]
		# samples_id = sorted_id[800 - K: 800]

		samples_join = []
		for idx in samples_id:
			samples_join.append(x[str(idx)])

		print samples_pr
		print samples_join
		print samples_id

		rank = st.get_users_rank()
		date_idx = []
		pr_ev = []
		pr_inc = []
		pr_dec = []
		for i in range(K):
			pr_ev.append([])
			pr_inc.append([])
			pr_dec.append([])
		date = self.log.start_date
		while date <= self.log.end_date - timedelta(7):
			date_idx.append((date - self.log.start_date).days)
			for i in range(K):
				try:
					change = -rank[date].index(str(samples_id[i])) + \
					rank[date + timedelta(7)].index(str(samples_id[i]))
					pr_ev[i].append(change)
					if change > 0:
						pr_inc[i].append(change)
						pr_dec[i].append(0)
					else:
						pr_dec[i].append(change)
						pr_inc[i].append(0)
				except:
					pr_ev[i].append(0)
					pr_dec[i].append(0)
					pr_inc[i].append(0)
			date += timedelta(7)
		for i in range(K):
			l = []
			date_idx_temp = []
			for j in range(len(pr_ev[i])):
				l.append(pr_ev[i][j])
				date_idx_temp.append(date_idx[j])
			plt.plot(date_idx_temp, l )
		plt.show()

		interval = timedelta(7)
		event_snapshot = self.log.get_event_snapshot('week', 'event')
		date = self.log.start_date
		data = []
		while date <= self.log.end_date - timedelta(7):
			data.append(len(event_snapshot[date]))
			date += interval

		prch = np.average(pr_ev , axis = 0)
		prchp = np.average(pr_inc , axis = 0)
		prchn = np.average(pr_dec , axis = 0)
		print(stats.spearmanr(prch, data))
		print(stats.pearsonr(prch, data))
		print(stats.spearmanr(prchp, data))
		print(stats.pearsonr(prchp, data))
		print(stats.spearmanr(prchn, data))
		print(stats.pearsonr(prchn, data))
		plt.plot([i for i in range(32)], prch)
		plt.show()
		plt.plot([i for i in range(32)], prchp)
		plt.show()
		plt.plot([i for i in range(32)], np.abs(prchn))
		plt.show()

	def freq_pr_corr(self):
		st = Stat()
		rank = st.get_users_rank()

		seq = Sequence()
		kgram_list , kgram_count = seq.create_weekly_sequences()

		scorr = {}
		pcorr = {}
		for kgram in kgram_list:
			scorr[kgram] = {}
			pcorr[kgram] = {}
			date = self.log.start_date
			while date <= self.log.end_date:
				lrank = []
				lfreq = []
				for i, user in enumerate(rank[date]):
					if user in kgram_count[date]:
						if kgram in kgram_count[date][user]:
							lfreq.append(kgram_count[date][user][kgram])
							lrank.append(i+1)
				if lrank is not [] and lfreq is not []:
					scorr[kgram][date] = stats.spearmanr(lfreq,lrank)
					pcorr[kgram][date] = stats.pearsonr(lfreq,lrank)


				date += timedelta(7)


		print scorr
		print pcorr

		for kgram in kgram_list:
			score = []
			dates = []
			for date in scorr[kgram]:
				if type(scorr[kgram][date][0]) == np.float64\
						and scorr[kgram][date][0] is not np.nan and \
								scorr[kgram][date][1] < 0.001:
					dates.append((date - self.log.start_date).days)
					score.append(scorr[kgram][date][0])
			if len(score) > 0:
				print(kgram)
				plt.plot(dates, score, '.')
				plt.show()



		for kgram in kgram_list:
			score = []
			dates = []
			for date in pcorr[kgram]:
				if type(pcorr[kgram][date][0]) == np.float64 \
						and pcorr[kgram][date][0] is not np.nan and\
							pcorr[kgram][date][1] < 0.001:
					dates.append((date - self.log.start_date).days)
					score.append(pcorr[kgram][date][0])
			if len(score) > 0:
				print(kgram)
				plt.plot(dates,score,'.')
				plt.show()



p = Plot()
p.pr_new_msgs_received()