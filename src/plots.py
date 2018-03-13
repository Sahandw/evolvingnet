import matplotlib.pyplot as plt
from datetime import timedelta
from log import Log
from graph import Graph_Creator , Stat
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
		x = self.log.join_date_since_inception()
		st = Stat()
		pr = st.pagerank()[self.log.end_date]
		graph = st.graphs[self.log.end_date]
		id = graph.vertex_properties['id']
		pr_list = []
		for v in graph.vertices():
			pr_list.append(float("{0:.5f}".format(pr[v])))
		count = Counter(pr_list)
		x = []
		y = []
		for key, item in count.iteritems():
			x.append(key)
			y.append(item)
		plt.plot(x,y, 'r.')
		plt.ylim([0,200])
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

		K = 200

		samples_pr = sorted_pr[-200:]
		samples_id = sorted_id[-200:]
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
		#plt.ylim([0,0.002])
		plt.show()






p = Plot()
p.pr_distr()

