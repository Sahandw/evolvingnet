from graph_tool.all import *
import graph_tool as gt
import graph_tool.draw as gtdraw
import graph_tool.centrality as gtc
from log import Log
from datetime import timedelta
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
project_folder = os.path.dirname(__file__).split("src")[0]



class Graph_Creator:
	def __init__(self , duration = 'week'):
		log = Log()
		self.msgs = log.get_event_snapshot(dur = duration , type = 'msg')
		self.nodes = log.get_event_snapshot(dur = duration , type = 'join')
		self.events = log.get_event_snapshot(dur = duration , type = 'event')
		self.start_date = log.start_date
		self.end_date = log.end_date
		self.dur = duration

	def create_cumulative_edge_list(self):
		date = self.start_date
		cumulative_msgs = {}
		cumulative_msg_list = []
		cumulative_nodes = {}
		cumulative_node_list = []
		while(date <= self.end_date):
			if date in self.msgs:
				l = [(el[0] , el[1]) for el in self.msgs[date]]
				cumulative_msg_list += l
			cumulative_msgs[date] = cumulative_msg_list[:]
			if date in self.nodes:
				l = [el[0] for el in self.nodes[date]]
				cumulative_node_list += l
			cumulative_nodes[date] = cumulative_node_list[:]

			td = 7
			if self.dur == 'day' : td = 1
			date += timedelta(td)

		return cumulative_msgs ,  cumulative_nodes

	def create_culumative_graphs(self):
		c_edge , c_nodes = self.create_cumulative_edge_list()
		date = self.start_date
		i = 0
		graphs = {}
		while(date <= self.end_date):

			adj_list = self.__get_adj_list__( c_edge[date] )

			with open('temp_graph.graphml' , 'w') as f:
				self.__create_graphml_file__(f , adj_list , c_nodes[date])

			g = load_graph('temp_graph.graphml')

			# for v in reversed(sorted(g.vertices())):
			# 	if v.out_degree < 1:
			# 		g.remove_vertex(v)

			graphs[date] = g

			td = 7
			if self.dur == 'day': td = 1
			date += timedelta(td)

		self.graphs = graphs
		return graphs

	def __create_graphml_file__(self, f , edges , nodes):
		f.write('<?xml version=\"1.0\" encoding=\"UTF-8\"?> ' +
				'<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" ' +
				'xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ' +
				'xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns' +
				'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n'+
				'<key id=\"d0\" for=\"node\" attr.name=\"id\" attr.type=\"string\"/> \n' +
				'<graph id=\"G\" edgedefault=\"undirected\">\n')


		for node in nodes:
			f.write('<node id = \"' + node + '\">\n'+
					'\t<data key = \"d0\">' + node + '</data> \n</node>\n')
		for edge in edges:
			f.write('<edge source = \"'+ edge[0] + '\" target = \"'+ edge[1] + '\" /> \n')


	def __get_adj_list__(self , l):
		undir_set = set()
		dir_set = set(l)
		for item in dir_set:
			if (item[1] , item[0]) in dir_set:
				if item[0] < item[1]:
					undir_set.add(item)
				else:
					undir_set.add((item[1] , item[0]))
		return list(undir_set)


	def create_id_invert(self):
		self.id_invert = {}
		g = self.graphs[self.end_date]
		id = g.vertex_properties['id']
		for v in g.vertices():
			self.id_invert[id[v]] = v



# gc = Graph_Creator('week')
#
# print gc.create_culumative_graphs()



#gc.create_node_edge_list_for_snapshots()




class Stat:
	def __init__(self , dur = 'week'):
		self.gc = Graph_Creator(dur)
		self.graphs = self.gc.create_culumative_graphs()

	def user_pagerank(self):
		return pickle.load(open(project_folder \
					+ 'data/pickles/weekly_pr.p', 'rb'))

	def create_user_pagerank(self):
		pr = {}
		last_graph = self.graphs[self.gc.end_date]
		id = last_graph.vertex_properties['id']
		prg = self.pagerank()
		for date , prank in prg.iteritems():
			temp = {}
			g = self.graphs[date]
			for v in g.vertices():
				if prank[v] > 0:
					temp[id[v]] = prank[v]
			pr[date] = temp
		pickle.dump(pr, open(project_folder \
						+ 'data/pickles/weekly_pr.p', 'wb'))
		return pr


	def create_users_rank(self):
		rank = {}
		prg = self.user_pagerank()
		for date , pr in prg.iteritems():
			date_rank = sorted(pr, key=pr.get, reverse=True )
			rank[date] = date_rank
		pickle.dump(rank, open(project_folder \
					+ 'data/pickles/weekly_ranks.p', 'wb'))
		return rank
	def get_users_rank(self):
		return pickle.load(open(project_folder \
					+ 'data/pickles/weekly_ranks.p', 'rb'))






	def pagerank(self ):
		pr = {}
		dates = sorted(self.graphs.iterkeys())
		for date in dates:
			if self.graphs[date].num_vertices() > 0:
				pr[date] = gtc.pagerank(self.graphs[date])
		self.pr = pr
		return pr

	def degree_centrality(self):
		dc = {}
		dates = sorted(self.graphs.iterkeys())
		i = 0
		for date in dates:
			if self.graphs[date].num_vertices() > 0:
				dc[date] = self.graphs[date].degree_property_map('total')
		return dc



	def test(self):
		pr = self.pagerank()
		dates = sorted(pr.iterkeys())
		s = []

		for date in dates:
			try:
				g = self.graphs[date]
				rank = pr[date]
				print rank[g.vertex(777)]
				id = g.vertex_properties['id']
				graphmlid = g.vertex_properties['_graphml_vertex_id']
				print(id[g.vertex(0)])
				print(graphmlid[g.vertex(0)])
				s.append(rank[1677])
			except Exception as e:
				continue

		x = [i for i in range(len(s))]
		plt.plot(x ,s)
		plt.show()



#
# st = Stat()
# st.create_user_pagerank()
# st.create_users_rank()
