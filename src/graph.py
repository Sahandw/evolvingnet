from graph_tool.all import *
import graph_tool as gt
import graph_tool.draw as gtdraw
from log import Log
from datetime import timedelta
import networkx as nx
import matplotlib.pyplot as plt



class Graph_Creator:
	def __init__(self , duration = 'week'):
		log = Log()
		self.msgs = log.create_msg_snapshot(duration)
		self.nodes = log.create_join_snapshots(duration)
		self.events = log.create_event_snapshot(duration)
		self.start_date = log.start_date
		self.end_date = log.end_date
	def create_cumulative_graph(self, start , end):
		g = Graph(directed = False)
		relevant_events , rel_msgs , rel_nodes = self.extract_events(start , end)
		id = g.new_vertex_property('string')
		id_invert = {}
		for node in rel_nodes:
			v = g.add_vertex()
			id[v] = node
			id_invert[node] = v
		edge_set = set()
		for msg in rel_msgs:
			u = msg[0]
			v = msg[1]
			if (v , u) in rel_msgs:
				if (v,u) not in edge_set:
					edge_set.add((u,v))
		for (u,v) in edge_set:
			g.add_edge(id_invert[u] , id_invert[v])
		return g




	def extract_events(self, start , end):
		relevant_dates = []
		for key in sorted(self.events.keys()):
			if key <= end and key >= start:
				relevant_dates.append(key)
		relevant_events = []
		relevant_msgs = []
		relevant_nodes  = []
		for key in relevant_dates:
			relevant_events += self.events[key]
			relevant_nodes += self.nodes[key]
			relevant_msgs += self.msgs[key]

		rel_nodes = [] # no timestamp
		rel_msgs = [] # no timestamp
		for item in relevant_nodes:
			rel_nodes.append(item[0])
		for item in relevant_msgs:
			rel_msgs.append((item[0] , item[1]))

		return relevant_events , rel_msgs , rel_nodes



	def create_node_edge_list_for_snapshots(self):
		graph_nodes = {}
		graph_edges = {}
		dates = []
		for date in sorted(self.events.iterkeys()):
			dates.append(date)
		temp = []
		for date in dates:
			temp += self.nodes[date]
			graph_nodes[date] = temp[:]
		print graph_nodes[self.start_date + timedelta(7)]
		for key,items in graph_nodes.iteritems():
			new_items = []
			for item in items:
				new_items.append(item[0])
			graph_nodes[key] = new_items
		print graph_nodes[self.start_date + timedelta(21)]

		temp = []
		for date in dates:
			temp += self.msgs[date]
			graph_edges[date] = temp[:]
		print graph_edges[self.start_date + timedelta(21)]
		for key, items in graph_edges.iteritems():
			new_items = []
			for item in items:
				new_items.append((item[0], item[1]))
				graph_edges[key] = new_items

		for key,items in graph_edges.iteritems():
			new_items = set()
			for item in items:
				if (item[1] , item[0]) in items:
					new_items.add(tuple(sorted([item[0],item[1]])))
					graph_edges[key] = new_items


		print graph_edges[self.start_date + timedelta(21)]





gc = Graph_Creator()
gc.create_node_edge_list_for_snapshots()
