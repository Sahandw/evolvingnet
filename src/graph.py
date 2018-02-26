from graph_tool.all import *
import graph_tool as gt
from log import Log
class Graph_Creator:
	def __init__(self):
		pass

	def create_cumulative_graph(self, date):
		g = Graph(directed = False)
		