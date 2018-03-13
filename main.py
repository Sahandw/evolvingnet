import os
from src.log import Log
from src.plots import Plot
project_folder = os.path.dirname(__file__).split("src")[0]



l = Log()
#l.create_snapshot_pickles()

p = Plot()
p.event_frequency('join', 'week')

#p.create_graph_visualizations('day')


