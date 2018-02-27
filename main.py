import os
from src.log import Log
from src.plots import Plot
project_folder = os.path.dirname(__file__).split("src")[0]



p = Plot()
p.event_frequency('event', 'week')