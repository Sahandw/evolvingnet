import matplotlib.pyplot as plt
from datetime import timedelta
from log import Log
class Plot:
	def __init__(self):
		pass


	def event_frequency(self, type = 'msg' , duration = 'week'):
		data = []
		interval = timedelta(7)
		if duration =='day' : interval= timedelta(1)
		log = Log()
		event_snapshot = {}
		if type == 'event':
			event_snapshot = log.create_event_snapshot(duration)
		if type == 'msg':
			event_snapshot = log.create_msg_snapshot(duration)
		if type == 'join':
			event_snapshot = log.create_join_snapshots(duration)


		date = log.start_date
		while date <= log.end_date:
			data.append(len(event_snapshot[date]))
			date += interval

		# for key in sorted(event_snapshot.keys()):
		# 	print key , event_snapshot[key]
		# print len(event_snapshot.keys())

		plt.plot(range(1, len(data) + 1), data)
		plt.show()