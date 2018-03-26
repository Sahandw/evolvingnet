from datetime import datetime, timedelta
import os
import pickle
project_folder = os.path.dirname(__file__).split("src")[0]


class Log:
	def __init__(self):
		with open(project_folder + 'data/data.txt')as f:
			self.raw_events = f.readlines()
		f.close()

		self.event_log = self.__create_log__()
		self.node_log  , self.msg_log = self.__create_node_and_message_event_log()

		self.start_date = datetime(2004, 3, 22, 00, 00, 00)
		self.end_date = datetime(2004, 11, 1, 00, 00, 00)



	def __create_log__(self):
		log = []
		for entry in self.raw_events:
			l = entry.split(' ')
			t = (l[0] + ' ' + l[1]).replace('\"', '')
			t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
			u, v, c = l[2], l[3], l[4].replace('\n', '')
			log.append((u, v, t, c))
		return log


	def __create_node_and_message_event_log(self):
		node_log = []
		msg_log = []
		for entry in self.event_log:
			t = entry[2]
			if entry[3] == '1' and entry[0] == entry[1]:
				node_log.append((entry[0], entry[2]))
			else:
				msg_log.append(entry)

		return node_log , msg_log


	def __create_join_snapshots__(self , duration = 'week'):
		date = self.start_date
		join_snapshot = {}

		interval = timedelta(7)
		if duration == 'day':
			interval = timedelta(1)

		while date <= self.end_date:
			join_snapshot[date] = []
			date += interval

		for entry in self.node_log:
			t = entry[1]
			week = (t - timedelta(t.weekday())).replace(hour=00, minute=00, second=00)
			day = t.replace(hour=00, minute=00, second=00)
			if duration == 'day':
				join_snapshot[day].append(entry)
			if duration == 'week':
				join_snapshot[week].append(entry)


		return join_snapshot


	def __create_msg_snapshot__(self , duration = 'week'):
		date = self.start_date
		msg_snapshot = {}
		interval = timedelta(7)
		if duration == 'day':
			interval = timedelta(1)

		while date <= self.end_date:
			msg_snapshot[date] = []
			date += interval

		for entry in self.msg_log:
			t = entry[2]
			week = (t - timedelta(t.weekday())).replace(hour=00, minute=00, second=00)
			day = t.replace(hour=00, minute=00, second=00)
			if duration == 'day':
				msg_snapshot[day].append(entry)
			if duration == 'week':
				msg_snapshot[week].append(entry)

		return msg_snapshot

	def __create_event_snapshot__(self , duration = 'week'):
		date = self.start_date
		event_snapshot = {}
		interval = timedelta(7)
		if duration == 'day':
			interval = timedelta(1)
		while date <= self.end_date:
			event_snapshot[date] = []
			date += interval

		for entry in self.event_log:
			t = entry[2]
			week = (t - timedelta(t.weekday())).replace(hour=00, minute=00, second=00)
			day = t.replace(hour=00, minute=00, second=00)
			if duration == 'day':
				event_snapshot[day].append(entry)
			if duration == 'week':
				event_snapshot[week].append(entry)

		return event_snapshot



	def create_snapshot_pickles(self):
		pickle.dump(self.__create_event_snapshot__('week') ,
					open(project_folder + 'data/pickles/weekly_events.p' , 'wb'))
		pickle.dump(self.__create_event_snapshot__('day'),
					open(project_folder + 'data/pickles/daily_events.p', 'wb'))
		pickle.dump(self.__create_join_snapshots__('week'),
					open(project_folder + 'data/pickles/weekly_joins.p', 'wb'))
		pickle.dump(self.__create_join_snapshots__('day'),
					open(project_folder + 'data/pickles/daily_joins.p', 'wb'))
		pickle.dump(self.__create_msg_snapshot__('week'),
					open(project_folder + 'data/pickles/weekly_msgs.p', 'wb'))
		pickle.dump(self.__create_msg_snapshot__('day'),
					open(project_folder + 'data/pickles/daily_msgs.p', 'wb'))

	def get_event_snapshot(self , dur = 'week' , type = 'event'):
		d = 'weekly'
		if dur == 'day': d = 'daily'
		name = project_folder + 'data/pickles/' + d + '_' + type + 's.p'
		return pickle.load(open(name , 'rb'))


	def join_date_since_inception(self):
		join_time = {}
		for join in self.node_log:
			join_time[join[0]] = (join[1] - self.start_date).days
		return join_time


	def __create_user_event_sequence__(self ):
		event_log = self.__create_log__()
		#print event_log[0] , event_log[8870]
		user_log = {}
		for entry in event_log:
			if entry[0] not in user_log:
				user_log[entry[0]] = []
			if entry[1] not in user_log:
				user_log[entry[1]] = []
			if entry[3] == '1' and entry[0] == entry[1]:
				user_log[entry[0]].append(('j' , entry[2]))
			else:
				user_log[entry[0]].append(('s' ,  entry[2]))
				user_log[entry[1]].append(('r' ,  entry[2]))

		pickle.dump(user_log, open(project_folder + 'data/pickles/user_log.p', 'wb'))

	def get_user_event_sequence(self):
		return pickle.load(open(project_folder + 'data/pickles/user_log.p' , 'rb'))

	def get_user_symb_sequence(self):
		return pickle.load(open(project_folder + 'data/pickles/user_symbols_time.p' , 'rb'))

	def get_user_seq(self):
		return pickle.load(open(project_folder + 'data/pickles/user_seq.p' , 'rb'))

	def __create_user_symbolic_sequence__(self):
		user_log = self.get_user_event_sequence()
		for user  , log  in  user_log.iteritems():
			for i in range(len(log)):
				j = i + 1
				ts = 'e'
				if j!= len(log):
					delta = (log[j][1] - log[i][1]).seconds
					ts = 'd'
					if delta < 600: ts = 'a'
					elif delta >= 600 and delta < 7200: ts = 'b'
					elif delta >= 7200 and delta < 86400: ts = 'c'
				log[i] = (log[i][0] , log[i][1] , ts)

		pickle.dump(user_log, open(project_folder + 'data/pickles/user_symbols_time.p', 'wb'))



		for user, log in user_log.iteritems():
			l = ''
			for entry in log:
				l += entry[0] +  entry[2]
			user_log[user] = l


		pickle.dump(user_log, open(project_folder + 'data/pickles/user_seq.p', 'wb'))





	def create_user_cumulative_events(self , dur = 'week'):
		event_sequence = self.get_user_event_sequence()
		user_ce = {}
		d = 1
		if dur == 'week':
			d = 7
		date = self.start_date
		while date <= self.end_date:
			user_ce[date] = {}
			for user in event_sequence.iterkeys():
				if date == self.start_date:
					user_ce[date][user] = []
				else: user_ce[date][user] = user_ce[date - timedelta(d)][user][:]
			for user , user_events in event_sequence.iteritems():
				for e in user_events:
					day = e[1].replace(hour=00, minute=00, second=00)
					week = (e[1] - timedelta(e[1].weekday())).replace(hour=00, minute=00, second=00)
					if dur == 'day' and day == date:
						user_ce[date][user].append((e[0],e[1]))
					if dur == 'week' and week == date:
						user_ce[date][user].append((e[0],e[1]))

			date += timedelta(d)
		if dur == 'week':
			pickle.dump(user_ce, open(project_folder + 'data/pickles/user_ce_weekly.p', 'wb'))
		else:
			pickle.dump(user_ce,open(project_folder + 'data/pickles/user_ce_daily.p', 'wb'))

	def get_user_cumulative_events(self , dur = 'week'):
		if dur == 'week':
			return pickle.load(open(project_folder + 'data/pickles/user_ce_weekly.p', 'rb'))
		else:
			return pickle.load(open(project_folder + 'data/pickles/user_ce_daily.p', 'rb'))


	def create_user_cumulative_seq(self, dur = 'week'):
		if dur == 'week':
			ce = pickle.load(open(project_folder + 'data/pickles/user_ce_weekly.p', 'rb'))
		else:
			ce = pickle.load(open(project_folder + 'data/pickles/user_ce_daily.p', 'rb'))
		user_cs = {}
		for date in ce.iterkeys():
			user_cs [date] = {}
			for user in ce[date].iterkeys():
				user_cs[date][user] = ""
			for user in ce[date].iterkeys():
				for i , e in enumerate(ce[date][user]):
					j = i + 1
					ts = 'e'
					if j != len(ce[date][user]):
						delta = (ce[date][user][j][1] - ce[date][user][i][1]).seconds
						ts = 'd'
						if delta < 600:
							ts = 'a'
						elif delta >= 600 and delta < 7200:
							ts = 'b'
						elif delta >= 7200 and delta < 86400:
							ts = 'c'
					user_cs[date][user] += e[0]
					user_cs[date][user] += ts
		if dur == 'week':
			pickle.dump(user_cs, open(project_folder + 'data/pickles/user_cs_weekly.p', 'wb'))
		else:
			pickle.dump(user_cs,open(project_folder + 'data/pickles/user_cs_daily.p', 'wb'))

	def get_user_cumulative_seq(self , dur = 'week'):
		if dur == 'week':
			return pickle.load(open(project_folder + 'data/pickles/user_cs_weekly.p', 'rb'))
		else:
			return pickle.load(open(project_folder + 'data/pickles/user_cs_daily.p', 'rb'))



	def create_user_date_events(self,dur = 'week'):
		event_sequence = self.get_user_event_sequence()
		user_ce = {}
		d = 1
		if dur == 'week':
			d = 7
		date = self.start_date
		while date <= self.end_date:
			user_ce[date] = {}
			for user in event_sequence.iterkeys():
				user_ce[date][user] = []
			for user, user_events in event_sequence.iteritems():
				for e in user_events:
					day = e[1].replace(hour=00, minute=00, second=00)
					week = (e[1] - timedelta(e[1].weekday())).replace(hour=00, minute=00, second=00)
					if dur == 'day' and day == date:
						user_ce[date][user].append((e[0], e[1]))
					if dur == 'week' and week == date:
						user_ce[date][user].append((e[0], e[1]))
			date += timedelta(d)
		if dur == 'week':
			pickle.dump(user_ce, open(project_folder + 'data/pickles/user_de_weekly.p', 'wb'))
		else:
			pickle.dump(user_ce, open(project_folder + 'data/pickles/user_de_daily.p', 'wb'))

	def get_user_date_events(self, dur='week'):
		if dur == 'week':
			return pickle.load(open(project_folder + 'data/pickles/user_de_weekly.p', 'rb'))
		else:
			return pickle.load(open(project_folder + 'data/pickles/user_de_daily.p', 'rb'))



	def create_user_date_seq(self,dur = 'week'):
		ce = self.get_user_date_events(dur)
		user_cs = {}
		for date in ce.iterkeys():
			user_cs [date] = {}
			for user in ce[date].iterkeys():
				user_cs[date][user] = ""
			for user in ce[date].iterkeys():
				for i , e in enumerate(ce[date][user]):
					j = i + 1
					ts = 'e'
					if j != len(ce[date][user]):
						delta = (ce[date][user][j][1] - ce[date][user][i][1]).seconds
						ts = 'd'
						if delta < 600:
							ts = 'a'
						elif delta >= 600 and delta < 7200:
							ts = 'b'
						elif delta >= 7200 and delta < 86400:
							ts = 'c'
					user_cs[date][user] += e[0]
					user_cs[date][user] += ts
		if dur == 'week':
			pickle.dump(user_cs, open(project_folder + 'data/pickles/user_ds_weekly.p', 'wb'))
		else:
			pickle.dump(user_cs,open(project_folder + 'data/pickles/user_ds_daily.p', 'wb'))

	def get_user_date_seq(self, dur='week'):
		if dur == 'week':
			return pickle.load(open(project_folder + 'data/pickles/user_ds_weekly.p', 'rb'))
		else:
			return pickle.load(open(project_folder + 'data/pickles/user_ds_daily.p', 'rb'))



		if dur == 'week':
			ce = pickle.load(open(project_folder + 'data/pickles/user_ce_weekly.p', 'rb'))
		else:
			ce = pickle.load(open(project_folder + 'data/pickles/user_ce_daily.p', 'rb'))
		user_cs = {}
		for date in ce.iterkeys():
			user_cs [date] = {}
			for user in ce[date].iterkeys():
				user_cs[date][user] = ""
			for user in ce[date].iterkeys():
				for i , e in enumerate(ce[date][user]):
					j = i + 1
					ts = 'e'
					if j != len(ce[date][user]):
						delta = (ce[date][user][j][1] - ce[date][user][i][1]).seconds
						ts = 'd'
						if delta < 600:
							ts = 'a'
						elif delta >= 600 and delta < 7200:
							ts = 'b'
						elif delta >= 7200 and delta < 86400:
							ts = 'c'
					user_cs[date][user] += e[0]
					user_cs[date][user] += ts
		if dur == 'week':
			pickle.dump(user_cs, open(project_folder + 'data/pickles/user_cs_weekly.p', 'wb'))
		else:
			pickle.dump(user_cs,open(project_folder + 'data/pickles/user_cs_daily.p', 'wb'))

	def event_frequency(self, type='msg', duration='week'):
		data = []
		interval = timedelta(7)
		if duration == 'day': interval = timedelta(1)
		event_snapshot = self.get_event_snapshot(duration, type)

		date = self.start_date
		while date <= self.end_date:
			data.append(len(event_snapshot[date]))
			date += interval

		return data






