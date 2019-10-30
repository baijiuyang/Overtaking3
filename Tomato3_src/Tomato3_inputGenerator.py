import random

v0s = []
rep = 10
v0 = [x/10.0 for x in range(8,14)]
for v in [x/10.0 for x in range(8,14)]:
	for _ in range(rep):
		v0s.append(v)
								
random.shuffle(v0s)

for subj in range(1,21):
	filename = 'Tomato3_subject' + str(subj).zfill(2) + '.csv'
	if subj%2 == 0:
		leader = 'avatar'
	else:
		leader = 'pole'
	data = ['Trial', 'd0', 'v0', 'leader', 'leaderOnset']
	with open(filename, mode='a') as file:		
		file.write(','.join(data)+'\n')
	for t in range(len(v0s)):	
		leaderOnset = random.uniform(3.0,4.0)
		data = [str(t+1), '2', str(v0s[t]), leader, str(leaderOnset)]
		with open(filename, mode='a') as file:
			file.write(','.join(data)+'\n')