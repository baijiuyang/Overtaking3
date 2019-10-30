# Tomato3_Overtaking
Include the source code for the data structure and analysis for pedestrian overtaking experiment
Data is not shared due to privacy agreement

Packages:

Python 3.6.9

numpy 1.16.4

matplotlib 3.1.1

pandas 0.25.1

scipy 1.3.1


Experimental details:

Participants:

Apparatus:
Samsung Odyssey
Avatar mode is randomly sampled from 40 options (20 males 20 females)
Yellow pole (0.2, 1.8, 0.2 meters)
Home pole is blue (0.4, 1.35, 0.4 meters)
Target pole is green (0.4, 3, 0.4 meters), is 14.2 meters away from home pole


Task:
participants were asked to follow a yellow pole or avatar, given the permission to overtake.

Conditions:
leader type (between subject): avatar, yellow pole
initial distance: 2 meter
Leader speed: 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 m/s
10 repetitions


data:
90Hz
File name: Tomato3_subj**_trial***_[d0], [v0].csv
Description: time series data in experimental trial
Column: leader position(x), leader position(z), leader position(y), follower position(x), follower position(z), follower position(y), follower yaw, pitch, roll, time stamp, leader model, at each frame
Tomato3_freewalk_subj**_s[*, session num]_trial***.csv
Description: time series data in freewalk trials
Column: follower position(x), follower position(z), follower position(y), follower yaw, pitch, roll, time stamp
row: data at each frame
Tomato3_subj**_IPD_[gender and IPD value in mm].txt
	Empty file

Notes:
For future experiments:
Ask participants to drink water before walking.
Do not block the camera on the hmd when initialization.
Odyssey can also lose track!

