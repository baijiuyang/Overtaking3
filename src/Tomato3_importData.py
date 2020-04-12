import numpy as np
import pandas as pd
import os
from Tomato3_dataStructure import Trial, Subject, Experiment
import pickle

Hz = 90
exp = Experiment()
for i in range(1,13):
    exp.subjects[i] = Subject(i)
    if i%2 == 0:
        exp.subjects[i].leader = 'avatar'
    else:
        exp.subjects[i].leader = 'pole'
    
input_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Tomato3_rawData\Tomato3_input'))
output_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Tomato3_rawData\Tomato3_output'))

for output_file in os.listdir(output_dir):
    output_file_path = os.path.join(output_dir, output_file)
    # import experimental data
    if  'Tomato3_subj' in output_file and '.csv' in output_file:
        with open(output_file_path, 'r') as f:
            df = pd.read_csv(f, header=None)
            lpos = np.array(df.iloc[:, [0,2,1]])
            fpos = np.array(df.iloc[:, [3,5,4]])
            fori = np.array(df.iloc[:, 6:9])
            tstamps = np.array(df.iloc[:, 9])
            leader_model = np.array(df.iloc[:, 10])
            subject_id = int(output_file[12:14])
            trial_id = int(output_file[20:23])
            v0 = float(output_file[27:30])
            if output_file[-5] == 'e':
                leader = 'pole'
            elif output_file[-5] == 'r':
                leader = 'avatar'
            leader_onset = None
            
            exp.subjects[subject_id].trials[trial_id] = Trial(subject_id, trial_id, lpos, fpos, fori, \
                                                                tstamps, v0, leader, leader_onset, leader_model)           
    # import IPD and gender data
    elif 'IPD' in output_file:    
        with open(output_file_path, 'r') as f:
            subject_id = int(output_file[12:14])
            exp.subjects[subject_id].gender == output_file[19]
            i = output_file.find('txt')
            exp.subjects[subject_id].IPD == float(output_file[20:i-1])
    # import freewalk data
    elif 'freewalk' in output_file:
        with open(output_file_path, 'r') as f:
            df = pd.read_csv(f, header=None)
            subject_id = int(output_file[21:23])
            session = int(output_file[-14])
            trial_id = int(output_file[-7:-4])
            v0 = 0
            leader = leader_onset = leader_model= None
            fpos = np.array(df.iloc[:,[0,2,1]])
            lpos = np.tile([0,0,0], (len(fpos), 1))
            fori = np.array(df.iloc[:,3:6])
            tstamps = np.array(df.iloc[:,-1])
            if session == 1:
                exp.subjects[subject_id].freewalk[trial_id] = Trial(subject_id, trial_id, lpos, fpos, fori, \
                                                                    tstamps, v0, leader, leader_onset, leader_model)
            else:
                trial_id += 4
                exp.subjects[subject_id].freewalk[trial_id] = Trial(subject_id, trial_id, lpos, fpos, fori, \
                                                                    tstamps, v0, leader, leader_onset, leader_model)
# import inputs
for input_file in os.listdir(input_dir):
    if 'Tomato3_subject' in input_file:
        subject_id = int(input_file[-6:-4])
        if subject_id in exp.subjects and exp.subjects[subject_id].trials != {}:
            # read experimental trials
            with open(os.path.join(input_dir, input_file), 'r') as f:
                df = pd.read_csv(f)
                for i in range(len(df)):
                    trial_id = df.iloc[i,0]
                    leader_onset = df.iloc[i,4]
                    exp.subjects[subject_id].trials[trial_id].leader_onset = leader_onset
 
with open('Tomato3_data.pickle', 'wb') as file:   
    pickle.dump(exp, file, pickle.HIGHEST_PROTOCOL)

# with open(filename, 'r') as file:
#     rows = file.read().split('\n')[:-1]
#     cells = [row.split(',') for row in rows]  
#     cells = np.array([[float(s) for s in row] for row in cells])
#     lpos = cells[:,[0,2,1]]
#     fpos = cells[:,[3,5,4]]
#     fori = cells[:,6:9]
#     tstamps = cells[:,-1]
