import subprocess
import time
import sys

args = ['/home/cicd/Universe2/build/bolt/bolt', '/home/cicd/Universe2/bolt/configs/amzn_benchmarks.cfg']

# Thresholds for checking accuracy and time after epoch number
epoch_check_num = 3
training_time_threshold = 450
accuracy_threshold = 0.31
total_training_time_threshold = 12000

p = subprocess.Popen(args, stdout=subprocess.PIPE, shell=False)
start = time.time()
while True:
    line = p.stdout.readline().decode('utf-8')
    if p.poll() is not None:
        break
    if line.startswith(f'Epoch {epoch_check_num}'):
        print (line, end = '')
        line = p.stdout.readline().decode('utf-8')

        # Kill job and signal an error if training time after epoch 3 is longer than desired.
        line = p.stdout.readline().decode('utf-8')
        print(line, end = '')
        training_time = line.split(' ')[-2]
        if float(training_time) > training_time_threshold:
            print(f'Epoch {epoch_check_num} training time *({training_time}))* took longer than expected *({training_time_threshold})*')
            sys.exit(1)
        
        line = p.stdout.readline().decode('utf-8')
        print(line, end = '')

        # Kill job and signal an error if accuracy after epoch 3 is less than desired.
        line = p.stdout.readline().decode('utf-8')
        print(line, end = '')
        accuracy = line.split(' ')[-2]
        if float(accuracy) < accuracy_threshold:
            print(f'Epoch {epoch_check_num} accuracy *({accuracy})* is lower than expected *({accuracy_threshold})*')
            sys.exit(1)

    curr = time.time()
    # Error if total training time is above threshold.
    if curr - start > total_training_time_threshold:
        print(f'Current training time took longer than expected *({total_training_time_threshold})*')
        sys.exit(1)
rc = p.poll()