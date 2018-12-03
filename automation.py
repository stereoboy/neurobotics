import os
import sys
import time
import subprocess

TEST_LOGDIR='./data5'
BEGIN_CNT=0
TEST_SIZE=10
PREFIX='debug'

def main():
    try:
        subp0 = None
        subp1 = None
        subp2 = None
        print(range(TEST_SIZE))
        for cnt in range(BEGIN_CNT, BEGIN_CNT + TEST_SIZE):
            print(cnt)
            subp0 = subprocess.Popen(['rosclean', 'purge'], stdin=subprocess.PIPE)
            subp0.communicate('y')

            cmd = ['xterm',  '-fn', '10x20', '-e', 'roslaunch neuro_stage_sim neuro_stage_sim.launch training_mode:=true']
            #cmd = ['gnome-terminal', '-e', 'roslaunch neuro_stage_sim neuro_stage_sim.launch training_mode:=true']
            subp1 = subprocess.Popen(cmd)

            logdirpath = os.path.join(TEST_LOGDIR, PREFIX + str(cnt))
            args = 'rosrun neuro_deep_planner neuro_costmap_planner_node.py --mode=train --dir=' + logdirpath
            subp2 = subprocess.Popen([args], shell=True)
            subp2.communicate()

            subp1.kill()
            subp1.wait()
            time.sleep(3)
    except Exception as exc:
        print("[ERROR] {}".format(exc))
        subp1.kill()
        subp1.wait()
        sys.exit()

if __name__== '__main__':
    main()
