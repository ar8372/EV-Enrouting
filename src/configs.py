import numpy as np


## FUNCTIONALITIES 
size =  1000 #1000 #500
noise = False
battery = True
find_optimal_CS = False
static_quality_score = False
dynamic_quality_score = False
demo = False
#####################################

tile_size = 10
drag = False
Last_CS = (-100,-100) # something which is not in 50*50
HOME =  [] #0
AGENT = 0
Grid_shape = (50,50) #(100,100) # later set automatically
no_actions = 4
terminatos = []
optimal_path = np.zeros((50,50))
non_path_reward = -80


########################################
optimal_value = 0
state =0 
can_initial_reward = -1