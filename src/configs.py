import numpy as np

class _globals:
    def __init__(self):
        self.size = 1000 
        self.noise = False 
        self.battery = True 
        self.find_optimal_CS = False 
        self.static_quality_score = False 
        self.dynamic_quality_score = False 
        self.demo = False 

        #----------------------------
        self.tile_size = 10 
        self.drag = False 
        self.Last_CS = (-100, -100)
        self.HOME = []
        self.AGENT = 0
        self.Grid_shape = (50,50) #(100,100) # later set automatically
        self.no_actions = 4
        self.terminatos = []
        self.optimal_path = np.zeros((50,50))
        self.non_path_reward = -80


        ########################################
        self.optimal_value = 0
        self.state =0 
        self.can_initial_reward = -1

        ## 
        self.optimal_policy = 0
    
    def load(self, no):
        pass 

"""
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

## 
optimal_policy = 0

"""