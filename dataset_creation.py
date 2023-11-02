import os
from utils.data import create_few_shots

""""This will take time"""

create_few_shots(dataset_name='amaz_bi',k=4,seeds=[32,5,232,100,42],set_up='src_is_cross')
create_few_shots(dataset_name='amaz_bi',k=4,seeds=[32],set_up='sim_in_cross') #No Need to create 5 seeds as here demos will be same in all seeds (Nearest neighbor always same)
