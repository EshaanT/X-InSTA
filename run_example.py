import os

# --model_name gpt2-large facebook/xglm-7.5B facebook/xglm-1.7B facebook/xglm-2.9B EleutherAI/gpt-j-6B bigscience/bloomz-7b1
#--seeds 32,5,232,100,42
#python3.8 test.py --dataset_name xcopa --set_up src_are_hr --model_name facebook/xglm-2.9B --batch_size 5 --use_demonstrations --seeds 32,5,232,100,42
#python3.8 test.py --dataset_name xcopa --set_up random --model_name facebook/xglm-7.5B --batch_size 1 --use_demonstrations --seeds 232,100,42

#"python3.8 test.py --dataset_name cls --set_up sim_in_en --model_name facebook/xglm-7.5B --batch_size 6 --use_demonstrations --tar_l de,fr,jp --k 4 --max_length 1024"

#--tar_l id,it,qu,sw,ta,th,tr,vi,zh    ar,de,en,es,fr,hi,it,jap,nl,pl,pt,ru,sw,ur,vi,zh     de,en,es,fr,ja,zh    en,es  en,de,fr,jp
#--set_up same_as_tar_cor_no_label same_as_tar sim_as_tar src_is_en src_is_cross
#--max_length 256 512 1024
#amaz_bi hateval   

cmd_line="python3.8 test.py --dataset_name amaz_bi --set_up src_is_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --k 4 --tar_l es --seeds 32"
#cmd_line="python3.8 test.py --dataset_name cls --set_up sim_in_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --max_length 1024 --seeds 32"
print(cmd_line)
ret_status = os.system(cmd_line)
if ret_status != 0:
    print("DRIVER (non-zero exit status from execution)>>{ret_status}<<")
    exit()
    
cmd_line="python3.8 test.py --dataset_name amaz_bi --set_up sim_in_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --k 4 --tar_l es --seeds 32"
#cmd_line="python3.8 test.py --dataset_name cls --set_up sim_in_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --max_length 1024 --seeds 32"
print(cmd_line)
ret_status = os.system(cmd_line)
if ret_status != 0:
    print("DRIVER (non-zero exit status from execution)>>{ret_status}<<")
    exit()
    
cmd_line="python3.8 test.py --dataset_name amaz_bi --set_up src_is_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --tar_l es --seeds 32"
#cmd_line="python3.8 test.py --dataset_name cls --set_up sim_in_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --max_length 1024 --seeds 32"
print(cmd_line)
ret_status = os.system(cmd_line)
if ret_status != 0:
    print("DRIVER (non-zero exit status from execution)>>{ret_status}<<")
    exit()
    
cmd_line="python3.8 test.py --dataset_name amaz_bi --set_up sim_in_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --tar_l es --seeds 32"
#cmd_line="python3.8 test.py --dataset_name cls --set_up sim_in_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --max_length 1024 --seeds 32"
print(cmd_line)
ret_status = os.system(cmd_line)
if ret_status != 0:
    print("DRIVER (non-zero exit status from execution)>>{ret_status}<<")
    exit()