import os



cmd_line="python3.8 test.py --dataset_name amaz_bi --set_up src_is_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --max_length 1024"
print(cmd_line)
ret_status = os.system(cmd_line)
if ret_status != 0:
    print("DRIVER (non-zero exit status from execution)>>{ret_status}<<")
    exit()

cmd_line="python3.8 test.py --dataset_name amaz_bi --set_up sim_in_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --k 4 --seeds 32 --max_length 1024"
print(cmd_line)
ret_status = os.system(cmd_line)
if ret_status != 0:
    print("DRIVER (non-zero exit status from execution)>>{ret_status}<<")
    exit()

cmd_line="python3.8 test.py --dataset_name amaz_bi --set_up src_is_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --max_length 1024"
print(cmd_line)
ret_status = os.system(cmd_line)
if ret_status != 0:
    print("DRIVER (non-zero exit status from execution)>>{ret_status}<<")
    exit()

cmd_line="python3.8 test.py --dataset_name amaz_bi --set_up sim_in_cross --model_name facebook/xglm-7.5B --batch_size 4 --use_demonstrations --use_logic --k 4 --seeds 32 --max_length 1024"
print(cmd_line)
ret_status = os.system(cmd_line)
if ret_status != 0:
    print("DRIVER (non-zero exit status from execution)>>{ret_status}<<")
    exit()