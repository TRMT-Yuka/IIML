"""
Training script for DICMOR
dataset_name: Selecting dataset (mosi or mosei)
seeds: This is a list containing running seeds you input
mr: missing rate ranging from 0.1 to 0.7
"""

# def DICMOR_run(
#         model_name, dataset_name, config=None, config_file="", seeds=[], mr=0.1, is_tune=False,
#         tune_times=500, feature_T="", feature_A="", feature_V="",
#         model_save_dir="./pt", res_save_dir="./result", log_dir="./log",
#         gpu_ids=[0], num_workers=4, verbose_level=1, mode='train'
# ):

# from run import DICMOR_run
from run import DICMOR_run_Missing_Weights

# task_name = "Vanilla"
task_name = "IIML"

mr_n_l = [7]
print("mr:",mr_n_l)
mws_l = [622,262,226]#[811,181,118,622,262,226,433,343,334]:#本来やるべきところ
print("mws:",mws_l)

def get_missing_weights(mws):
    l,v,a = str(mws)
    return [int(l)/10,int(v)/10,int(a)/10]

for mr_n in mr_n_l:
    for mws in mws_l:
        DICMOR_run_Missing_Weights(model_name='dicmor',
                   dataset_name='mosei',
                   seeds=[1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119],
                   mr=mr_n/10,model_save_dir="./pt/"+task_name+"/mr_0"+str(mr_n)+"/"+str(mws)+"/",
                   res_save_dir="./result/"+task_name+"/mr_0"+str(mr_n)+"/"+str(mws)+"/",
                   log_dir="./log/"+task_name+"/mr_0"+str(mr_n)+"/"+str(mws)+"/",
                   gpu_ids=[0,1],
                   missing_weights=get_missing_weights(mws)
                   )