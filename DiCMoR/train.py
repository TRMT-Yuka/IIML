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

from run import DICMOR_run

for mr_n in [7]:
    DICMOR_run(model_name='dicmor',
               dataset_name='mosei',
               seeds=[1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119],
               mr=mr_n/10,model_save_dir="./pt/vanilla/mr_0"+str(mr_n), res_save_dir="./result/vanilla/mr_0"+str(mr_n), log_dir="./log/vanilla/mr_0"+str(mr_n),gpu_ids=[0,1])