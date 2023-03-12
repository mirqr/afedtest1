######### Shared config #########
config_dict = {
    #### Data ####

    "step1_path" : './data_step1_models/',
    "step2_path" : './data_step2_prediction/',
    #"dataset_name": 'minst',
    #"dataset_name": 'fashion_minst',
    #"dataset_name": 'kmnist',
    "seed": 1,  # set random seed
    "num_clients": 10,  # choices: Int

}




def get_config_dict():
    return config_dict

