######### Shared config #########
config_dict = {
    #### Data ####

    "step1_path" : './data_step1_models/',
    "step2_path" : './data_step2_prediction/',
    "dataset": "CIFAR10",  # choices: CIFAR10, CIFAR100
    "seed": 1,  # set random seed
    "num_clients": 10,  # choices: Int

}




def get_config_dict():
    return config_dict

