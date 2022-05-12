import os
import numpy as np
from glob import glob
from tqdm import tqdm

class Generate_label():
    @staticmethod
    def Load_single_force(config_path):
        FNa_short_path = "{}/F_Na_s.txt".format(config_path)
        FCl_short_path = "{}/F_Cl_s.txt".format(config_path)

        FNa_short = np.loadtxt(FNa_short_path)
        FCl_short = np.loadtxt(FCl_short_path)

        return(FNa_short, FCl_short)
    @staticmethod
    def Assemble_force(folder_path, config_list, save_path):
        FNa_short_list = ()
        FCl_short_list = ()
        for config_name in config_list:
            config_path = "{}/{}".format(folder_path, config_name)
            FNa_short, FCl_short = Generate_label.Load_single_force(config_path)
            FNa_short_list += (FNa_short,)
            FCl_short_list += (FCl_short,)
        FNa_short_matrix = np.array(FNa_short_list)
        FCl_short_matrix = np.array(FCl_short_list)

        np.save("{}/fN_short".format(save_path), FNa_short_matrix)
        np.save("{}/fC_short".format(save_path), FCl_short_matrix)

if __name__ == "__main__":
    TRAIN_INPUT_PATH = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/TrainInput"
    VALID_INPUT_PATH = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/ValidInput"

    train_config_name = [i for i in range(1001,2501)]
    valid_config_name = [i for i in range(2501,3001)]

    Generate_label.Assemble_force(  "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/Split_data",
                                    train_config_name,
                                    TRAIN_INPUT_PATH)

    Generate_label.Assemble_force(  "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/Split_data",
                                    valid_config_name,
                                    VALID_INPUT_PATH)






