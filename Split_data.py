import os
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm

# ==================================================
# Split shell MD data from YiHao
# into split folder with coord and force txt file
# all length unit is ans
# all force unit is 10J/mol/K
#
#                               2022-04-04
# ==================================================


class Data_split():
    @staticmethod
    def Load_single_config(config_path,nAtom):
        nNa = int(nAtom/2)
        f = open(config_path,"r")

        Coord_electron = []
        Coord_core = []

        Force_electron_short = []
        Force_electron_long = []
        Force_core_short = []
        Force_core_long = []

        line_buffer = f.readline()
        for index in range(nAtom):
            line_buffer = f.readline()
            Coord_electron.append([float(i) for i in line_buffer.strip().split()])
        for index in range(nAtom):
            line_buffer = f.readline()
            Coord_core.append([float(i) for i in line_buffer.strip().split()])

        line_buffer = f.readline()
        for index in range(nAtom):
            line_buffer = f.readline()
            Force_electron_short.append([float(i) for i in line_buffer.strip().split()])

        line_buffer = f.readline()
        for index in range(nAtom):
            line_buffer = f.readline()
            Force_electron_long.append([float(i) for i in line_buffer.strip().split()])

        line_buffer = f.readline()
        for index in range(nAtom):
            line_buffer = f.readline()
            Force_core_short.append([float(i) for i in line_buffer.strip().split()])

        line_buffer = f.readline()
        for index in range(nAtom):
            line_buffer = f.readline()
            Force_core_long.append([float(i) for i in line_buffer.strip().split()])

        Coord_electron = np.array(Coord_electron)
        Coord_core = np.array(Coord_core)

        Force_electron_short = np.array(Force_electron_short)
        Force_electron_long = np.array(Force_electron_long)
        Force_core_short = np.array(Force_core_short)
        Force_core_long = np.array(Force_core_long)

        R_Na = Coord_core[:nNa,:]
        R_Cl = Coord_core[nNa:,:]

        r_Na = Coord_electron[:nNa,:]
        r_Cl = Coord_electron[nNa:,:]

        F_Na_s = Force_core_short[:nNa,:]
        F_Cl_s = Force_core_short[nNa:,:]

        F_Na_l = Force_core_long[:nNa,:]
        F_Cl_l = Force_core_long[nNa:,:]

        f_Na_s = Force_electron_short[:nNa,:]
        f_Cl_s = Force_electron_short[nNa:,:]

        f_Na_l = Force_electron_long[:nNa,:]
        f_Cl_l = Force_electron_long[nNa:,:]

        return(R_Na, R_Cl, r_Na, r_Cl, F_Na_s, F_Na_l, F_Cl_s, F_Cl_l, f_Na_s, f_Na_l, f_Cl_s, f_Cl_l)
    @staticmethod
    def Build_folder(folder_path, rewrite = False):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            if rewrite:
                shutil.rmtree(folder_path)
                os.makedirs(folder_path)
    @staticmethod
    def Rebuild_VMD_data(save_path, R_Na, R_Cl, r_Na, r_Cl, Box):
        total_number = R_Na.shape[0] + R_Cl.shape[0] + r_Na.shape[0] + r_Cl.shape[0]
        with open(save_path,"w") as save_file:
            save_file.write("\t{}\n".format(total_number))
            save_file.write("NaCl_with_shell_model_configration  Box_range {}  {}  {}\n".format(Box[0],Box[1],Box[2]))
            for index in range(R_Na.shape[0]):
                save_file.write("NA\t{}\t{}\t{}\n".format(R_Na[index,0],R_Na[index,1],R_Na[index,2]))
            for index in range(R_Cl.shape[0]):
                save_file.write("CL\t{}\t{}\t{}\n".format(R_Cl[index,0],R_Cl[index,1],R_Cl[index,2]))
            for index in range(r_Na.shape[0]):
                save_file.write("H\t{}\t{}\t{}\n".format(r_Na[index,0],r_Na[index,1],r_Na[index,2]))
            for index in range(r_Cl.shape[0]):
                save_file.write("H\t{}\t{}\t{}\n".format(r_Cl[index,0],r_Cl[index,1],r_Cl[index,2]))
    @staticmethod
    def Run_split(data_folder_path, config_list, save_folder_path, nAtom):
        Data_split.Build_folder(save_folder_path)
        for config_name in tqdm(config_list):
            config_load_path = "{}/{}".format(data_folder_path,config_name)
            config_save_path = "{}/{}".format(save_folder_path,config_name)
            Data_split.Build_folder(config_save_path)
            Data_split.Build_folder("{}/features".format(config_save_path))

            R_Na, R_Cl, r_Na, r_Cl, F_Na_s, F_Na_l, F_Cl_s, F_Cl_l, f_Na_s, f_Na_l, f_Cl_s, f_Cl_l = Data_split.Load_single_config(config_load_path,nAtom)
            box = np.array([24.0959889334770030, 24.0959889334770030, 24.0959889334770030])

            Data_split.Rebuild_VMD_data("{}/Config.xyz".format(config_save_path),R_Na,R_Cl,r_Na,r_Cl,box)

            np.savetxt("{}/Box.txt".format(config_save_path), box)

            np.savetxt("{}/R_Na.txt".format(config_save_path),R_Na)
            np.savetxt("{}/R_Cl.txt".format(config_save_path),R_Cl)
            np.savetxt("{}/r_Na.txt".format(config_save_path),r_Na)
            np.savetxt("{}/r_Cl.txt".format(config_save_path),r_Cl)

            np.savetxt("{}/F_Na_s.txt".format(config_save_path),F_Na_s)
            np.savetxt("{}/F_Na_l.txt".format(config_save_path),F_Na_l)
            np.savetxt("{}/F_Cl_s.txt".format(config_save_path),F_Cl_s)
            np.savetxt("{}/F_Cl_l.txt".format(config_save_path),F_Cl_l)

            np.savetxt("{}/f_Na_s.txt".format(config_save_path),f_Na_s)
            np.savetxt("{}/f_Na_l.txt".format(config_save_path),f_Na_l)
            np.savetxt("{}/f_Cl_s.txt".format(config_save_path),f_Cl_s)
            np.savetxt("{}/f_Cl_l.txt".format(config_save_path),f_Cl_l)


if __name__ == "__main__":
    train_config_name = [ i for i in range(1001,2501)]
    valid_config_name = [ i for i in range(2501,3001)]
    total_config_name = [ i for i in range(1001,3001)]

    Data_split.Run_split(   data_folder_path= "/DATA/users/yanghe/projects/NeuralNetwork_PES/Public_data/shellNaCl/",
                            config_list=total_config_name,
                            save_folder_path="/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/Split_data",
                            nAtom=216)


    # Data_split.Run_split(   data_folder_path= "/DATA/users/yanghe/projects/Shell_NaCl_model/shellNaCl/",
    #                         config_list=valid_config_name,
    #                         save_folder_path="/DATA/users/yanghe/projects/Shell_NaCl_model/Train_model/data/Data_ans_op1/valid_data",
    #                         nAtom=216)







