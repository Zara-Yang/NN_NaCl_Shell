import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def Draw_input_hist(input_n, input_c):
    sns.set_theme(style = "ticks")

    fig, axs = plt.subplots(ncols = 2, nrows = 7, figsize = (12,35))
    sns.despine(fig)

    for index in range(7):
        sns.histplot(input_n_flatten[:,index],ax = axs[index,0], bins = 100)
        axs[index,0].set_title("Na shell feature {}".format(index))

        sns.histplot(input_c_flatten[:,index],ax = axs[index,1], bins = 100)
        axs[index,1].set_title("Cl shell feature {}".format(index))
    plt.savefig("Input_dist.jpg")

def Draw_force_hist(force_n, force_c):
    sns.set_theme(style = "ticks")

    fig, axs = plt.subplots(ncols = 2, nrows = 3, figsize = (12,15))
    plt.title("Shell Force hist")
    sns.despine(fig)

    for index in range(3):
        sns.histplot(force_n[:,index],ax = axs[index,0], bins = 100)
        axs[index,0].set_title("Na shell force axis-{}".format(index))

        sns.histplot(force_c[:,index],ax = axs[index,1], bins = 100)
        axs[index,1].set_title("Cl shell force axis-{}".format(index))
    plt.savefig("Force_dist.jpg")



if __name__ == "__main__":
    DATA_FOLDER = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/TrainInput"
    input_n = np.load("{}/xn_normal.npy".format(DATA_FOLDER))
    input_c = np.load("{}/xc_normal.npy".format(DATA_FOLDER))

    force_n = np.load("{}/fn_short.npy".format(DATA_FOLDER))
    force_c = np.load("{}/fc_short.npy".format(DATA_FOLDER))

    input_n_flatten = input_n.reshape((-1,7))
    input_c_flatten = input_c.reshape((-1,7))

    force_n_flatten = force_n.reshape((-1, 3))
    force_c_flatten = force_c.reshape((-1, 3))

    Draw_force_hist(force_n_flatten, force_c_flatten)


