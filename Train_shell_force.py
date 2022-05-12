import os
import numpy as np
from tqdm import tqdm
from glob import glob
import torch
from torch import nn
import torch.optim as optim

# ==================================================
#
#   Train neural networl model for NaCl shell force.
#
#                                  ZaraYang
#
#===================================================


class Data_loader():
    def __init__(self,xN_path, xC_path, dxNN_path, dxNC_path, dxCN_path, dxCC_path, fN_path, fC_path):
        self.xN = np.load(xN_path)
        self.xC = np.load(xC_path)

        self.dxNN = np.load(dxNN_path)
        self.dxNC = np.load(dxNC_path)

        self.dxCN = np.load(dxCN_path)
        self.dxCC = np.load(dxCC_path)

        self.fN = np.load(fN_path) / 350
        self.fC = np.load(fC_path) / 1740

        self.config_number = self.fN.shape[0]
    def Generator(self,shuffle = False):
        if not shuffle:
            config_list = np.arange(self.config_number)
        else:
            config_list = np.random.permutation(self.config_number)
        for iconfig in config_list:
            batch_xN = torch.from_numpy(self.xN[iconfig,:,:]).to(torch.float32)
            batch_xC = torch.from_numpy(self.xC[iconfig,:,:]).to(torch.float32)

            batch_dxNN = torch.from_numpy(self.dxNN[iconfig,:,:,:,:]).to(torch.float32)
            batch_dxNC = torch.from_numpy(self.dxNC[iconfig,:,:,:,:]).to(torch.float32)

            batch_dxCN = torch.from_numpy(self.dxCN[iconfig,:,:,:,:]).to(torch.float32)
            batch_dxCC = torch.from_numpy(self.dxCC[iconfig,:,:,:,:]).to(torch.float32)

            batch_fN = torch.from_numpy(self.fN[iconfig,:,:]).to(torch.float32)
            batch_fC = torch.from_numpy(self.fC[iconfig,:,:]).to(torch.float32)

            yield(  batch_xN, batch_xC,
                    batch_dxNN, batch_dxNC,
                    batch_dxCN, batch_dxCC,
                    batch_fN, batch_fC)

class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        N_net = [7,5,5,1]
        C_net = [7,5,5,1]

        self.Nw1 = nn.Parameter(torch.randn(N_net[0],N_net[1])/1)
        self.Nb1 = nn.Parameter(torch.randn(N_net[1])/1)
        self.Nw2 = nn.Parameter(torch.randn(N_net[1],N_net[2])/1)
        self.Nb2 = nn.Parameter(torch.randn(N_net[2])/1)
        self.Nw3 = nn.Parameter(torch.randn(N_net[2],N_net[3])/1)
        self.Nb3 = nn.Parameter(torch.randn(N_net[3])/1)

        self.Cw1 = nn.Parameter(torch.randn(C_net[0],C_net[1])/1)
        self.Cb1 = nn.Parameter(torch.randn(C_net[1])/1)
        self.Cw2 = nn.Parameter(torch.randn(C_net[1],C_net[2])/1)
        self.Cb2 = nn.Parameter(torch.randn(C_net[2])/1)
        self.Cw3 = nn.Parameter(torch.randn(C_net[2],C_net[3])/1)
        self.Cb3 = nn.Parameter(torch.randn(C_net[3])/1)

    def forward(self, x_N, x_C, dx_NN, dx_CN, dx_NC, dx_CC):
        z1_N = torch.matmul(x_N, self.Nw1)
        z2_N = torch.matmul(torch.tanh(z1_N), self.Nw2)

        z1_C = torch.matmul(x_C, self.Cw1)
        z2_C = torch.matmul(torch.tanh(z1_C), self.Cw2)

        ap1_NN = torch.matmul(dx_NN, self.Nw1) / torch.cosh(z1_N) ** 2
        ap2_NN = torch.matmul(ap1_NN, self.Nw2) / torch.cosh(z2_N) ** 2
        y_NN = torch.matmul(ap2_NN, self.Nw3)

        ap1_CN = torch.matmul(dx_CN, self.Nw1) / torch.cosh(z1_N) ** 2
        ap2_CN = torch.matmul(ap1_CN, self.Nw2) / torch.cosh(z2_N) ** 2
        y_CN = torch.matmul(ap2_CN, self.Nw3)

        ap1_CC = torch.matmul(dx_CC, self.Cw1) / torch.cosh(z1_C) ** 2
        ap2_CC = torch.matmul(ap1_CC, self.Cw2) / torch.cosh(z2_C) ** 2
        y_CC = torch.matmul(ap2_CC, self.Cw3)

        ap1_NC = torch.matmul(dx_NC, self.Cw1) / torch.cosh(z1_C) ** 2
        ap2_NC = torch.matmul(ap1_NC, self.Cw2) / torch.cosh(z2_C) ** 2
        y_NC = torch.matmul(ap2_NC, self.Cw3)

        y_N = torch.sum(y_NN, axis=(-1, -2)) + torch.sum(y_NC, axis=(-1, -2))
        y_C = torch.sum(y_CC, axis=(-1, -2)) + torch.sum(y_CN, axis=(-1, -2))

        return y_N, y_C

    def save_weight(self,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savetxt("{}/Nw1.txt".format(save_path), self.Nw1.detach().numpy())
        np.savetxt("{}/Nw2.txt".format(save_path), self.Nw2.detach().numpy())
        np.savetxt("{}/Nw3.txt".format(save_path), self.Nw3.detach().numpy())
        np.savetxt("{}/Cw1.txt".format(save_path), self.Cw1.detach().numpy())
        np.savetxt("{}/Cw2.txt".format(save_path), self.Cw2.detach().numpy())
        np.savetxt("{}/Cw3.txt".format(save_path), self.Cw3.detach().numpy())

class BPNet_linear(nn.Module):
    def __init__(self):
        super(BPNet_linear, self).__init__()
        N_net = [7,5,5,1]
        C_net = [7,5,5,1]

        self.Nw1 = nn.Parameter(torch.randn(N_net[0],N_net[1])/1)
        self.Nb1 = nn.Parameter(torch.randn(N_net[1])/1)
        self.Nw2 = nn.Parameter(torch.randn(N_net[1],N_net[2])/1)
        self.Nb2 = nn.Parameter(torch.randn(N_net[2])/1)
        self.Nw3 = nn.Parameter(torch.randn(N_net[2],N_net[3])/1)
        self.Nb3 = nn.Parameter(torch.randn(N_net[3])/1)

        self.Cw1 = nn.Parameter(torch.randn(C_net[0],C_net[1])/1)
        self.Cb1 = nn.Parameter(torch.randn(C_net[1])/1)
        self.Cw2 = nn.Parameter(torch.randn(C_net[1],C_net[2])/1)
        self.Cb2 = nn.Parameter(torch.randn(C_net[2])/1)
        self.Cw3 = nn.Parameter(torch.randn(C_net[2],C_net[3])/1)
        self.Cb3 = nn.Parameter(torch.randn(C_net[3])/1)

    def forward(self, x_N, x_C, dx_NN, dx_CN, dx_NC, dx_CC):
        ap1_NN = torch.matmul(dx_NN, self.Nw1)
        ap2_NN = torch.matmul(ap1_NN, self.Nw2)
        y_NN = torch.matmul(ap2_NN, self.Nw3)

        ap1_CN = torch.matmul(dx_CN, self.Nw1)
        ap2_CN = torch.matmul(ap1_CN, self.Nw2)
        y_CN = torch.matmul(ap2_CN, self.Nw3)

        ap1_CC = torch.matmul(dx_CC, self.Cw1)
        ap2_CC = torch.matmul(ap1_CC, self.Cw2)
        y_CC = torch.matmul(ap2_CC, self.Cw3)

        ap1_NC = torch.matmul(dx_NC, self.Cw1)
        ap2_NC = torch.matmul(ap1_NC, self.Cw2)
        y_NC = torch.matmul(ap2_NC, self.Cw3)

        y_N = torch.sum(y_NN, axis=(-1, -2))# + torch.sum(y_NC, axis=(-1, -2))
        y_C = torch.sum(y_CC, axis=(-1, -2))# + torch.sum(y_CN, axis=(-1, -2))

        return y_N, y_C

    def save_weight(self,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savetxt("{}/Nw1.txt".format(save_path), self.Nw1.detach().numpy())
        np.savetxt("{}/Nw2.txt".format(save_path), self.Nw2.detach().numpy())
        np.savetxt("{}/Nw3.txt".format(save_path), self.Nw3.detach().numpy())
        np.savetxt("{}/Cw1.txt".format(save_path), self.Cw1.detach().numpy())
        np.savetxt("{}/Cw2.txt".format(save_path), self.Cw2.detach().numpy())
        np.savetxt("{}/Cw3.txt".format(save_path), self.Cw3.detach().numpy())

def Train_network(train_data, valid_data, epoch_number, model_save_path, model_save_name):
    net = BPNet_linear()
    optimizer = optim.Adam(net.parameters())
    info_file = open("{}/{}.dat".format(model_save_path, model_save_name),"w")

    # save model
    optimize_model = None
    optimize_loss = 1000
    optimize_index = 0
    for epoch_index in range(epoch_number):
        train_generator = train_data.Generator(True)
        valid_generator = valid_data.Generator(False)

        avg_train_loss = 0
        avg_valid_loss = 0
        net.train()
        for batch_xN, batch_xC, batch_dxNN, batch_dxNC, batch_dxCN, batch_dxCC, batch_fN, batch_fC in train_generator:
            optimizer.zero_grad()
            yN_pred, yC_pred = net(batch_xN, batch_xC, batch_dxNN, batch_dxCN, batch_dxNC, batch_dxCC)
            loss = torch.sum(torch.abs(yN_pred - batch_fN)) + torch.sum(torch.abs(yC_pred - batch_fC))
            avg_loss = torch.mean(torch.abs(yN_pred - batch_fN)) + torch.mean(torch.abs(yC_pred - batch_fC))
            loss.backward()
            optimizer.step()
            avg_train_loss += float(avg_loss)
        net.eval()
        MAEN = 0
        MAEC = 0
        MSEN = 0
        MSEC = 0
        for batch_xN, batch_xC, batch_dxNN, batch_dxNC, batch_dxCN, batch_dxCC, batch_fN, batch_fC in valid_generator:
            yN_pred, yC_pred = net(batch_xN, batch_xC, batch_dxNN, batch_dxCN, batch_dxNC, batch_dxCC)
            MSEN += torch.mean(torch.pow(yN_pred - batch_fN, 2))
            MSEC += torch.mean(torch.pow(yC_pred - batch_fC, 2))
            MAEN += torch.mean(torch.abs(yN_pred - batch_fN))
            MAEC += torch.mean(torch.abs(yC_pred - batch_fC))

            avg_loss = torch.mean(torch.abs(yN_pred - batch_fN)) + torch.mean(torch.abs(yC_pred - batch_fC))
            avg_valid_loss += float(avg_loss)

        MSEN = MSEN / valid_data.config_number
        MSEC = MSEC / valid_data.config_number

        MAEN = MAEN / valid_data.config_number
        MAEC = MAEC / valid_data.config_number

        RMSEN = torch.sqrt(MSEN)
        RMSEC = torch.sqrt(MSEC)

        avg_valid_loss = avg_valid_loss / valid_data.config_number
        avg_train_loss = avg_train_loss / train_data.config_number

        if avg_valid_loss < optimize_loss:
            print("optimize epoch : ", epoch_index)
            optimize_loss = avg_valid_loss
            optimize_index = epoch_index
            torch.save(net.state_dict(), "{}/{}.pth".format(model_save_path,model_save_name))
            net_traced = torch.jit.trace(net, (batch_xN[0], batch_xC[0], batch_dxNN[0], batch_dxNC[0], batch_dxCN[0], batch_dxCC[0]))
            net_traced.save("{}/{}.pt".format(model_save_path,model_save_name))
            net.save_weight("{}/Shell_weights/".format(model_save_path))
        info_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch_index,avg_train_loss,avg_valid_loss,RMSEN,RMSEC,MAEN,MAEC))
        info_file.flush()
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(epoch_index,avg_train_loss,avg_valid_loss,RMSEN,RMSEC,MAEN,MAEC))

if __name__ == "__main__":
    TRAIN_DATA_FOLDER = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/TrainInput"
    VALID_DATA_FOLDER = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/ValidInput"

    train_shell_dataset = Data_loader(  xN_path="{}/xn.npy".format(TRAIN_DATA_FOLDER),
                                        xC_path="{}/xc.npy".format(TRAIN_DATA_FOLDER),
                                        dxNN_path="{}/xnnd.npy".format(TRAIN_DATA_FOLDER),
                                        dxNC_path="{}/xncd.npy".format(TRAIN_DATA_FOLDER),
                                        dxCN_path="{}/xcnd.npy".format(TRAIN_DATA_FOLDER),
                                        dxCC_path="{}/xccd.npy".format(TRAIN_DATA_FOLDER),
                                        fN_path = "{}/fn_short.npy".format(TRAIN_DATA_FOLDER),
                                        fC_path = "{}/fc_short.npy".format(TRAIN_DATA_FOLDER))
    valid_shell_dataset = Data_loader(  xN_path="{}/xn.npy".format(VALID_DATA_FOLDER),
                                        xC_path="{}/xc.npy".format(VALID_DATA_FOLDER),
                                        dxNN_path="{}/xnnd.npy".format(VALID_DATA_FOLDER),
                                        dxNC_path="{}/xncd.npy".format(VALID_DATA_FOLDER),
                                        dxCN_path="{}/xcnd.npy".format(VALID_DATA_FOLDER),
                                        dxCC_path="{}/xccd.npy".format(VALID_DATA_FOLDER),
                                        fN_path = "{}/fn_short.npy".format(VALID_DATA_FOLDER),
                                        fC_path = "{}/fc_short.npy".format(VALID_DATA_FOLDER))

    # train_generator = train_dataset.Generator()

    Train_network(  train_shell_dataset,
                    valid_shell_dataset ,
                    100000,
                    "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/data/Models/" ,
                    "NaCl_shell_force_linear")


