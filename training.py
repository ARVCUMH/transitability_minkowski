import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("/home/arvc/Desktop/Antonio/minkowski/scripts/examples"))
from minkunet import MinkUNet34C
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import yaml
from dataset import preprocessingDataset2f


def compute_th(net, data, device):
    print("Calculando threshold")
    net.eval()
    optimal_th_list = []
    with torch.no_grad():
        for i, cloud in enumerate(tqdm(data)):
            test_coords, test_feats, test_label = cloud
            test_in_field = ME.TensorField(test_feats.to(dtype=torch.float32),
                                      coordinates=test_coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                      minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device)

            test_label = test_label.to(device)
            test_output = net(test_in_field.sparse())
            logit = test_output.slice(test_in_field)
            test_label_gt = test_label.cpu().numpy()
            precision, recall, thresholds = metrics.precision_recall_curve(test_label_gt, logit.F.cpu().numpy())
            fscore = (2 * precision * recall) / (precision + recall)
            # locate the index of the largest f score
            ix = np.argmax(fscore)
            print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
            optimal_th_list.append(thresholds[ix])
        return sum(optimal_th_list)/len(optimal_th_list)


def test(net, data, device, validation_loss, mean_th):
    net.eval()
    all_accuracy = []
    all_recall = []
    all_precision = []
    print("Calculando métricas")
    with torch.no_grad():
        for i, cloud in enumerate(tqdm(data)):
            test_coords, test_feats, test_label = cloud
            test_in_field = ME.TensorField(test_feats.to(dtype=torch.float32),
                                      coordinates=test_coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                      minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device)

            test_label = test_label.to(device)
            test_output = net(test_in_field.sparse())
            logit = test_output.slice(test_in_field)
            val_loss = criterion(logit.F, test_label.unsqueeze(1).float())
            test_label_gt = test_label.cpu().numpy()
            pred=np.where(logit.F.cpu().numpy() > mean_th, 1, 0)
            all_accuracy.append(metrics.accuracy_score(pred, test_label_gt))
            all_recall.append(metrics.recall_score(test_label_gt, pred))
            all_precision.append(metrics.precision_score(test_label_gt, pred))
            validation_loss.append(val_loss.item())
            print('\t\t Loss:', val_loss.item())


        mean_acc=sum(all_accuracy) / len(all_accuracy)
        mean_r=sum(all_recall) / len(all_recall)
        mean_p=sum(all_precision) / len(all_precision)
        print('Mean Accuracy all batches of validation:',mean_acc , '\t Threshold:', mean_th)
        print('Mean Precision all batches of validation:', mean_p, '\t Threshold:', mean_th)
        print('Mean Recall all batches of validation:', mean_r, '\t Threshold:', mean_th)


    return mean_acc, validation_loss,mean_r,mean_p

def load_data(root_train, root_valid, voxel_size):

    full_dataset = preprocessingDataset2f(root_train,mode="train", voxel=voxel_size)
    train_data_final = DataLoader(full_dataset, batch_size=1, collate_fn=ME.utils.batch_sparse_collate,num_workers=0,shuffle=True)

    valid_data= preprocessingDataset2f(root_valid,mode="valid", voxel=voxel_size)
    valid_data_final = DataLoader(valid_data, batch_size=8, collate_fn=ME.utils.batch_sparse_collate,num_workers=0,shuffle=True)

    return train_data_final, valid_data_final

def yaml_loader():
    with open("config.yaml", 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml

if __name__ == '__main__':

    #PARAMETROS DE ENTRENAMIENTO normales y z sin normalizar

    parsed_yaml=yaml_loader()
    voxel_size = parsed_yaml["voxel_size"]
    epochs = parsed_yaml["epochs"]
    train_root= parsed_yaml["root_train"]
    valid_root= parsed_yaml["root_valid"]
    saved= parsed_yaml["save_model"]

    training_loss = []
    validation_loss = []

    isExist = os.path.exists("Voxel"+str(voxel_size))
    if not isExist:
        os.mkdir("Voxel"+str(voxel_size))

    best_metric = 0
    best_threshold = []
    train_data_final, valid_data_final = load_data(train_root, valid_root, voxel_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    model = MinkUNet34C(2, 1).to(device)
    optimizer = SGD(model.parameters(), lr=parsed_yaml["lr"])

    for epoch in range(epochs):
        model.train()
        print("Epoch:", epoch)
        for i,data in enumerate(tqdm(train_data_final)):
            coords, feats, label= data
            in_field = ME.TensorField(feats.to(dtype=torch.float32),coordinates=coords,quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                      minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,device=device)
            # Forward
            input = in_field.sparse()
            output = model(input)
            out_field = output.slice(in_field)
            # Loss
            loss = criterion(out_field.F, label.to(device).unsqueeze(1).float()) #.F son las features, esta funcion es una clase abstracta para calcular el gradiente
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iteration batch: ', i, ', Loss: ', loss.item())
            training_loss.append(loss.item())
        #VALIDATION EACH EPOCH
        th_opt = compute_th(model, valid_data_final, device)
        mean_acc, validation_loss,mean_r,mean_p=test(model, valid_data_final, device, validation_loss, th_opt)

        if best_metric < mean_p:
            print("Model saved, epoch:", epoch)
            best_metric = mean_p
            torch.save(model.state_dict(),
            'Voxel'+str(voxel_size)+'/BestModel'+str(epoch)+'_th_'+str(th_opt)+"voxel_size"+
            str(voxel_size)+'_'+str(mean_p)+'.pth')
            print("------------------")
        else:
            print("It does not saved the model, since does not improve the previous model saved", epoch)
            print("------------------")
