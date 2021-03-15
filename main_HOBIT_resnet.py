import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import torch.fft as fft
import random
import argparse

from torch.utils import data
from loaders.Patient_data_loader import Patient_data_loader
from loaders.Patient_data_loader_all import Patient_data_loader_all
from loaders.Simulation_ich_loader2  import Simulation_ICH_loader
from models.unet import Unet
from models.resBlock import ResBlock
from utils.train import BayesianQSM_train
from utils.medi import SMV_kernel, dipole_kernel, DLL2
from utils.loss import *
from utils.files import *
from utils.test import compute_rmse, compute_ssim, compute_hfen, compute_R_ICH

'''
    FINE of resnet on top of pre-trained unet3d and resnet
'''
if __name__ == '__main__':

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0, 1')
    parser.add_argument('--loader', type=int, default=1)  # 0: in-vivo data, 1: simulated data
    parser.add_argument('--patient_type', type=str, default='ICH')  # or MS_old, MS_new
    parser.add_argument('--patientID', type=int, default=8)
    parser.add_argument('--optm', type=int, default=0) # 0: Adam, 1: LBFGS
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--rho', type=int, default=30)
    opt = {**vars(parser.parse_args())}

    patient_type = opt['patient_type']
    patientID = opt['patientID']

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM'

    # dataloader
    if opt['loader'] == 0:  # in-vivo rdf 
        dataLoader_train = Patient_data_loader(
            patientType=opt['patient_type'], 
            patientID=str(opt['patientID']),
            flag_input=2
        )
        print('Using real RDF')
    elif opt['loader'] == 1:  # simulated rdf from FINE
        dataLoader_train = Simulation_ICH_loader(split='test', patientID=opt['patient_type']+str(opt['patientID']))
        print('Using simulated RDF')
    # parameters
    lr = 7e-4  # 7e-4 for HOBIT
    batch_size = 1
    B0_dir = (0, 0, 1)
    voxel_size = dataLoader_train.voxel_size
    volume_size = dataLoader_train.volume_size

    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

    # network of HOBIT
    unet3d = Unet(
        input_channels=1, 
        output_channels=1, 
        num_filters=[2**i for i in range(5, 10)],
        use_deconv=1,
        flag_rsa=0
    )
    unet3d.to(device0)
    weights_dict = torch.load('./weights/unet3d_fine.pt')
    unet3d.load_state_dict(weights_dict)

    resnet = ResBlock(
        input_dim=2, 
        filter_dim=32,
        output_dim=1
    )
    resnet.to(device)
    weights_dict = torch.load('./weights/resnet_fine.pt', map_location='cuda:0')
    resnet.load_state_dict(weights_dict)

    if opt['loader'] == 0:
        dataDir = rootDir + '/Data_with_N_std/20190920_MEDI_3mm/ICH{}'.format(opt['patientID'])
        rootDir = rootDir + '/result_2nets/invivo'
    else:
        dataDir = rootDir + '/Data_with_N_std/20190920_MEDI_3mm/ICH{}'.format(opt['patientID'])
        rootDir = rootDir + '/result_2nets/simu'
    mask_srd = np.real(load_mat(dataDir+'/ich_srd_{}.mat'.format(patientID), varname='mask_srd'))

    # optimizer
    if opt['optm'] == 1:
        optimizer = optim.LBFGS(resnet.parameters(), history_size=3, max_iter=4)
        K = 1
    else:
        optimizer = optim.Adam(resnet.parameters(), lr=lr, betas=(0.5, 0.999))
        K = 4

    epoch = 0
    niter = 5
    loss_iters = np.zeros(niter)
    while epoch < 2:
        epoch += 1
        # for idx, (rdf_inputs, rdfs, masks, weights, wGs, D) in enumerate(trainLoader):  # for real rdf
        for idx, (qsms, rdf_inputs, rdfs, wG, weights, masks_csf, D) in enumerate(trainLoader):  # for simulated rdf

            if epoch == 1:
                unet3d.eval(), resnet.eval()
                unet3d.to(device)
                rdf_inputs = rdf_inputs.to(device, dtype=torch.float)
                with torch.no_grad():
                    qsm_inputs = unet3d(rdf_inputs).cpu().detach()

            else:
                # to GPU device
                rdf_inputs = rdf_inputs.to(device, dtype=torch.float)
                qsm_inputs1 = qsm_inputs.to(device, dtype=torch.float)
                inputs_cat = torch.cat((rdf_inputs, qsm_inputs1), dim=1)
                rdfs = rdfs.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)
                masks_csf = masks_csf.to(device, dtype=torch.float)
                D = D.to(device, dtype=torch.complex64)[None, ...]

                # label
                try:
                    chi_true = np.squeeze(np.asarray(qsms))
                    mask_csf = np.squeeze(np.asarray(masks_csf.cpu().detach()))
                    wG = wG.to(device, dtype=torch.float)
                    flag = 1
                except:
                    flag = 0

                # save initial QSM
                with torch.no_grad():
                    outputs = resnet(inputs_cat)
                    QSMnet_hobit = np.squeeze(np.asarray(outputs.cpu().detach()))

                print('Saving initial results')
                QSMnet_hobit = QSMnet_hobit - np.mean(QSMnet_hobit[mask_csf==1])
                adict = {}
                adict['QSMnet_hobit'] = QSMnet_hobit
                sio.savemat(rootDir+'/QSMnet_hobit_ICH={}_alpha={}_optm={}.mat'.format(opt['patientID'], \
                opt['alpha'], opt['optm']), adict)

    epoch = 0
    t0 = time.time()
    mu = torch.zeros(volume_size, device=device)
    alpha = opt['alpha'] * torch.ones(1, device=device)  # 0.5
    rho = opt['rho'] * torch.ones(1, device=device)  # 30
    lambda_TV = 0 * torch.ones(1, device=device)
    P = 1 * torch.ones(1, device=device)
    while epoch < niter:
        epoch += 1
        # dll2 update
        with torch.no_grad():
            dc_layer = DLL2(D[0, 0, ...], weights[0, 0, ...], rdfs[0, 0, ...], wG[0, 0, ...], \
                            device=device, lambda_TV=lambda_TV, P=P, alpha=alpha, rho=rho)
            x = dc_layer.CG_iter(phi=outputs[0, 0, ...], mu=mu, max_iter=100)
            x = P * x

        # network update
        for k in range(K):
            def closure():
                optimizer.zero_grad()
                outputs = resnet(inputs_cat)
                outputs_cplx = outputs.type(torch.complex64)
                # loss
                RDFs_outputs = torch.real(fft.ifftn((fft.fftn(outputs_cplx, dim=[2, 3, 4]) * D), dim=[2, 3, 4]))
                diff = torch.abs(rdfs - RDFs_outputs)
                loss_fidelity = (1 - alpha) * 0.5 * torch.sum((weights*diff)**2)
                loss_l2 = rho * 0.5 * torch.sum((x - outputs[0, 0, ...] + mu)**2)
                loss = loss_fidelity + loss_l2
                # loss = loss_fidelity
                loss.backward()
                return loss  
            optimizer.step(closure)

            # forward again to compute fidelity loss
            outputs = resnet(inputs_cat)  
            outputs_cplx = outputs.type(torch.complex64)
            # loss
            RDFs_outputs = torch.real(fft.ifftn((fft.fftn(outputs_cplx, dim=[2, 3, 4]) * D), dim=[2, 3, 4]))
            diff = torch.abs(rdfs - RDFs_outputs)
            loss_fidelity = torch.sum((weights*diff)**2)
            fidelity_fine = 'epochs: [%d/%d], Ks: [%d/%d], time: %ds, Fidelity loss: %f' % (epoch, niter, k+1, K, time.time()-t0, loss_fidelity.item())
            print(fidelity_fine)

        # dual update
        with torch.no_grad():
            mu = mu + x - outputs[0, 0, ...]

        # metrics
        if flag == 1:
            chi_recon = np.squeeze(np.asarray(x.cpu().detach()))
            metrics_dll2 = 'DLL2: RMSE = {}, R_ICH = {}'.format(compute_rmse(chi_recon, chi_true, mask_csf), \
                    compute_R_ICH(chi_recon, chi_true, mask_srd))
            print(metrics_dll2)
            # chi_recon = np.squeeze(np.asarray(x.cpu().detach()))
            # metrics_dll2 = 'DLL2: RMSE = {}, SSIM = {}'.format(compute_rmse(chi_recon, chi_true, mask_csf), \
            #         compute_ssim(torch.tensor(chi_recon[np.newaxis, np.newaxis, ...]), torch.tensor(chi_true[np.newaxis, np.newaxis, ...]), masks_csf))
            # print(metrics_dll2)
            # chi_recon = np.squeeze(np.asarray(outputs.cpu().detach()))
            # metrics_fine = 'FINE: RMSE = {} SSIM = {}'.format(compute_rmse(chi_recon, chi_true, mask_csf), \
            #         compute_ssim(torch.tensor(chi_recon[np.newaxis, np.newaxis, ...]), torch.tensor(chi_true[np.newaxis, np.newaxis, ...]), masks_csf))
            # print(metrics_fine)
    # save
    DLL2 = np.squeeze(np.asarray(x.cpu().detach()))
    DLL2 = DLL2 - np.mean(DLL2[mask_csf==1])
    adict = {}
    adict['DLL2'] = DLL2
    sio.savemat(rootDir+'/DLL2_ICH={}_alpha={}_optm={}.mat'.format(opt['patientID'], \
                opt['alpha'], opt['optm']), adict)

    # save
    FINE = resnet(inputs_cat)[:, 0, ...]
    FINE = np.squeeze(np.asarray(FINE.cpu().detach()))
    FINE = FINE - np.mean(FINE[mask_csf==1])
    adict = {}
    adict['FINE'] = FINE
    sio.savemat(rootDir+'/FINE_ICH={}_alpha={}_optm={}.mat'.format(opt['patientID'], \
                opt['alpha'], opt['optm']), adict)

    Truth = chi_true - np.mean(chi_true[mask_csf==1])
    adict = {}
    adict['Truth'] = Truth
    sio.savemat(rootDir+'/Truth_ICH={}_alpha={}_optm={}.mat'.format(opt['patientID'], \
                opt['alpha'], opt['optm']), adict)
