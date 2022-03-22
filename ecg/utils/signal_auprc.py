"""Functions for training and running group classification."""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.utils.extmath import softmax
from scipy.special import expit
from sklearn.metrics import f1_score, classification_report, confusion_matrix, average_precision_score, roc_auc_score

import torch
import torchvision
import tqdm
import pdb

import hha

def run(
        num_epochs=150,
        file_list='FileList_hh_firstEncountersValTest.csv', 
        modelname="resnet18",
        dimension=1,
        binary=True,
        tasks="Group",
        nodes=2,
        preprocessed=True,
        pretrained=False,
        output=None,
        device=None,
        n_train_patients=None,
        num_workers=5,
        batch_size=20,
        seed=0,
        optimizer='SDG',
        lr_step_period=15,
        bias=True,
        weighted=False,
        oversample=False,
        downsample=False,
        run_test=False,
        rank_auprc=False, 
        ):
    """Trains/tests Hypertension/Hypertrophic cardiomyopathy classification  model.

    Args:
        num_epochs (int, optional): Number of epochs during training
            Defaults to 100.
        file_list (str, required): Name of file list to use, ie list of samples to train/test
            Defaults to 'FileList_hh_firstEncountersValTest.csv'
        modelname (str, required): Name of model architecture
            Defaults to resnet18
        dimension (int, required): Dimension of model architecture
            Defaults to 1 (1D models)
        binary (bool, required): Whether to train binary classification
            Defaults to True.
        tasks (str, optional): Name of task to predict. Options are the headers
            of FileList.csv.
            Defaults to Group.
        nodes (int, required): numbers of nodes, representing number of classes,
            Defaults to 1, for binary case.
        preprocessed (bool, optional): Whether to use preprocessed signals, ie notch and baseline filtered
            Defaults to True.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to False.
        output (str or None, optional): Name of directory to place outputs
            Defaults to None (replaced by output/ecg/<modelname>_<pretrained/random>/).
        device (str or None, optional): Name of device to run on. See
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            for options. If ``None'', defaults to ``cuda'' if available, and ``cpu'' otherwise.
            Defaults to ``None''.
        n_train_patients (str or None, optional): Number of training patients. Used to ablations
            on number of training patients. If ``None'', all patients used.
            Defaults to ``None''.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 5.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 20.
        seed (int, optional): Seed for random number generator.
            Defaults to 0.
        optimizer (str, optional) Specify what optimizer to use.
            Defaults to SDG.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            If ``None'', learning rate is not decayed.
            Defaults to 15.
        bias (float, optional): Add bias to final layer of model, 
            Defaults to 0.0
        weighted (bool, optional): Decides whether or not to weight classes during training
            Default to False.
        oversample (bool, optional): Decides whether or not to oversample minority classes during training
            Default to False.
        downsample(bool, optional): Decides whether or not to downsample signal by half
            Default to False.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        rank_auprc (bool, optional): Whether or not to run on test on higheeest ranked auPRC.
            Defaults to False.
    """

    ## Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Setting default output directory
    print(output)
    if output is not None:
        output = os.path.join(output, "ecg", "{}_{}_{}_{}_{}_{}_{}_{}".format(modelname, 
                                                                "pretrained" if pretrained else "random", 
                                                                "weighted" if weighted else "nonweighted",
                                                                "oversampled" if oversample else "nonoversampled",
                                                                "bias" if bias else "nobias", 
                                                                optimizer, 
                                                                batch_size,
                                                                "downsample" if downsample  else "None"
                                                                ))
    else:
        output = os.path.join('output', "ecg", "{}_{}_{}_{}_{}_{}_{}_{}".format(modelname, 
                                                                "pretrained" if pretrained else "random", 
                                                                "weighted" if weighted else "nonweighted",
                                                                "oversampled" if oversample else "nonoversampled",
                                                                "bias" if bias else "nobias", 
                                                                optimizer, 
                                                                batch_size,
                                                                "downsample" if downsample  else "None"
                                                                ))

    ### Making directory is does not exist
    os.makedirs(output, exist_ok=True)

    ## Setting device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Setting up model
    ### VGG
    if modelname in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
        model = hha.models.vgg_1D.VGG_1D(modelname, num_classes=nodes)
     ### ResNets   
    if modelname in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']:
        model = eval('hha.models.resnet_1D.' + modelname + '(num_classes=' + str(nodes) +')')
    ### DenseNets   
    if modelname in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:
        model = eval('hha.models.densenet_1D.' + modelname + '(num_classes=' + str(nodes) +')')
    ### Mayo 
    if modelname in ['MayoV1', 'MayoV2', 'MayoV3']:
        assert(dimension==2)
        model = eval('hha.models.mayo_2D.' + modelname + '(num_classes=' + str(nodes) +')')
    ### Custom 
    if modelname in ['c_152', 'c_101', 'c_50']:
        #model = hha.models.custom_1D.custom_1D(modelname, num_classes=nodes)
        replace_stride_with_dilation = [5, 7, 9]
        replace_stride_with_dilation = [True, True, True]
        kernel_size = 3
        model = eval('hha.models.custom_1D.' + modelname + '(num_classes=' + str(nodes) +', replace_stride_with_dilation=replace_stride_with_dilation, kernel_size=kernel_size)')


    if pretrained:
        ## Implementing data parallelism at the module level.
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
        pt_path = 'clinical_labels'
        pretrained_path = os.path.join(pt_path, "ecg", "{}_{}_{}_{}_{}_{}_{}".format(modelname, 
                                                        "random", 
                                                        "nonweighted",
                                                        "nonoversample",
                                                        "nobias", 
                                                        "adam", 
                                                        batch_size))
        checkpoint = torch.load(os.path.join(pretrained_path, "best.pt"))
        print(os.path.join(pretrained_path, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        print("pretrained model loaded")
        model = model.module 
        ## Modifying the last layer of nodes
        model.classifier = torch.nn.Linear(model.classifier.in_features, nodes)
    print(model) 
    print(file_list)

    ## Initializing well:atural log(pos/neg) for final bias term #natural log(pos/total) for final bias term
    if bias:
        if nodes == 1:
            bias_terms = [-0.83] 
            model.classifier.bias.data = torch.tensor(bias_terms)
        ## TODO: Add an option for normal bias setting etc
        if nodes == 3: 
            bias_terms = [0.0, -0.85, -3.92] # This is for HTN, HCM and athletes
            model.classifier.bias.data = torch.tensor(bias_terms)
    if not bias:
        bias_terms = [0.0] * nodes
        model.classifier.bias.data = torch.tensor(bias_terms)

    ## Implementing data parallelism at the module level.
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    if lr_step_period is None:
        lr_step_period = math.inf
    if optimizer == 'adam':
        learning_rate = 1e-3
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    baseline,notch = not preprocessed, not preprocessed
    ## Computing mean and std
    mean,std = hha.utils.get_mean_and_std(hha.datasets.Ecg(split="train", 
                                        file_list=file_list, 
                                        baseline=baseline, 
                                        notch=notch, 
                                        preprocessed=preprocessed), samples=1000)

    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "dimension": dimension, 
              "preprocessed": preprocessed,
              "downsample": downsample,
              "baseline": baseline,
              "notch": notch,
              "file_list": file_list
              }

    train_dataset = hha.datasets.Ecg(split="train", **kwargs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset
                                                    , batch_size=batch_size
                                                    , num_workers=num_workers
                                                    , shuffle=True
                                                    , pin_memory=(device.type == "cuda")
                                                    , drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(hha.datasets.Ecg(split="validate", **kwargs)
                                                , batch_size=batch_size
                                                , num_workers=num_workers
                                                , shuffle=True
                                                , pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'validate': val_dataloader}

    outcome  = train_dataset.outcome
    targets = [j[2] for j in outcome ]
    class_count = np.unique(targets, return_counts=True)[1]
    print(class_count)

    if oversample and not weighted: 
        #############
        # Oversample the minority classes
        outcome  = train_dataset.outcome
        targets = [j[2] for j in outcome ]
        class_count = np.unique(targets, return_counts=True)[1]
        print(class_count)
        weight = 1. / class_count
        samples_weight = torch.from_numpy(np.array([weight[int(float(t))] for t in targets]))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight)) 
        weighted_loader = torch.utils.data.DataLoader(train_dataset
                                                        , batch_size=batch_size
                                                        , num_workers=num_workers
                                                        , shuffle=False
                                                        , pin_memory=(device.type == "cuda")
                                                        , drop_last=True
                                                        , sampler=sampler)
        dataloaders = {'train': weighted_loader, 'validate': val_dataloader}
        #############

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        bestauPRC = float(0.)
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'validate']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)
                ## Running current epoch
                loss, yhat, y, epoch_metrics, __ = hha.utils.signal_auprc.run_epoch(model
                                                        , dataloaders[phase]
                                                        , phase == "train"
                                                        , optim
                                                        , device
                                                        , binary=binary
                                                        , weighted=weighted)
                ## Writing to file
                if binary:
                    threshold = 0.5
                    yhat = expit(yhat)
                    metrics_predictions_ndx = 1
                    predictions = epoch_metrics[:, metrics_predictions_ndx] 
                    calculated_metrics = pd.DataFrame(log_epoch_metrics(epoch_metrics))
                    print(roc_auc_score(y, yhat, average='weighted'))
                    print(average_precision_score(y, yhat, average='weighted'))
                    auprc = average_precision_score(y, yhat, average='weighted')
                    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch
                                            , phase
                                            , loss
                                            , calculated_metrics['0.0']['loss']
                                            , calculated_metrics['1.0']['loss']
                                            , f1_score(y, predictions, average='weighted')
                                            , calculated_metrics['0.0']['f1-score']
                                            , calculated_metrics['1.0']['f1-score']
                                            , roc_auc_score(y, yhat, average='weighted')
                                            , average_precision_score(y, yhat, average='weighted')
                                            , time.time() - start_time
                                            , y.size
                                            , sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count()))
                                            , sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))
                                            , batch_size))
                else:
                    yhat = softmax(yhat)
                    metrics_predictions_ndx = 1
                    predictions = epoch_metrics[:, metrics_predictions_ndx]
                    y_encode = np.eye(np.int(y.max()+1))[y.astype(int)]
                    calculated_metrics = pd.DataFrame(log_epoch_metrics(epoch_metrics))
                    print(roc_auc_score(y_encode, yhat, average='weighted'))
                    print(average_precision_score(y_encode, yhat , average='weighted'))
                    auprc = average_precision_score(y_encode, yhat, average='weighted')

                    per_class_loss = calculated_metrics[[str(j) for j in np.arange(0, nodes).astype(float)]].loc['loss'].values.tolist()
                    per_class_f1score = calculated_metrics[[str(j) for j in np.arange(0, nodes).astype(float)]].loc['f1-score'].values.tolist()

                    line_out = [epoch, phase, loss] + per_class_loss + [f1_score(y, predictions, average='weighted')] + per_class_f1score + [roc_auc_score(y_encode, yhat, average='weighted')] + [average_precision_score(y_encode, yhat, average='weighted')] + [time.time() - start_time] + [y.size] + [sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count()))] + [sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))] + [batch_size]
                    f.write(",".join(str(np.round(x,4)) if isinstance(x, np.float32) else str(x) for x in line_out) + '\n')
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'auprc': auprc,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            
            if auprc > bestauPRC:
                torch.save(save, os.path.join(output, "best_auprc.pt"))
                bestauPRC = auprc
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss
        
        if rank_auprc:
            # Loading best weights for highest auPRC
            checkpoint = torch.load(os.path.join(output, "best_auprc.pt"))
            print(os.path.join(output, "best_auprc.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best auPRC {} from epoch {}\n".format(checkpoint["auprc"], checkpoint["epoch"]))
            f.flush()
        else:
            # Loading best weights for lowest loss
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            print(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        if run_test:
            for split in ["validate", "test"]:
                # Performance without test-time augmentation
                print("Running on ....", split)
                dataloader = torch.utils.data.DataLoader(hha.datasets.Ecg(split=split, **kwargs)
                                                        ,batch_size=batch_size
                                                        , num_workers=num_workers
                                                        , shuffle=True
                                                        , pin_memory=(device.type == "cuda"))
                loss, yhat, y, epoch_metrics, fnames = hha.utils.signal_auprc.run_epoch(model,
                                                                                    dataloader, 
                                                                                    False, 
                                                                                    None, 
                                                                                    device, 
                                                                                    binary=binary,
                                                                                    weighted=weighted)
                # Write full performance to file
                pred_out = os.path.join(output, "{}_predictions.csv".format(split))
                if rank_auprc:
                    pred_out = os.path.join(output, "{}_predictions_auprc.csv".format(split))
                boot_out = os.path.join(output, "{}_bootstrap.csv".format(split))
                if rank_auprc:
                    boot_out = os.path.join(output, "{}_bootstrap_auprc.csv".format(split))

                if binary:
                    yhat = expit(yhat)
                    with open(pred_out, "w") as g:
                        g.write("{},{},{}\n".format('filename', 'true_class', 'prob_class'))
                        for (filename, true, pred) in zip(fnames, y, yhat):
                            g.write("{},{},{:.4f}\n".format(filename, true, pred[0]))
                            g.flush()
                    threshold = 0.5
                    predictions = np.zeros(yhat.shape, dtype=int)
                    predictions[yhat < threshold] = 0 
                    predictions[yhat >= threshold] = 1
                    print(classification_report(y, predictions)) #, target_names=target_names))
                    print(pd.DataFrame(confusion_matrix(y, predictions)))
                    with open(boot_out, "w") as g:
                        g.write("Split,  metric,  average, min,  max \n")
                        g.write("{},  AUC,  {:.3f}, {:.3f},  {:.3f}\n".format(split, *hha.utils.bootstrap(y, yhat, roc_auc_score) ))
                        g.write("{},  AP,   {:.3f}, {:.3f},  {:.3f}\n".format(split, *hha.utils.bootstrap(y, yhat, average_precision_score) ))
                        g.write("{},  F1,   {:.3f}, {:.3f},   {:.3f}\n".format(split, *hha.utils.bootstrap(y, predictions, f1_score) ))
                        g.flush()

                else:
                    yhat = softmax(yhat)
                    with open(pred_out, "w") as g:
                        headers = ['filename', 'true_class'] + ['prob_' + str(i) for i in np.arange(0, nodes) ] + ['\n']
                        g.write(",".join(x for x in headers))
                        for (filename, true, pred) in zip(fnames, y, yhat):
                            line_out = [filename, true] +  [i for i in pred]
                            g.write(",".join(str(np.round(x,4)) if isinstance(x, np.float32) else x for x in line_out) + '\n' )
                            g.flush()

                    pred = np.argmax(yhat, axis=1)
                    print(f1_score(y, pred, average=None))
                    print(classification_report(y, pred)) #, target_names=target_names))
                    print(pd.DataFrame(confusion_matrix(y, pred)))
                    y_encode = np.eye(np.int(y.max()+1))[y.astype(int)]
                    pred_encode = np.eye(np.int(y.max()+1))[pred.astype(int)]
                    with open(boot_out, "w") as g:
                        g.write("Split, group ,metric,  average, min,  max \n")
                        for node in range(0, nodes):
                            g.write("{}, {},  AUC,  {:.3f}, {:.3f} , {:.3f}\n".format(split, node, *hha.utils.bootstrap(y_encode[:,node], pred_encode[:,node], roc_auc_score) ))
                            g.write("{}, {},   AP,   {:.3f}, {:.3f} , {:.3f}\n".format(split, node, *hha.utils.bootstrap(y_encode[:,node], pred_encode[:,node], average_precision_score) ))
                            g.write("{}, {},   F1,   {:.3f}, {:.3f} , {:.3f}\n".format(split, node, *hha.utils.bootstrap(y_encode[:,node], pred_encode[:,node], f1_score) ))
                            g.flush()



def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None, binary=True, weighted=False):
    """Run one epoch of training/evaluation for classification.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """
    ## Setting self.training = True, 
    ## beware that some layers have different behavior during train/and evaluation 
    ## (like BatchNorm, Dropout) so setting it matters
    model.train(train)

    total = 0  # total training loss
    n = 0      # number of signals processed

    yhat = []
    y = []
    sample_loss = []
    fnames = []

    with torch.set_grad_enabled(train): # True:training, False:inference
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome, fname) in dataloader:
                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)
                fnames.append(fname)

                # FORWARD PASS
                outputs = model(X)
                if save_all:
                    yhat.append(outputs.to("cpu").detach().numpy())

                if not save_all:
                    yhat.append(outputs.to("cpu").detach().numpy())

                if not weighted:
                    if binary:
                        # Loss
                        criterion = torch.nn.BCEWithLogitsLoss()
                        loss = criterion(outputs.view(-1), outcome)
                        # Track per sample loss
                        criterion_weighted_manual = torch.nn.BCEWithLogitsLoss(reduction='none')
                        loss_weighted_manual = criterion_weighted_manual(outputs.view(-1), outcome)
                        sample_loss.append(np.expand_dims(loss_weighted_manual.to("cpu").detach().numpy(), axis=1))
                    else:
                        ## Loss 
                        criterion = torch.nn.CrossEntropyLoss()
                        loss = criterion(outputs, outcome.long())
                        # Track per sample loss
                        criterion_vec = torch.nn.CrossEntropyLoss(reduction='none')
                        loss_vec = criterion_vec(outputs, outcome.long())
                        sample_loss.append(np.expand_dims(loss_vec.to("cpu").detach().numpy(), axis=1))

                if weighted:
                    if binary:
                        # Loss
                        weights = torch.tensor([2.8]).cuda() 
                        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
                        loss = criterion(outputs.view(-1), outcome)
                        # Track per sample weighted loss
                        criterion_weighted_manual = torch.nn.BCEWithLogitsLoss(pos_weight=weights, reduction='none')
                        loss_weighted_manual = criterion_weighted_manual(outputs.view(-1), outcome)
                        sample_loss.append(np.expand_dims(loss_weighted_manual.to("cpu").detach().numpy(), axis=1))
                    else:
                        # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
                        weights = torch.tensor([1.0, 3.0, 1.0]).cuda() # This is for HTN, HCM and Athletes
                        criterion = torch.nn.CrossEntropyLoss(weight=weights)
                        loss = criterion(outputs, outcome.long())
                        # Track per sample weighted loss
                        criterion_weighted_manual = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
                        loss_weighted_manual = criterion_weighted_manual(outputs, outcome.long())
                        sample_loss.append(np.expand_dims(loss_weighted_manual.to("cpu").detach().numpy(), axis=1))

                ## statistics
                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) ".format(total / n, loss.item()))
                pbar.update()
            
            ## Calculating the FORWARD pass (continued)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
    
    if not save_all:
        yhat = np.concatenate(yhat)

    y = np.concatenate(y)
    fnames = np.concatenate(fnames)
    flat_loss = [item for sublist in sample_loss for item in sublist]
    y_true = np.expand_dims(y, axis=1)
    per_sampleLoss = np.expand_dims(np.array(flat_loss).flatten(), axis=1)

    if binary:
        yhat_nn_function = expit(yhat)
        threshold = 0.5
        predictions = np.zeros(yhat_nn_function.shape, dtype=int)
        predictions[yhat_nn_function < threshold] = 0 
        predictions[yhat_nn_function >= threshold] = 1 
    else:
        yhat_nn_function = softmax(yhat)
        predictions = np.expand_dims(np.argmax(yhat_nn_function, axis=1), axis=1)
    
    epoch_metrics = np.concatenate([y_true, predictions, per_sampleLoss], axis=1)
    
    return total / n, yhat, y, epoch_metrics, fnames


def log_epoch_metrics(metrics):    
    
    metrics_label_ndx = 0 
    metrics_yhat_ndx = 1
    metrics_loss_ndx = 2 

    metrics_dict = classification_report(metrics[:, metrics_label_ndx ], metrics[:, metrics_yhat_ndx ], output_dict=True)
    metrics_dict['loss/all'] = metrics[:, metrics_loss_ndx].mean()
    
    for __,v in enumerate([*metrics_dict][:-4]):
        class_index = metrics[:,metrics_label_ndx] == np.float(v)
        metrics_dict[v]['loss'] = metrics[class_index, metrics_loss_ndx].mean()
        metrics_dict[v]['correct'] = np.sum(metrics[class_index, 0 ] == metrics[class_index, 1]) / np.float32(np.sum(class_index)) * 100

    print(confusion_matrix(metrics[:, 0 ], metrics[:, 1 ]))
    print(classification_report(metrics[:, 0 ], metrics[:, 1 ]))
    for key in metrics_dict:
        print(key, '->', metrics_dict[key])
    return metrics_dict