"""
This script is an adaption of a vertical FL pipeline defined by Wei et al. 
(see https://arxiv.org/abs/2202.04309), which was consequently modified by @BalticBytes
(see https://github.com/BalticBytes/vertical-federated-learning-kang).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from standalone_model import CentralizedModel, OrgModel, TopModel
from utils import batch_split


def main(args):
    # dataset loading
    X = pd.read_csv("data/adult.csv", index_col=False)

    # get the feature and label data
    y = X["income"].apply(lambda x: bool(">" in x)).astype("int").to_numpy()
    X = X.drop(["income"], axis=1)

    n_samples, n_features = X.shape
    columns = list(X.columns)

    # print some dataset characteristics
    print(
        f"Dataset: adult\n",
        f"Number of samples: {n_samples}\n",
        f"Number of features: {n_features}\n",
        f"Number of classes: 2\n",
        f"Postive ratio: {sum(y)/len(y)*100:.2f}%\n",
        f"Training mode: {args.training_mode}",
    )

    # initialize the arrays to return to the main function
    loss_list = []
    auc_list = []

    # initialize preprocessing
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = StandardScaler()

    if args.training_mode == "vertical":
        # set up the attribute split scheme for vertical FL
        attribute_groups = []
        attribute_idx = 0
        n_features_per_org, remainder = divmod(n_features, args.org_num)
        for i in range(args.org_num - 1):
            attribute_groups.append(
                columns[attribute_idx : attribute_idx + n_features_per_org]
            )
            attribute_idx += n_features_per_org
        attribute_groups.append(
            columns[attribute_idx : attribute_idx + n_features_per_org + remainder]
        )
        print(f"Attribute distribution: {attribute_groups}")

        # set up the random seed for dataset splitting
        random_seed = 1001

        data_splits = {}
        encoded_data_splits = {}
        X_train = {}
        X_test = {}
        train_loaders = []
        test_loaders = []

        for i in range(args.org_num):
            # split the data vertically with one-hot encoding for multiple organizations
            data_splits[i] = X[attribute_groups[i]]
            if data_splits[i].select_dtypes(exclude="number").shape[1] > 0:
                encoded = encoder.fit_transform(
                    data_splits[i].select_dtypes(exclude="number")
                )
            else:
                encoded = np.empty(shape=(n_samples, 0))
            if data_splits[i].select_dtypes(include="number").shape[1] > 0:
                scaled = scaler.fit_transform(
                    data_splits[i].select_dtypes(include="number")
                )
            else:
                scaled = np.empty(shape=(n_samples, 0))
            encoded_data_splits[i] = np.concatenate([encoded, scaled], axis=1)
            print(
                f"Shape of the encoded data held by Organization {i}: {np.shape(encoded_data_splits[i])}"
            )

            # split the encoded data samples into training and test datasets
            if i == 0:
                # labels are held by the server so we only need to store them once
                X_train[i], X_test[i], y_train, y_test = train_test_split(
                    encoded_data_splits[i], y, test_size=0.2, random_state=random_seed
                )
            else:
                X_train[i], X_test[i], _, _ = train_test_split(
                    encoded_data_splits[i], y, test_size=0.2, random_state=random_seed
                )

            # create data loaders
            X_train[i] = torch.from_numpy(X_train[i]).float()
            X_test[i] = torch.from_numpy(X_test[i]).float()
            train_loaders.append(DataLoader(X_train[i], batch_size=args.batch_size))
            test_loaders.append(
                DataLoader(X_test[i], batch_size=len(X_test[i]), shuffle=False)
            )

        # append labels to data loader lists on last position
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()
        train_loaders.append(DataLoader(y_train, batch_size=args.batch_size))
        test_loaders.append(DataLoader(y_test, batch_size=args.batch_size))

        # set the neural network layer sizes
        org_hidden_units = [[128]] * args.org_num
        org_output_dim = [64] * args.org_num
        top_hidden_units = [64]
        top_output_dim = 1

        # build the client models
        org_models = {}
        org_optimizers = []
        for i in range(args.org_num):
            org_models[i] = OrgModel(
                X_train[i].shape[-1], org_hidden_units[i], org_output_dim[i]
            )

            org_optimizers.append(
                torch.optim.Adam(org_models[i].parameters(), lr=0.002)
            )

        # build the top model over the client models
        top_model = TopModel(
            sum(org_output_dim), top_hidden_units, top_output_dim
        )
        top_optimizer = torch.optim.Adam(top_model.parameters(), lr=0.002)
        criterion = nn.BCELoss()

        top_model.train()
        for epoch in range(args.epochs):
            batch_list = batch_split(len(y_train), args.batch_size)

            # train
            for batch_idxs in batch_list:
                # reset optimizers
                top_optimizer.zero_grad()
                for i in range(args.org_num):
                    org_optimizers[i].zero_grad()

                # calculate organization outputs
                org_outputs = {}
                for i in range(args.org_num):
                    org_outputs[i] = org_models[i](X_train[i][batch_idxs])
                org_outputs = torch.cat(list(org_outputs.values()), dim=1)

                # calculate top model output
                top_outputs = top_model(org_outputs)
                log_probs = torch.sigmoid(top_outputs).squeeze()

                # calculate loss and backprop
                loss = criterion(log_probs, y_train[batch_idxs])
                loss.backward()
                top_optimizer.step()
                for i in range(args.org_num):
                    org_optimizers[i].step()

            # evaluate
            org_outputs = {}
            for i in range(args.org_num):
                org_outputs[i] = org_models[i](X_test[i])
            org_outputs = torch.cat(list(org_outputs.values()), dim=1)

            top_outputs = top_model(org_outputs)
            log_probs = torch.sigmoid(top_outputs).squeeze()
            auc = roc_auc_score(y_test, log_probs.data)

            print(
                f"[Epoch {epoch: >2}] Train Loss = {loss.detach().item():.4f}, Test AUC = {auc:.4f}"
            )
            loss_list.append(loss.detach().item())
            auc_list.append(auc)

    elif args.training_mode == "centralized":
        encoded = encoder.fit_transform(X.select_dtypes(exclude="number"))
        scaled = scaler.fit_transform(X.select_dtypes(include="number"))
        X = np.concatenate([encoded, scaled], axis=1)

        print(f"Client data shape: {X.shape}, postive ratio: {sum(y)/len(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()

        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        hidden_units = np.array([128, 64, 32])
        model = CentralizedModel(
            input_dim=X_train.shape[-1], hidden_units=hidden_units, num_classes=1
        )
        top_optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(args.epochs):
            # train
            for data, targets in train_loader:
                top_optimizer.zero_grad()
                top_outputs = model(data)
                logits = torch.sigmoid(top_outputs).squeeze()
                loss = criterion(logits, targets)
                loss.backward()
                top_optimizer.step()

            # evaluate
            for data, targets in test_loader:
                top_outputs = model(data)
                log_probs = torch.sigmoid(top_outputs).squeeze()
                auc = roc_auc_score(targets.data, log_probs.data)

            print(
                f"[Epoch {epoch: >2}] Train Loss = {loss.detach().item():.4f}, Test AUC = {auc:.4f}"
            )
            loss_list.append(loss.detach().item())
            auc_list.append(auc)

    return loss_list, auc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vertical FL quickstart with PyTorch")
    parser.add_argument(
        "--batch-size", type=int, default=500, help="batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="number of training epochs"
    )
    parser.add_argument(
        "--training-mode",
        default="vertical",
        choices=["centralized", "vertical"],
        help="define the learning setup (vertical or centralized)",
    )
    parser.add_argument(
        "--org-num",
        type=int,
        default=2,
        help="number of organizations (ignored for centralized training)",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    loss_list, auc_list = main(args)
    elapsed = time.perf_counter() - start_time

    sec, millis = divmod(elapsed, 1)
    mins, sec = divmod(sec, 60)
    hrs, mins = divmod(mins, 60)
    print(f"Elapsed time: {hrs:.0f}:{mins:02.0f}:{sec:02.0f}:{millis*1000:03.0f}")
    print(loss_list)
    print(auc_list)
