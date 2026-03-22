# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..modules import diagnostics
from ..modules import helper
from ..modules import utils

def fit(config, model, train_dl, model_children, regular_param, optimizer, latent_dim, RHO, l1, n_dimensions):
    """This function trains the model on the train set. It computes the losses and does the backwards propagation, and updates the optimizer as well.
    Args:
        model (modelObject): The model you wish to train
        train_dl (torch.DataLoader): Defines the batched data which the model is trained on
        model_children (list): List of model parameters
        regular_param (float): Determines proportionality constant for the gradient descent step.
        optimizer (torch.optim): Chooses optimizer for gradient descent.
        RHO (float): Float used for KL Divergence (Not currently a feature)
        l1 (boolean): If `True`, use L1 regularization. Otherwise, don't.
        n_dimensions (int): Number of dimensions.
    Returns:
        list, model object: Training loss and trained model
    """

    print("### Beginning Training")
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_l1 = 0.0
    device = helper.get_device()

    for idx, inputs in enumerate(tqdm(train_dl)):
        inputs = inputs.to(device).float()
        optimizer.zero_grad()
        reconstructions = model(inputs)

        if hasattr(config, "custom_loss_function") and config.custom_loss_function == "loss_function_swae":
            z = model.encode(inputs)
            loss, mse_loss, l1_loss = utils.loss_function_swae(inputs, z, reconstructions, latent_dim)
        else:
            loss, mse_loss, l1_loss = utils.mse_sum_loss_l1(
                model_children=model_children,
                true_data=inputs,
                reconstructed_data=reconstructions,
                reg_param=regular_param,
                validate=True,
            )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() if hasattr(loss, 'item') else loss
        running_mse += mse_loss.item() if hasattr(mse_loss, 'item') else mse_loss
        running_l1 += l1_loss.item() if hasattr(l1_loss, 'item') else l1_loss

    epoch_loss = running_loss / (idx + 1)
    print(f"Training Loss: {epoch_loss:.6f}")
    return epoch_loss, running_mse/(idx+1), running_l1/(idx+1), model

def validate(model, test_dl, model_children, reg_param):
     """Function used to validate the training. Not necessary for doing compression, but gives a good indication of wether the model selected is a good fit or not.
    Args:
        model (modelObject): Defines the model one wants to validate. The model used here is passed directly from `fit()`.
        test_dl (torch.DataLoader): Defines the batched data which the model is validated on
        model_children (list): List of model parameters
        regular_param (float): Determines proportionality constant for the gradient descent step.
    Returns:
        float: Validation loss
    """
     print("### Beginning Validating")
    model.eval()
    running_loss = 0.0
    device = helper.get_device()

    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(test_dl)):
            inputs = inputs.to(device).float()
            reconstructions = model(inputs)
            loss, _, _ = utils.mse_sum_loss_l1(
                model_children=model_children,
                true_data=inputs,
                reconstructed_data=reconstructions,
                reg_param=reg_param,
                validate=True,
            )
            running_loss += loss.item() if hasattr(loss, 'item') else loss

    epoch_loss = running_loss / (idx + 1)
    print(f"Validation Loss: {epoch_loss:.6f}")
    return epoch_loss

def seed_worker(worker_id):

    """PyTorch implementation to fix the seeds
    Args:
        worker_id ():
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(model, variables, train_data, test_data, project_path, config, tracker=None):

    """Does the entire training loop by calling the `fit()` and `validate()`. Appart from this, this is the main function where the data is converted
        to the correct type for it to be trained, via `torch.Tensor()`. Furthermore, the batching is also done here, based on `config.batch_size`,
        and it is the `torch.utils.data.DataLoader` doing the splitting.
        Applying either `EarlyStopping` or `LR Scheduler` is also done here, all based on their respective `config` arguments.
        For reproducibility, the seeds can also be fixed in this function.
    Args:
        model (modelObject): The model you wish to train
        variables (_type_): _description_
        train_set (ndarray): Array consisting of the train set
        test_set (ndarray): Array consisting of the test set
        project_path (string): Path to the project directory
        config (dataClass): Base class selecting user inputs
    Returns:
        modelObject: fully trained model ready to perform compression and decompression
    """
    #Added tracker=None to accept the emissions tracker
    device = helper.get_device()
    model = model.to(device).float()

    if config.deterministic_algorithm:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.use_deterministic_algorithms(True)
        g = torch.Generator()
        g.manual_seed(0)
    else:
        g = None

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    valid_tensor = torch.tensor(test_data, dtype=torch.float32)

    if config.data_dimension == 2:
        if config.model_type == "dense":
            train_ds = train_tensor.view(train_data.shape[0], -1)
            valid_ds = valid_tensor.view(test_data.shape[0], -1)
        elif config.model_type == "convolutional":
            train_ds = train_tensor.view(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
            valid_ds = valid_tensor.view(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
    else:
        train_ds = train_tensor.view(train_data.shape[0], -1)
        valid_ds = valid_tensor.view(test_data.shape[0], -1)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=not config.deterministic_algorithm, worker_init_fn=seed_worker if g else None, generator=g)
    valid_dl = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model_children = list(model.children())
    
    if config.early_stopping:
        early_stopping = utils.EarlyStopping(patience=config.early_stopping_patience, min_delta=config.min_delta)
    if config.lr_scheduler:
        lr_scheduler = utils.LRScheduler(optimizer=optimizer, patience=config.lr_scheduler_patience)

    train_loss, val_loss = [], []

    if config.activation_extraction:
        hooks = model.store_hooks()

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")

        t_loss, _, _, model = fit(config, model, train_dl, model_children, config.reg_param, optimizer, config.latent_space_size, config.RHO, config.l1, config.data_dimension)
        train_loss.append(t_loss)

        if config.test_size:
            v_loss = validate(model, valid_dl, model_children, config.reg_param)
        else:
            v_loss = t_loss
        val_loss.append(v_loss)

        if config.lr_scheduler: lr_scheduler(v_loss)
        
        # CARBON METRICS
        #  flush the tracker to ensure a row is written for every epoch
        if tracker is not None:
            tracker.flush()

        if config.early_stopping:
            early_stopping(v_loss)
            if early_stopping.early_stop: break

        if config.intermittent_model_saving and epoch % config.intermittent_saving_patience == 0:
            helper.model_saver(model, os.path.join(project_path, f"model_{epoch}.pt"))

    if config.activation_extraction:
        activations = diagnostics.dict_to_square_matrix(model.get_activations())
        model.detach_hooks(hooks)
        np.save(os.path.join(project_path, "activations.npy"), activations)

    np.save(os.path.join(project_path, "loss_data.npy"), np.array([train_loss, val_loss]))

    print(f"Training Complete.")
    return model
