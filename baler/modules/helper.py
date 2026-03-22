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

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from math import ceil
import gzip
import csv

from tqdm import tqdm

sys.path.append(os.getcwd())
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..modules import training, plotting, data_processing, diagnostics


def get_arguments():
    """Determines the arguments one is able to apply in the command line when running Baler."""
    parser = argparse.ArgumentParser(
        prog="baler",
        description=(
            "Baler is a machine learning based compression tool for big data.\n\n"
            "Baler has three running modes:\n\n"
            '\t1. Derivation: Using a configuration file and a "small" input dataset, Baler derives a '
            "machine learning model optimized to compress and decompress your data.\n\n"
            "\t2. Compression: Using a previously derived model and a large input dataset, Baler compresses "
            "your data and outputs a smaller compressed file.\n\n"
            "\t3. Decompression: Using a previously compressed file as input and a model, Baler decompresses "
            "your data into a larger file."
        ),
        epilog="Enjoy!",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="newProject, train, compress, decompress, plot, info",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        nargs=2,
        metavar=("WORKSPACE", "PROJECT"),
        help="Specifies workspace and project.\n"
        "e.g. --project CFD firstTry"
    )
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Verbose mode"
    )
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    workspace_name = args.project[0]
    project_name = args.project[1]
    config_path = (
        f"workspaces.{workspace_name}.{project_name}.config.{project_name}_config"
    )

    if args.mode == "newProject":
        config = None
    else:
        # Create an instance of the class so it can hold data
        config = Config() 
        module = importlib.import_module(config_path)
        module.set_config(config)
        importlib.import_module(config_path).set_config(config)

    return (
        config,
        args.mode,
        workspace_name,
        project_name,
        args.verbose,
    )


def create_new_project(
    workspace_name: str,
    project_name: str,
    verbose: bool = False,
    base_path: str = "workspaces",
) -> None:
    workspace_path = os.path.join(base_path, workspace_name)
    project_path = os.path.join(base_path, workspace_name, project_name)
    if os.path.exists(project_path):
        print(f"The workspace and project ({project_path}) already exists.")
        return
    os.makedirs(project_path)

    required_directories = [
        os.path.join(workspace_path, "data"),
        os.path.join(project_path, "config"),
        os.path.join(project_path, "output", "compressed_output"),
        os.path.join(project_path, "output", "decompressed_output"),
        os.path.join(project_path, "output", "plotting"),
        os.path.join(project_path, "output", "training"),
    ]

    if verbose:
        print(f"Creating project {project_name} in workspace {workspace_name}...")
    for directory in required_directories:
        if verbose:
            print(f"Creating directory {directory}...")
        os.makedirs(directory, exist_ok=True)

    with open(
        os.path.join(project_path, "config", f"{project_name}_config.py"), "w"
    ) as f:
        f.write(create_default_config(workspace_name, project_name))


@dataclass
class Config:
    input_path: str = ""
    compression_ratio: float = 2.0
    epochs: int = 10
    early_stopping: bool = True
    early_stopping_patience: int = 100 # Fixed typo from 'stoppin'
    lr_scheduler: bool = True
    lr_scheduler_patience: int = 50
    min_delta: float = 0.0
    model_name: str = "AE"
    model_type: str = "dense"
    custom_norm: bool = False
    l1: bool = True
    reg_param: float = 0.001
    RHO: float = 0.05
    lr: float = 0.001
    batch_size: int = 512
    test_size: float = 0.2
    data_dimension: int = 1
    intermittent_model_saving: bool = False
    separate_model_saving: bool = False
    intermittent_saving_patience: int = 100
    mse_avg: bool = False
    mse_sum: bool = True
    emd: bool = False
    deterministic_algorithm: bool = True
    # Extra fields for your ATLAS setup
    save_error_bounded_deltas: bool = False
    number_of_columns: int = 0
    latent_space_size: int = 0


def create_default_config(workspace_name: str, project_name: str) -> str:
    return f"""
# === Configuration options ===

def set_config(c):
    c.input_path                    = "workspaces/{workspace_name}/data/{project_name}_data.npz"
    c.data_dimension                = 1
    c.compression_ratio             = 2.0
    c.apply_normalization           = True
    c.model_name                    = "AE"
    c.model_type                     = "dense"
    c.epochs                        = 5
    c.lr                            = 0.001
    c.batch_size                    = 512
    c.early_stopping                = True
    c.lr_scheduler                  = True

# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                     = 0
    c.lr_scheduler_patience        = 50
    c.custom_norm                  = False
    c.reg_param                    = 0.001
    c.RHO                          = 0.05
    c.test_size                    = 0
    c.extra_compression            = False
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 100
    c.mse_avg                      = False
    c.mse_sum                      = True
    c.emd                          = False
    c.l1                           = True
    c.activation_extraction        = False
    c.deterministic_algorithm      = True
    c.separate_model_saving        = False
"""


def model_init(model_name: str):
    model_object = data_processing.initialise_model(model_name)
    return model_object


def numpy_to_tensor(data):
    return torch.from_numpy(data)


def normalize(data, custom_norm):
    data = data_processing.normalize(data, custom_norm=custom_norm)
    return data

def process(
    input_path,
    custom_norm,
    test_size,
    apply_normalization,
    convert_to_blocks,
    verbose,
):
    loaded = np.load(input_path)
    data = loaded["data"]

    if verbose:
        print("Original Dataset Shape - ", data.shape)

    original_shape = data.shape

    if convert_to_blocks:
        data = data_processing.convert_to_blocks_util(convert_to_blocks, data)

    normalization_features = data_processing.find_minmax(data)
    if apply_normalization:
        print("Normalizing the data...")
        data = normalize(data, custom_norm)
    if not test_size:
        train_set = data
        test_set = train_set
    else:
        train_set, test_set = train_test_split(
            data, test_size=test_size, random_state=1
        )

    return (train_set, test_set, normalization_features, original_shape)


def renormalize(data, true_min_list, feature_range_list):
    return data_processing.renormalize_func(data, true_min_list, feature_range_list)


def train(model, number_of_columns, train_set, test_set, project_path, config, tracker=None):
    # tracker=None added to pass the EmissionTracker into the core loop
    return training.train(
        model, number_of_columns, train_set, test_set, project_path, config, tracker=tracker
    )


def plotter(output_path, config):
    plotting.plot(output_path, config)
    print("=== Done ===")
    print("Your plots are available in:", os.path.join(output_path, "plotting"))


def loss_plotter(path_to_loss_data, output_path, config):
    return plotting.loss_plot(path_to_loss_data, output_path, config)


def model_saver(model, model_path):
    return data_processing.save_model(model, model_path)


def encoder_decoder_saver(model, encoder_path, decoder_path):
    return data_processing.encoder_saver(
        model, encoder_path
    ), data_processing.decoder_saver(model, decoder_path)


def detacher(tensor):
    return tensor.cpu().detach().numpy()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def save_error_bounded_requirement(config, decoded_output, data_batch):
    # Safe division to prevent RuntimeWarnings
    epsilon = 1e-12
    
    rms_pred_error = (
        np.divide(
            np.subtract(decoded_output, data_batch),
            data_batch + epsilon,
        )
        * 100
    )

    # Clean up non-finite results (inf, -inf, nan)
    rms_pred_error[~np.isfinite(rms_pred_error)] = 0.0

    rms_pred_error_index = np.where(
        abs(rms_pred_error) > config.error_bounded_requirement
    )
    rows_idx, col_idx = rms_pred_error_index
    deltas = []
    if len(rows_idx) > 0 and len(col_idx) > 0:
        rms_pred_error_exceeding_error_bound = np.subtract(
            decoded_output,
            data_batch,
            dtype=np.float16,
        )
        for i in range(len(rows_idx)):
            deltas.append(rms_pred_error_exceeding_error_bound[rows_idx[i]][col_idx[i]])
    return deltas, rms_pred_error_index


def compress(model_path, config):
    loaded = np.load(config.input_path)
    data_before = loaded["data"]
    original_shape = data_before.shape

    if hasattr(config, "convert_to_blocks") and config.convert_to_blocks:
        data_before = data_processing.convert_to_blocks_util(
            config.convert_to_blocks, data_before
        )

    if config.apply_normalization:
        print("Normalizing...")
        data = normalize(data_before, config.custom_norm)
    else:
        data = data_before
    
    number_of_columns = 0
    try:
        n_features = 0
        if config.data_dimension == 1:
            column_names = np.load(config.input_path)["names"]
            number_of_columns = len(column_names)
            config.latent_space_size = ceil(number_of_columns / config.compression_ratio)
            config.number_of_columns = number_of_columns
            n_features = number_of_columns
        elif config.data_dimension == 2:
            if config.model_type == "dense":
                number_of_rows = data.shape[1]
                config.number_of_columns = data.shape[2]
                n_features = number_of_rows * config.number_of_columns
            else:
                number_of_rows = original_shape[1]
                config.number_of_columns = original_shape[2]
                n_features = config.number_of_columns
            config.latent_space_size = ceil(
                (number_of_rows * config.number_of_columns) / config.compression_ratio
            )
        else:
            raise NameError(f"Data dimension can only be 1 or 2. Got {config.data_dimension}")
    except AttributeError:
        number_of_columns = config.number_of_columns
        latent_space_size = config.latent_space_size

    bs = config.batch_size
    device = get_device()
    model_object = data_processing.initialise_model(config.model_name)
    model = data_processing.load_model(
        model_object,
        model_path=model_path,
        n_features=n_features,
        z_dim=config.latent_space_size,
    )
    model.eval()

    if config.data_dimension == 2:
        if config.model_type == "convolutional" and config.model_name == "Conv_AE_3D":
            data_tensor = torch.tensor(data, dtype=torch.float32).view(
                data.shape[0] // bs, 1, bs, data.shape[1], data.shape[2]
            )
        elif config.model_type == "convolutional":
            data_tensor = torch.tensor(data, dtype=torch.float32).view(
                data.shape[0], 1, data.shape[1], data.shape[2]
            )
        elif config.model_type == "dense":
            data_tensor = torch.tensor(data, dtype=torch.float32).view(
                data.shape[0], data.shape[1] * data.shape[2]
            )
    elif config.data_dimension == 1:
        data_tensor = torch.tensor(data, dtype=torch.float64)

    data_dl = DataLoader(data_tensor, batch_size=bs, shuffle=False)

    error_bound_batch, error_bound_deltas, error_bound_index, compressed = [], [], [], []

    with torch.no_grad():
        for idx, data_batch in enumerate(tqdm(data_dl)):
            data_batch = data_batch.to(device)
            compressed_output = model.encode(data_batch)

            if config.save_error_bounded_deltas:
                decoded_output = model.decode(compressed_output)
                decoded_output = detacher(decoded_output)
            
            compressed_output = detacher(compressed_output)
            data_batch = detacher(data_batch)

            if config.save_error_bounded_deltas:
                deltas, rms_idx = save_error_bounded_requirement(config, decoded_output, data_batch)
                if len(rms_idx[0]) > 0:
                    error_bound_batch.append(idx)
                    error_bound_deltas.append(deltas)
                    error_bound_index.append(rms_idx)

            if idx == 0:
                compressed = compressed_output
            else:
                compressed = np.concatenate((compressed, compressed_output))

    return (compressed, error_bound_batch, error_bound_deltas, error_bound_index)


def decompress(model_path, input_path, input_path_deltas, input_batch_index, model_name, config, output_path, original_shape):
    loaded = np.load(input_path)
    data = loaded["data"]
    names = loaded["names"]
    normalization_features = loaded["normalization_features"]

    if config.model_type == "convolutional":
        final_layer_details = np.load(os.path.join(output_path, "training", "final_layer.npy"), allow_pickle=True)

    if config.save_error_bounded_deltas:
        error_bound_batch = np.load(gzip.GzipFile(input_batch_index, "r"), allow_pickle=True)[0]
        error_bound_deltas = np.load(gzip.GzipFile(input_path_deltas, "r"), allow_pickle=True)
        error_bound_index = np.load(gzip.GzipFile(input_batch_index, "r"), allow_pickle=True)[1]

    bs = config.batch_size
    device = get_device()
    model_dict = torch.load(str(model_path), map_location=device)
    number_of_columns = len(model_dict[list(model_dict.keys())[-1]])

    model_object = data_processing.initialise_model(config.model_name)
    model = data_processing.load_model(model_object, model_path=model_path, n_features=number_of_columns, z_dim=len(data[0]))
    model.eval()

    if config.model_type == "convolutional":
        model.set_final_layer_dims(final_layer_details)

    data_tensor = torch.from_numpy(data).to(device)
    data_dl = DataLoader(data_tensor, batch_size=bs, shuffle=False)

    decompressed = []
    with torch.no_grad():
        for idx, data_batch in enumerate(tqdm(data_dl)):
            out = detacher(model.decode(data_batch.to(device)))
            
            if config.save_error_bounded_deltas and idx in error_bound_batch:
                delta_idx = np.where(error_bound_batch == idx)[0][0]
                deltas = error_bound_deltas[delta_idx]
                row_idx, col_idx = error_bound_index[delta_idx]
                for i in range(len(row_idx)):
                    out[row_idx[i]][col_idx[i]] -= deltas[i]

            if idx == 0:
                decompressed = out
            else:
                decompressed = np.concatenate((decompressed, out))

    if config.data_dimension == 2 and config.model_type == "dense":
        decompressed = decompressed.reshape((len(decompressed), original_shape[1], original_shape[2]))

    return decompressed, names, normalization_features


def diagnose(input_path: str, output_path: str) -> None:
    diagnostics.diagnose(input_path, output_path)


def perform_hls4ml_conversion(output_path, config):
    import hls4ml
    model_path = os.path.join(output_path, "compressed_output", "model.pt")
    model_object = data_processing.initialise_model(config.model_name)
    model = data_processing.load_model(model_object, model_path=model_path, n_features=config.number_of_columns, z_dim=config.latent_space_size)
    model.to("cpu")

    hls_config = hls4ml.utils.config_from_pytorch_model(model, granularity="name", default_reuse_factor=config.default_reuse_factor, default_precision=config.default_precision)
    hls_config["Model"]["Strategy"] = config.Strategy

    cfg = hls4ml.converters.create_config(backend="Vivado")
    cfg["Part"], cfg["ClockPeriod"], cfg["IOType"], cfg["HLSConfig"], cfg["PytorchModel"], cfg["InputShape"], cfg["OutputDir"] = config.Part, config.ClockPeriod, config.IOType, hls_config, model, config.InputShape, config.OutputDir

    hls_model = hls4ml.converters.pytorch_to_hls(cfg)
    hls_model.compile()
    hls_model.build(csim=config.csim, synth=config.synth, cosim=config.cosim, export=config.export)
    
    
   #function to create a CSV to collect metrics during each process(Training, compress, and decompress)
def add_to_performance_log(path, metrics_dict, file_name="training_metrics.csv"):
    """
    Appends a dictionary of metrics as a new row in a CSV file using the csv module.
    """
    csv_path = os.path.join(path, file_name)
    file_exists = os.path.isfile(csv_path)

    # Extract headers (keys) and values from the dictionary
    headers = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Open in 'a' (append) mode
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # If the file is brand new, write the header row first
        if not file_exists:
            writer.writerow(headers)
        
        # Write the actual data row
        writer.writerow(values)

    print(f"Metrics successfully logged to {csv_path}")
    
 #Get the power limit of the laptop
#It would help for Green AI comparison
def get_cpu_power_limit():
    try:
        with open("/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw", "r") as f:
            uw = int(f.read().strip())
            return uw / 1_000_000  # Convert to Watts
    except FileNotFoundError:
        return 45.0
