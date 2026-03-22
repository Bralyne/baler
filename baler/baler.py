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
import time
from thop import profile
from math import ceil
import numpy as np
import torch
import csv
import gzip
from .modules import helper, clean
from .modules.profiling import pytorch_profile
from thop import profile

__all__ = (
    "perform_compression",
    "perform_decompression",
    "perform_diagnostics",
    "perform_plotting",
    "perform_training",
    "print_info",
)


# -------------------------
# Main Entry
# -------------------------
def main():
    config, mode, workspace_name, project_name, verbose = helper.get_arguments()
    project_path = os.path.join("workspaces", workspace_name, project_name)
    output_path = os.path.join(project_path, "output")

    if mode == "newProject":
        helper.create_new_project(workspace_name, project_name, verbose)
    elif mode == "train":
        perform_training(output_path=output_path, config=config, verbose=verbose)
    elif mode == "diagnose":
        perform_diagnostics(output_path, verbose)
    elif mode == "compress":
        perform_compression(output_path, config, verbose)
    elif mode == "decompress":
        perform_decompression(output_path, config, verbose)
    elif mode == "plot":
        perform_plotting(output_path, config, verbose)
    elif mode == "info":
        print_info(output_path, config)
    elif mode == "clean":
        clean.clean(config)
    elif mode == "convert_with_hls4ml":
        helper.perform_hls4ml_conversion(output_path, config)
    else:
        raise NameError(f"Baler mode {mode} not recognised. Use baler --help to see available modes.")


# -------------------------
# Training
# -------------------------
def perform_training(output_path, config, verbose: bool):
    """Main function calling the training functions, ran when --mode=train is selected.
        The three functions called are: `helper.process`, `helper.mode_init` and `helper.training`.

        Depending on `config.data_dimensions`, the calculated latent space size will differ.

    Args:
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information

    Raises:
        NameError: Baler currently only supports 1D (e.g. HEP) or 2D (e.g. CFD) data as inputs.
    """
    import warnings
    import logging
    import os
    import time
    import torch
    import numpy as np
    from math import ceil
    from thop import profile

    # Silence warnings and CodeCarbon INFO logs
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger("codecarbon").setLevel(logging.ERROR)
    
    from codecarbon import EmissionsTracker
    training_path = os.path.join(output_path, "training")
    
    # Initialize tracker with high resolution (1s)
    tracker = EmissionsTracker(
        measure_power_secs=1,
        api_call_interval=1,
        project_name="Baler_Training",
        output_dir=training_path,
        output_file="emission_training.csv",
        save_to_file=True,
        tracking_mode='machine', 
    )

    #  START TRACKING ---
    tracker.start()

    (
        train_set_norm,
        test_set_norm,
        normalization_features,
        original_shape,
    ) = helper.process(
        config.input_path,
        config.custom_norm,
        config.test_size,
        config.apply_normalization,
        config.convert_to_blocks if hasattr(config, "convert_to_blocks") else None,
        verbose,
    )

    if verbose:
        print("Training and testing sets normalized")

    try:
        n_features = 0
        if config.data_dimension == 1:
            number_of_columns = train_set_norm.shape[1]
            config.latent_space_size = ceil(
                number_of_columns / config.compression_ratio
            )
            config.number_of_columns = number_of_columns
            n_features = number_of_columns
        elif config.data_dimension == 2:
            if config.model_type == "dense":
                number_of_rows = train_set_norm.shape[1]
                number_of_columns = train_set_norm.shape[2]
                n_features = number_of_columns * number_of_rows
            else:
                number_of_rows = original_shape[1]
                number_of_columns = original_shape[2]
                n_features = number_of_columns
            config.latent_space_size = ceil(
                (number_of_rows * number_of_columns) / config.compression_ratio
            )
            config.number_of_columns = number_of_columns
        else:
            raise NameError(
                "Data dimension can only be 1 or 2. Got config.data_dimension value = "
                + str(config.data_dimension)
            )
    except AttributeError:
        if verbose:
            print(
                f"{config.number_of_columns} -> {config.latent_space_size} dimensions"
            )
        assert number_of_columns == config.number_of_columns

    if verbose:
        print(
            f"Intitalizing Model with Latent Size - {config.latent_space_size} and Features - {n_features}"
        )

    device = helper.get_device()
    if verbose:
        print(f"Device used for training: {device}")

    model_object = helper.model_init(config.model_name)
    model = model_object(n_features=n_features, z_dim=config.latent_space_size)
    model.to(device)

    if config.model_name == "Conv_AE_3D" and hasattr(
        config, "compress_to_latent_space"
    ):
        model.set_compress_to_latent_space(config.compress_to_latent_space)

    if verbose:
        print(f"Model architecture:\n{model}")
    
    model.eval()
    dummy_input = torch.randn(1, n_features).to(device).float()
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2 
    
    # PRINT
    print("\n" + "-"*60)
    print(f"{'Layer Name':<30} | {'MACs':<12} | {'Parameters':<12}")
    print("-"*60)
    for name, module in model.named_modules():
        if hasattr(module, 'total_ops') and module.total_ops > 0:
            m_macs = int(module.total_ops)
            m_params = sum(p.numel() for p in module.parameters())
            print(f"{name:<30} | {m_macs:<12,} | {m_params:<12,}")
    print("-"*60 + "\n")

    model.train()

    # --- START PROFILING ---
    start_wall = time.time()
    start_perf = time.perf_counter()
    start_cpu = time.process_time()

    trained_model = helper.train(
        model, number_of_columns, train_set_norm, test_set_norm, training_path, config, tracker=tracker
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    #  STOP TRACKING & PROFILING 
    end_wall = time.time()
    end_perf = time.perf_counter()
    end_cpu = time.process_time()
    
    tracker.stop()

    wall_diff = end_wall - start_wall
    perf_diff = end_perf - start_perf
    cpu_diff = end_cpu - start_cpu

    if torch.cuda.is_available():
        avg_power_w = 45.0 
    else:
        avg_power_w = helper.get_cpu_power_limit()
        
    energy_joules = avg_power_w * wall_diff
    total_gflops = (flops * len(train_set_norm) * config.epochs) / 1e9

    try:
        loss_history = np.load(os.path.join(training_path, "loss_data.npy"))
        val_loss_row = loss_history[1]
        best_index = np.argmin(val_loss_row)
        best_val_loss = val_loss_row[best_index]
        best_epoch = int(best_index + 1)
    except Exception as e:
        best_epoch, best_val_loss = "N/A", "N/A"

    #  PRINTING ---
    print("\n" + "="*50)
    print("TRAINING PERFORMANCE SUMMARY")
    print(f"1. Wall-Clock (time.time):      {wall_diff:.4f}s")
    print(f"2. Perf Counter (True-Walltime):         {perf_diff:.4f}s")
    print(f"3. Process Time (CPU):   {cpu_diff:.4f}s")
    print("-" * 50)
    print(f"Model MACs:               {int(macs)}")
    print(f"Model FLOPs:              {int(flops)}")
    print(f"Estimated Energy:         {energy_joules:.2f} Joules (@{avg_power_w}W)")
    print(f"Total Training GFLOPs:   {total_gflops:.4f}")
    print("-" * 50)
    print(f"Best Training Epoch:      {best_epoch}")
    print(f"Lowest Validation Loss:  {best_val_loss:.6e}")
    print("="*50 + "\n")

    metrics = {
        "seed": getattr(config, "seed", "N/A"),
        "mode": "train",
        "wall_time_sec": round(wall_diff, 4),
        "perf_time_sec": round(perf_diff, 4),
        "cpu_time_sec": round(cpu_diff, 4),
        "MACs": int(macs),
        "FLOPs": int(flops),
        "Power_Watts": avg_power_w,
        "Energy_Joules": round(energy_joules, 4),
        "Total_GFLOPs": round(total_gflops, 4),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }
    helper.add_to_performance_log(os.path.join(output_path, "training"), metrics, file_name="training_metrics.csv")

    if config.apply_normalization:
        np.save(os.path.join(training_path, "normalization_features.npy"), normalization_features)

    if config.separate_model_saving:
        helper.encoder_decoder_saver(
            trained_model,
            os.path.join(output_path, "compressed_output", "encoder.pt"),
            os.path.join(output_path, "compressed_output", "decoder.pt"),
        )
    else:
        helper.model_saver(trained_model, os.path.join(output_path, "compressed_output", "model.pt"))

# -------------------------
# Compress
# -------------------------

def perform_compression(output_path, config, verbose: bool):
    """Main function calling the compression functions, ran when --mode=compress is selected.
        The main function being called here is: `helper.compress`

         If `config.extra_compression` is selected, the compressed file is further compressed via zip
         Else, the function returns a compressed file of `.npz`, only compressed by Baler.

    Args:
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information

    Outputs:
        An `.npz` file which includes:
        - The compressed data
        - The data headers
        - Normalization features if `config.apply_normalization=True`
    """
    print("Compressing...")
    normalization_features = []

    if config.apply_normalization:
        normalization_features = np.load(
            os.path.join(output_path, "training", "normalization_features.npy")
        )

    # COMPUTE COMPRESSION COMPLEXITY (MACs/FLOPs) ---
    from thop import profile
    device = helper.get_device()
    
    # Identify the model path to load for profiling
    path_to_model = os.path.join(output_path, "compressed_output", "encoder.pt" if config.separate_model_saving else "model.pt")
    
    # load the model architecture to profile it
    model_object = helper.model_init(config.model_name)
    #use the config number of columns for the input size
    model = model_object(n_features=config.number_of_columns, z_dim=config.latent_space_size)
    model.to(device)
    model.eval()

    # Create dummy input based on feature size
    dummy_input = torch.randn(1, config.number_of_columns).to(device).float()
    
    # Profile the Encoder
    # During compression, only the 'encoder' part is  doing the work.
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2 
    
    # START PROFILING ---
    start_wall = time.time()          # System time
    start_perf = time.perf_counter()  # High-res true wall time
    start_cpu  = time.process_time()  # CPU work time

    # CARBON TRACKER FOR COMPRESSION ---
    from codecarbon import OfflineEmissionsTracker
    # measure_power_secs=1 ensures it logs every second for the time-series data
    tracker = OfflineEmissionsTracker(
        project_name="compression",
        output_dir=os.path.join(output_path, "compressed_output"),
        output_file="emission_compression.csv",
        country_iso_code="ZAF",# South Africa
        measure_power_secs=1,
        log_level="error",
    )
    tracker.start()

    if config.separate_model_saving:
        (
            compressed,
            error_bound_batch,
            error_bound_deltas,
            error_bound_index,
        ) = helper.compress(
            model_path=os.path.join(output_path, "compressed_output", "encoder.pt"),
            config=config,
        )
    else:
        (
            compressed,
            error_bound_batch,
            error_bound_deltas,
            error_bound_index,
        ) = helper.compress(
            model_path=os.path.join(output_path, "compressed_output", "model.pt"),
            config=config,
        )

    # Ensure GPU tasks are finished before stopping the clocks
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # STOP CARBON TRACKER ---
    tracker.stop()

    # STOP TRIPLE PROFILING ---
    end_wall = time.time()
    end_perf = time.perf_counter()
    end_cpu  = time.process_time()

    # CALCULATIONS ---
    wall_diff = end_wall - start_wall
    perf_diff = end_perf - start_perf
    cpu_diff  = end_cpu - start_cpu

    # Power and Energy 
    if torch.cuda.is_available():
        avg_power_w = 45.0
    else:
        avg_power_w = helper.get_cpu_power_limit()
        
    energy_joules = avg_power_w * wall_diff
    
    # Total work for compression is (FLOPs per sample * Total samples in the file)
    # This represents the computational footprint of compressing this specific file
    total_compression_gflops = (flops * len(compressed)) / 1e9

    # Printing results as requested
    print("\n" + "="*40)
    print("COMPRESSION PERFORMANCE SUMMARY")
    print(f"1. Wall-Clock (time.time):      {wall_diff:.4f}s")
    print(f"2. Perf Counter (True-Walltime):    {perf_diff:.4f}s")
    print(f"3. Process Time (CPU work):     {cpu_diff:.4f}s")
    print("-" * 40)
    print(f"Inference MACs:                 {int(macs)}")
    print(f"Inference FLOPs:                {int(flops)}")
    print(f"Compression Energy:             {energy_joules:.4f} Joules")
    print(f"Total Compression GFLOPs:       {total_compression_gflops:.6f}")
    print("="*40 + "\n")

    # Metrics for CSV logging - Saved in compressed_output folder
    metrics = {
        "seed": getattr(config, "seed", "N/A"),
        "mode": "compress",
        "wall_time_sec": round(wall_diff, 4),
        "perf_time_sec": round(perf_diff, 4),
        "cpu_time_sec": round(cpu_diff, 4),
        "MACs": int(macs),
        "FLOPs": int(flops),
        "Power_Watts": avg_power_w,
        "Energy_Joules": round(energy_joules, 4),
        "Total_GFLOPs": round(total_compression_gflops, 6)
    }
    
    # Save specifically to compress_metrics.csv in the compressed_output folder
    helper.add_to_performance_log(
        os.path.join(output_path, "compressed_output"), 
        metrics, 
        file_name="compress_metrics.csv"
    )

    # SAVING
    names = np.load(config.input_path)["names"]

    if config.extra_compression:
        if verbose:
            print("Extra compression selected")
            print(
                f"Saving compressed file to {os.path.join(output_path, 'compressed_output', 'compressed.npz')}"
            )
        np.savez_compressed(
            os.path.join(output_path, "compressed_output", "compressed.npz"),
            data=compressed,
            names=names,
            normalization_features=normalization_features,
        )
    else:
        if verbose:
            print("Extra compression not selected")
            print(
                f"Saving compressed file to {os.path.join(output_path, 'compressed_output', 'compressed.npz')}"
            )
        np.savez(
            os.path.join(output_path, "compressed_output", "compressed.npz"),
            data=compressed,
            names=names,
            normalization_features=normalization_features,
        )

    if config.save_error_bounded_deltas:
        # FIX: Added dtype=object to fix VisibleDeprecationWarning for ragged sequences
        error_bound_batch_index = np.array(
            [error_bound_batch, error_bound_index], dtype=object
        )
        f_batch_index = gzip.GzipFile(
            os.path.join(
                output_path,
                "compressed_output",
                "compressed_batch_index_metadata.npz.gz",
            ),
            "w",
        )
        f_deltas = gzip.GzipFile(
            os.path.join(output_path, "compressed_output", "compressed_deltas.npz.gz"),
            "w",
        )
        # Added np.asanyarray with dtype=object to ensure clean saving of deltas
        np.save(file=f_deltas, arr=np.array(error_bound_deltas, dtype=object))
        np.save(
            file=f_batch_index,
            arr=error_bound_batch_index,
        )
        f_batch_index.close()
        f_deltas.close()


# ----------------------------------------------------------------------
# Decompression
# -------------------------
def perform_decompression(output_path, config, verbose: bool):
    """Main function calling the decompression functions, ran when --mode=decompress is selected.
        The main function being called here is: `helper.decompress`

        If `config.apply_normalization=True` the output is un-normalized with the same normalization features saved from `perform_training()`.

    Args:
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """
    
    import warnings
    import logging
    # Silence Deprecation/Future warnings and CodeCarbon INFO logs
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger("codecarbon").setLevel(logging.ERROR)
    
    from codecarbon import EmissionsTracker
    log_dir = os.path.join(output_path, "decompressed_output")
    
    tracker = EmissionsTracker(
        measure_power_secs=1,
        project_name="Baler_Decompression",
        output_dir=log_dir,
        output_file="emission_decompression.csv",
        save_to_file=True,
    )
    tracker.start()

    try:
        print("Decompressing with Triple Profiling...")

       
        model_name = config.model_name
        data_before = np.load(config.input_path)["data"]

        # COMPUTE DECOMPRESSION COMPLEXITY (MACs/FLOPs) ---
        from thop import profile
        device = helper.get_device()
        
        # load the model architecture to profile the decoder part
        model_object = helper.model_init(config.model_name)
        # Decoder takes the latent_space_size as input and outputs original number_of_columns
        model = model_object(n_features=config.number_of_columns, z_dim=config.latent_space_size)
        model.to(device)
        model.eval()

        # Create dummy input based on the latent space size (the input to the decoder)
        dummy_input = torch.randn(1, config.latent_space_size).to(device).float()
        
        
        class DecoderWrapper(torch.nn.Module):
            def __init__(self, decode_func):
                super().__init__()
                self.decode_func = decode_func
            def forward(self, x):
                return self.decode_func(x)

        # Try profiling the decoder
        try:
            
            target = getattr(model, 'decoder', getattr(model, 'dec', model))
            macs, params = profile(target, inputs=(dummy_input,), verbose=False)
            
            
            if macs == 0:
                decoder_module = DecoderWrapper(model.decode)
                macs, params = profile(decoder_module, inputs=(dummy_input,), verbose=False)
        except Exception:
            macs, params = 0, 0
            
        flops = macs * 2 

        # START PROFILING ---
        start_wall = time.time()          # System/Wall clock
        start_perf = time.perf_counter()  # High-resolution true wall time
        start_cpu  = time.process_time()  # Pure CPU execution time

        if config.separate_model_saving:
            decompressed, names, normalization_features = helper.decompress(
                model_path=os.path.join(output_path, "compressed_output", "decoder.pt"),
                input_path=os.path.join(output_path, "compressed_output", "compressed.npz"),
                input_path_deltas=os.path.join(
                    output_path, "compressed_output", "compressed_deltas.npz.gz"
                ),
                input_batch_index=os.path.join(
                    output_path,
                    "compressed_output",
                    "compressed_batch_index_metadata.npz.gz",
                ),
                model_name=model_name,
                config=config,
                output_path=output_path,
                original_shape=data_before.shape,
            )
        else:
            decompressed, names, normalization_features = helper.decompress(
                model_path=os.path.join(output_path, "compressed_output", "model.pt"),
                input_path=os.path.join(output_path, "compressed_output", "compressed.npz"),
                input_path_deltas=os.path.join(
                    output_path, "compressed_output", "compressed_deltas.npz.gz"
                ),
                input_batch_index=os.path.join(
                    output_path,
                    "compressed_output",
                    "compressed_batch_index_metadata.npz.gz",
                ),
                model_name=model_name,
                config=config,
                output_path=output_path,
                original_shape=data_before.shape,
            )
        
        if verbose:
            print(f"Model used: {model_name}")

        if hasattr(config, "convert_to_blocks") and config.convert_to_blocks:
            if config.model_type == "dense":
                decompressed = decompressed.reshape(
                    data_before.shape[0], data_before.shape[1], data_before.shape[2]
                )
            else:
                decompressed = decompressed.reshape(
                    data_before.shape[0], 1, data_before.shape[1], data_before.shape[2]
                )

        if config.apply_normalization:
            normalization_features = np.load(
                os.path.join(output_path, "training", "normalization_features.npy"),
            )
            decompressed = helper.renormalize(
                decompressed,
                normalization_features[0],
                normalization_features[1],
            )

        try:
            type_list = config.type_list
            decompressed = np.transpose(decompressed)
            for index, column in enumerate(decompressed):
                decompressed[index] = decompressed[index].astype(type_list[index])
            decompressed = np.transpose(decompressed)
        except AttributeError:
            pass

        # Ensure all hardware tasks (like GPU kernels) are finished
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        #  STOP  PROFILING ---
        end_wall = time.time()
        end_perf = time.perf_counter()
        end_cpu  = time.process_time()

        # CALCULATIONS ---
        wall_diff = end_wall - start_wall
        perf_diff = end_perf - start_perf
        cpu_diff  = end_cpu - start_cpu
        distortion_mse = np.mean((data_before - decompressed) ** 2)

        # Power and Energy (Using dynamic hardware check)
        if torch.cuda.is_available():
            avg_power_w = 45.0
        else:
            avg_power_w = helper.get_cpu_power_limit()
            
        energy_joules = avg_power_w * wall_diff
        
        # Total work for decompression (FLOPs per sample * Total samples)
        total_decompress_gflops = (flops * len(decompressed)) / 1e9

        # Print results to terminal with requested formatting
        print("\n" + "="*40)
        print("DECOMPRESSION PERFORMANCE SUMMARY")
        print(f"1. Wall-Clock (time.time):      {wall_diff:.4f}s")
        print(f"2. Perf Counter (True-Walltime):    {perf_diff:.4f}s")
        print(f"3. Process Time (CPU work):     {cpu_diff:.4f}s")
        print("-" * 40)
        print(f"Inference MACs:                 {int(macs)}")
        print(f"Inference FLOPs:                {int(flops)}")
        print(f"Decompression Energy:           {energy_joules:.4f} Joules")
        print(f"Total Decompress GFLOPs:        {total_decompress_gflops:.6f}")
        print(f"Distortion (MSE):               {distortion_mse:.6e}")
        print("="*40 + "\n")

        # Logging metrics to the specific decompression_metrics.csv
        metrics = {
            "seed": getattr(config, "seed", "N/A"),
            "mode": "decompress",
            "wall_time_sec": round(wall_diff, 4),
            "perf_time_sec": round(perf_diff, 4),
            "cpu_time_sec": round(cpu_diff, 4),
            "MACs": int(macs),
            "FLOPs": int(flops),
            "Power_Watts": avg_power_w,
            "Energy_Joules": round(energy_joules, 4),
            "Total_GFLOPs": round(total_decompress_gflops, 6),
            "distortion_mse": distortion_mse
        }
        
        # Saved in decompression_output as requested
        helper.add_to_performance_log(
            os.path.join(output_path, "decompressed_output"), 
            metrics, 
            file_name="decompression_metrics.csv"
        )

        #  SAVING  ---
        if config.extra_compression:
            np.savez_compressed(
                os.path.join(output_path, "decompressed_output", "decompressed.npz"),
                data=decompressed,
                names=names,
            )
        else:
            np.savez(
                os.path.join(output_path, "decompressed_output", "decompressed.npz"),
                data=decompressed,
                names=names,
            )
            
    finally:
        # Stop tracking and output final emissions report
        emissions_kg = tracker.stop()
        if verbose:
            print(f"Decompression Carbon Footprint: {emissions_kg:.6f} kg CO2")
            
            
            
            
            
            
# -------------------------
# Diagnostics / Plotting / Info
# -------------------------
def perform_diagnostics(project_path, verbose: bool):
    output_path = os.path.join(project_path, "plotting")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = os.path.join(project_path, "training", "activations.npy")
    helper.diagnose(input_path, output_path)


def perform_plotting(output_path, config, verbose: bool):
    helper.loss_plotter(os.path.join(output_path, "training", "loss_data.npy"), output_path, config)
    helper.plotter(output_path, config)


def print_info(output_path, config):
    print("### Compression Information ###")
    original = config.input_path
    compressed = os.path.join(output_path, "compressed_output", "compressed.npz")

    orig_size = os.stat(original).st_size / (1024 * 1024)
    comp_size = os.stat(compressed).st_size / (1024 * 1024)

    print(f"Original Size: {orig_size:.4f} MB")
    print(f"Compressed Size: {comp_size:.4f} MB")
    print(f"Ratio: {orig_size / comp_size:.4f}")
