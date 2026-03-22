import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

def get_cumulative_metrics(file_path, is_training):
    """
    Processes the CSV: sums all rows for training, 
    
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    
    df = pd.read_csv(file_path)
    
    if is_training:
        # For training: Sum all timestamps to get the total cumulative footprint
        metrics = {
            "cpu_energy": df["cpu_energy"].sum(),
            "gpu_energy": df["gpu_energy"].sum(),
            "ram_energy": df["ram_energy"].sum(),
            "emissions": df["emissions"].sum(),
            "duration": df["duration"].sum()
        }
    else:
        # For compress/decompress: Use the last row (cumulative summary)
        last_row = df.iloc[-1]
        metrics = {
            "cpu_energy": last_row["cpu_energy"],
            "gpu_energy": last_row["gpu_energy"],
            "ram_energy": last_row["ram_energy"],
            "emissions": last_row["emissions"],
            "duration": last_row["duration"]
        }
    return metrics

def plot_comparisons(base_output_path):
    # 1. Setup Paths
    plot_dir = os.path.join(base_output_path, "plotting")
    os.makedirs(plot_dir, exist_ok=True)

    # Define the specific CSV file names
    files = {
        "Decompression": os.path.join(base_output_path, "decompressed_output", "emission_decompression.csv"),
        "Compression": os.path.join(base_output_path, "compressed_output", "emission_compression.csv"),
        "Training": os.path.join(base_output_path, "training", "emission_training.csv")
    }

    # Collect and Process Data
    results = {}
    for mode, path in files.items():
        results[mode] = get_cumulative_metrics(path, is_training=(mode == "Training"))

    # Format Data for Plotting
    KWH_TO_JOULES = 3.6e6
    hw_data = []
    summary_data = []
    
    # Define order for the X-axis
    order = ["Decompression", "Compression", "Training"]

    for mode in order:
        res = results[mode]
        if res:
            # Hardware breakdown (in Joules)
            hw_data.append({"Procedure": mode, "Hardware": "CPU", "Energy (J)": res["cpu_energy"] * KWH_TO_JOULES})
            hw_data.append({"Procedure": mode, "Hardware": "GPU", "Energy (J)": res["gpu_energy"] * KWH_TO_JOULES})
            hw_data.append({"Procedure": mode, "Hardware": "RAM", "Energy (J)": res["ram_energy"] * KWH_TO_JOULES})
            
            # Overall metrics
            summary_data.append({
                "Procedure": mode, 
                "CO2 Emission (kg)": res["emissions"],
                "Duration (s)": res["duration"]
            })

    df_hw = pd.DataFrame(hw_data)
    df_sum = pd.DataFrame(summary_data)

    # PLOTTING ---
    sns.set_theme(style="whitegrid")
    hw_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"] # Blue, Orange, Green

    # Create figure 
    fig = plt.figure(figsize=(14, 12))

    # Define grid: 2 rows, 2 columns
    # ax1 (Energy) spans both columns of Row 0
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    # ax2 (Emissions) is Row 1, Col 0
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    # ax3 (Duration) is Row 1, Col 1
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    # Hardware Energy Plot
    sns.barplot(data=df_hw, x="Procedure", y="Energy (J)", hue="Hardware", palette=hw_palette, ax=ax1)
    ax1.set_yscale('log')
    ax1.set_title("Energy Consumption per Hardware Component", fontsize=15, fontweight='bold')
    ax1.set_ylabel("Log(Energy [J])", fontsize=12)
    ax1.set_xlabel("Procedure", fontsize=12)

    # Total CO2 Emission Plot
    sns.barplot(data=df_sum, x="Procedure", y="CO2 Emission (kg)", ax=ax2, color="#1f77b4")
    ax2.set_yscale('log')
    ax2.set_title("Total CO2 Emission", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Log(CO2 Emission [kg])", fontsize=12)

    # 3. Total Duration Plot
    sns.barplot(data=df_sum, x="Procedure", y="Duration (s)", ax=ax3, color="#1f77b4")
    ax3.set_yscale('log')
    ax3.set_title("Total Duration", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Log(Duration [s])", fontsize=12)

    plt.tight_layout(pad=3.0)
    
    # Save the combined dashboard to a single PDF
    output_filename = os.path.join(plot_dir, "environmental_impact_comparison.pdf")
    plt.savefig(output_filename)
    plt.close()

    print(f"\n[Done] All plots saved in one file: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide the base output folder path.")
        print("Usage: poetry run python plot_comparison.py <path>")
        sys.exit(1)
        
    plot_comparisons(sys.argv[1])
