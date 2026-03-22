import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages

def plot_training_emissions(folder_path, total_epochs):
    csv_path = os.path.join(folder_path, "emission_training.csv")
    output_pdf = os.path.join(folder_path, "training_emission.pdf")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    #LOAD DATA
    raw_data = {"cpu": [], "gpu": [], "ram": [], "total": [], "time": [], "co2": []}
    
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_data["cpu"].append(float(row.get("cpu_energy", 0)))
            raw_data["gpu"].append(float(row.get("gpu_energy", 0)))
            raw_data["ram"].append(float(row.get("ram_energy", 0)))
            raw_data["total"].append(float(row.get("energy_consumed", 0)))
            raw_data["time"].append(float(row.get("duration", 0)))
            raw_data["co2"].append(float(row.get("emissions", 0)))

    # CALCULATE DELTAS
    # We want exactly total_epochs worth of differences.

    KWH_TO_JOULES = 3.6e6
    deltas = {}

    for key in raw_data.keys():
       
        data_trimmed = raw_data[key][:total_epochs]
        
        
        data_with_base = np.insert(data_trimmed, 0, 0.0)
        diff_result = np.diff(data_with_base)
        
        
        final_plot_data = np.insert(diff_result, 0, 0.0)
        
        if key in ["time", "co2"]:
            deltas[key] = final_plot_data
        else:
            deltas[key] = final_plot_data * KWH_TO_JOULES

    # SET THE X-AXIS
    # This will be exactly [0, 1, 2, ..., total_epochs] (11 points)
    x_axis = np.arange(0, total_epochs + 1)

    with PdfPages(output_pdf) as pdf:
        
        # PAGE 1: HARDWARE ENERGY (J) ---
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
        hw_configs = [('cpu', 'CPU Energy (J)'), 
                      ('gpu', 'GPU Energy (J)'), 
                      ('ram', 'RAM Energy (J)')]
        
        for i, (key, ylabel) in enumerate(hw_configs):
            axes1[i].plot(x_axis, deltas[key], color='tab:blue', linewidth=1)
            axes1[i].set_title(ylabel.replace(" (J)", ""))
            axes1[i].set_xlabel("Epoch")
            axes1[i].set_ylabel(ylabel)
            axes1[i].set_xlim(0, total_epochs)
            axes1[i].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig1)
        plt.close()

        # PAGE 2: TOTAL CONSUMPTION (J) and DURATION (s) ---
        fig2, (ax_e, ax_d) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax_e.plot(x_axis, deltas["total"], color='tab:blue', linewidth=1)
        ax_e.set_title("Total energy consumed per epoch")
        ax_e.set_xlabel("Epoch")
        ax_e.set_ylabel("Total Energy Consumed (J)")
        ax_e.set_xlim(0, total_epochs)
        ax_e.grid(True, alpha=0.3)
        
        ax_d.plot(x_axis, deltas["time"], color='tab:blue', linewidth=1)
        ax_d.set_title("Duration of one epoch")
        ax_d.set_xlabel("Epoch")
        ax_d.set_ylabel("Duration (s)")
        ax_d.set_xlim(0, total_epochs)
        ax_d.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close()

        # PAGE 3: CO2 EMISSIONS (kg) ---
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(x_axis, deltas["co2"], color='tab:blue', linewidth=1)
        ax3.set_title("CO2 Emission")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("CO2 Emission (kg)")
        ax3.set_xlim(0, total_epochs)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig3)
        plt.close()

    print(f" PDF generated for {total_epochs} epochs.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <folder_path> <total_epochs>")
        sys.exit(1)
        
    folder = sys.argv[1]
    epochs = int(sys.argv[2])
    plot_training_emissions(folder, epochs)
