import sys
import numpy as np
import matplotlib.pyplot as plt
# plot every 4 th point of higest  l=plot every 2 point  of mid and plot every for the lowest
# multiple lines in the same plot with different colors
# plot the error data across different resolutions
# turbulent 
# todo compile the code and run it
# 
  
def main():
    if len(sys.argv) != 5:
        print("Usage: python plot_error_data.py <low_res.txt> <mid_res.txt> <high_res.txt> <output.png>")
        sys.exit(1)

    # Command-line arguments
    low_res_path, mid_res_path, high_res_path, output_path = sys.argv[1:5]

    # File paths
    file_paths = [low_res_path, mid_res_path, high_res_path]
    labels = ["Low Resolution (64)", "Mid Resolution (128)", "High Resolution (256)"]

    # Read data from each file
    data = []
    for i, path in enumerate(file_paths):
        try:
            with open(path, "r") as file:
                values = np.loadtxt(file)
                data.append(values)
                print(f"Loaded data from {path}")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            sys.exit(1)

    # Plot the data
    plt.figure(figsize=(10, 6))
    for i, (values, label) in enumerate(zip(data, labels)):
        plt.plot(values, label=label)

    # Add labels, title, and legend
    plt.xlabel("Pixel Index")
    plt.ylabel("Error Value")
    plt.title("Error Data Across Different Resolutions")
    plt.legend()

    # Save the plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
