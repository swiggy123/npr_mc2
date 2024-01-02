import re
import pandas as pd
import seaborn as sns


# Specify the log file path
def get_best_model(log_file_path, ckpt_dir="small_trained"):
    ckpt_dir = f"{ckpt_dir}_ckpt"
    # Initialize the best model name variable
    best_model_name = None

    # Read the log file and extract the best model name
    with open(log_file_path, "r", encoding="UTF-8") as file:
        multiline_text = file.read()
    pattern = rf"creating {ckpt_dir}/best_model\n(.+?)\n"
    match = re.search(pattern, multiline_text, re.DOTALL)
    if match:
        model_path = match.group(1)
    else:
        print("Model path not found.")
    # Regular expression pattern
    pattern = rf"copying\s+(.*?)\/epoch_\d+\/\w+\.json\s+->\s+{ckpt_dir}/best_model$"

    # Search for the pattern in the line
    match = re.search(pattern, model_path)
    if match:
        model_path = match.group(1)
        return model_path
    else:
        print("Model path not found in the line.")


def get_loss(log_file_path, best_model_name):
    # Initialize empty lists to store names, epochs, steps, and loss values
    names = []
    epochs = []
    steps = []
    loss_values = []

    # Define regular expression patterns to extract name, epoch, step, and loss values
    name_pattern = r"initialize checkpoint at (.+)"
    epoch_pattern = r"\[epoch (\d+)/"
    step_pattern = r"\(global step (\d+): loss: ([\d.]+), lr: ([\d.]+)"
    average_pattern = r"average loss:\s*([\d.]+)"
    average_losses = []
    epochs_for_avg_losses = []
    current_epoch = 0  # Initialize the current_epoch to 0
    current_name = None
    inside_target = False  # Flag to indicate if we are inside the target section
    # Read the log file and extract name, epoch, step, and loss values using regex
    with open(log_file_path, "r", encoding="UTF-8") as log_file:
        for line in log_file:
            name_match = re.search(name_pattern, line)
            if name_match:
                current_name = name_match.group(1)
            if f"{best_model_name}" in line:
                if inside_target:
                    break  # Stop if we reach the second occurrence of the target
                else:
                    inside_target = True  # Start recording when we find the first occurrence
                    continue

            if inside_target:
                epoch_match = re.search(epoch_pattern, line)
                step_match = re.search(step_pattern, line)
                # Find all matches in the text
                average_match = re.findall(average_pattern, line)

                # Extract the average loss if it's present in a line
                if average_match:
                    average_loss = float(average_match[0])
                    average_losses.append(average_loss)
                    epochs_for_avg_losses.append(current_epoch)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1)) + 1
                elif step_match:
                    step = int(step_match.group(1))
                    loss = float(step_match.group(2))
                    names.append(current_name)
                    epochs.append(current_epoch)
                    steps.append(step)
                    loss_values.append(loss)

    # Create a Pandas DataFrame
    data = {"Name": names, "Epoch": epochs, "Step": steps, "Loss": loss_values}

    df_steps = pd.DataFrame(data)
    data_avg_los = pd.DataFrame({"Epoch": epochs_for_avg_losses, "Loss": average_losses})

    return df_steps, data_avg_los
