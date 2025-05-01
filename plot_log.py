import re
import matplotlib.pyplot as plt

# Load the log file
log_path = "./logs/train_2025-04-30_20-51.log"
with open(log_path, "r") as f:
    log_data = f.readlines()

num_batches = 3568

losses = []
accuracies = []
updates = []

for line in log_data:
    loss_match = re.search(r"Avg Loss: ([\d.]+)", line)
    acc_match = re.search(r"Masked Acc: ([\d.]+)%", line)
    epoch_match = re.search(r"Epoch \[(\d+)/", line)
    batch_match = re.search(r"Batch \[(\d+)\]", line)

    if loss_match and acc_match and epoch_match and batch_match:
        epoch = int(epoch_match.group(1))
        batch = int(batch_match.group(1))
        update = (epoch - 1) * num_batches + batch

        losses.append(float(loss_match.group(1)))
        accuracies.append(float(acc_match.group(1)))
        updates.append(update)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)  # width=10in, height=5in, high-res
ax1.set_xlabel('Number of Updates')
ax1.set_ylabel('Avg Loss', color='darkorange')
ax1.plot(updates, losses, color='darkorange', label='Avg Loss')
ax1.tick_params(axis='y', labelcolor='darkorange')

ax2 = ax1.twinx()
ax2.set_ylabel('Masked Accuracy (%)', color='seagreen')
ax2.plot(updates, accuracies, color='seagreen', label='Masked Accuracy')
ax2.tick_params(axis='y', labelcolor='seagreen')

fig.tight_layout()
plt.title("Training Avg Loss and Masked Accuracy")
plt.grid(True)
plt.savefig("log.png", bbox_inches='tight', dpi=300)
