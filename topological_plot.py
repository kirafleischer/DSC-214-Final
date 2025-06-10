import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

df = pd.read_csv("topological_output/losses.csv")
epochs = df["epoch"]

fig, axs = plt.subplots(2, 2)
blocks = [1, 2, 3, 4]
titles = ["Block 1", "Block 2", "Block 3", "Block 4"]

for idx, ax in enumerate(axs.flatten()):
    ax.plot(epochs, df[f"train_block{blocks[idx]}"], label="Train", color="blue")
    ax.plot(epochs, df[f"valid_block{blocks[idx]}"], label="Valid", color="orange")
    ax.plot(epochs, df[f"test_block{blocks[idx]}"], label="Test", color="green")
    ax.set_title(titles[idx])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("topological_output/loss_subplots.png")
print('Saved figure to topological_output/loss_subplots.png')