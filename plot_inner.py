"""
Plot the inner loss at each inner optimization step in all TTT layers on imagenet validation set throughout training
(assume experiment result folders located under./exp)
Inner loss only applies to TTT layers
For linear attention and self-attention, we set inner loss to inf to ensure it's meaningless
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')


folder_name = ''  # fill in the name of one experiment folder to visualize inner loss curves

file_name = './exp/%s/all_stat_dict.pth' % (folder_name)
all_stat_dict = torch.load(file_name)
print(file_name)

inner_loss = all_stat_dict['val/inner_loss']
nrows = 3
ncols = 4
assert(nrows * ncols == len(inner_loss))
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 6))

for layer in range(len(inner_loss)):
	for itr in range(len(inner_loss[layer])):
		loss = inner_loss[layer][itr]

		x_axis = np.arange(len(loss)) + 1
		axs.flatten()[layer].plot(x_axis, loss, label='%d' % (itr))
	axs.flatten()[layer].legend()
	axs.flatten()[layer].set_title('layer %d' % (layer + 1))

fig.suptitle(folder_name)
plt.show()
