"""
Plot and compare the validation error curve of multiple experiments
(assume experiment result folders located under./exp)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')


folder_names = []  # fill in multiple experiment folder names to compare val loss


_, axs = plt.subplots(figsize=(10,8))

for i in range(len(folder_names)):
	folder_name = folder_names[i]
	print(folder_name)

	try:
		label_name = label_names[i]
	except NameError:
		label_name = folder_name

	try:
		color = colors[i]
	except NameError:
		color = None

	file_name = 'exp/%s/all_stat_dict.pth' % (folder_name)
	all_stat_dict = torch.load(file_name)
	x_axis = np.arange(len(all_stat_dict['val/loss'])) + 1

	val_err = (1. - np.asarray(all_stat_dict['val/prec@1'])) * 100
	axs.plot(x_axis, val_err, label=label_name, color=color)

	axs.set_ylabel('val err (%)')
	axs.legend(fontsize="10")
	axs.set_xlabel('epoch')
	print(f'{100. - val_err[-1]:.4f}%')

plt.grid()
plt.show()
