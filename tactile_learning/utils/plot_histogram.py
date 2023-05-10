import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
print(plt.rcParams["font.size"])
plt.rcParams["font.size"] = 12.0


tasks = ['Joystick Movement', 'Bottle Opening', 'Cup Unstacking', 'Bowl Unstacking', 'Book Opening']

bc_success = [1, 1, 1, 1, 1]
nn_image_success = [40, 1, 1, 20, 50]
nn_tactile_success = [60, 60, 1, 20, 1]
nn_task_success = [80, 30, 40, 30, 60]
nn_torque_success = [70, 30, 20, 40, 30]
tdex_success = [80, 60, 80, 70, 90]
bet_success = [1, 10, 1, 20, 10]
ibc_success = [1, 10, 5, 30, 1]
bc_gmm_success = [1, 1, 10, 1, 1]

N = len(tasks)
ind = np.arange(N)

width = 0.1
task_idxs = [0-width, N-width]
fig, ax = plt.subplots(figsize=(20,5))
ax.grid(axis="y")

rects = ax.bar(ind, bc_success, width, label="BC", zorder=10)
rects2 = ax.bar(ind+width, nn_image_success, width, label="NN-Image", zorder=20)
rects3 = ax.bar(ind+2*width, nn_tactile_success, width, label="NN-Tactile", zorder=30)
rects4 = ax.bar(ind+3*width, nn_task_success, width, label="NN-Task", zorder=40)
rects5 = ax.bar(ind+4*width, nn_torque_success, width, label="NN-Torque", zorder=50)
rects6 = ax.bar(ind+5*width, bet_success, width, label="BET", zorder=60)
rects7 = ax.bar(ind+6*width, ibc_success, width, label="IBC", zorder=70)
rects8 = ax.bar(ind+7*width, bc_gmm_success, width, label="BC-GMM", zorder=80)
rects9 = ax.bar(ind+8*width, tdex_success, width, label="T-DEX (Ours))", zorder=90)



# play_amount = [1, 5, 20, 80, 150]
# play_amount = [str(x) for x in play_amount]
# book_opening_success = [60, 60, 90, 90, 90]
# cup_unstacking_success = [20, 10, 40, 60, 80]
# N = len(play_amount)
# ind = np.arange(N)
# width = 0.3
# task_idxs = [0 - width, N - width]
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.grid(axis="y")
# rects1 = ax.bar(ind, book_opening_success, width, label="Book Opening", zorder=10)
# rects3 = ax.bar(
#     ind + width, cup_unstacking_success, width, label="Cup Unstacking", zorder=20
# )

ax.set_ylim(-10, 100)
ax.set_xticks(ind + 9*width / 2)
ax.set_xticklabels(tasks, fontsize=19)
# ax.set_xlabel("Tasks", fontsize=15)
ax.set_ylabel("Success Rate (%)", fontsize=20)
ax.set_title("Method Successes", fontsize=30)
# ax.legend(fontsize=10)
ax.legend(ncol=5, bbox_to_anchor=(0.5, 0.985), fontsize=13, loc='upper center', borderaxespad=0)
plt.savefig("figures/method_vs_success.png")