import numpy as np
import matplotlib.pyplot as plt

avg_reward = np.load('no_target_net/average_data.npy')
reward = np.load('no_target_net/episode_data.npy')

plt.rc('font', family='serif')
plt.figure(figsize=(12, 14))

ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)        
ax.spines["right"].set_visible(False) 

ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# plt.plot(np.arange(2000), reward, color=[0.866, 0.596, 0.850])
# plt.plot(np.arange(2000), avg_reward, color=[0.6, 0.384, 0.239], linewidth=2.5)
plt.plot(np.arange(2000), reward, color=[0.709, 0.341, 0.050])
plt.plot(np.arange(2000), avg_reward, color=[0.105, 0.207, 0.733], linewidth=2.5)
plt.xlabel('Episode', fontsize=17)
plt.ylabel('Reward', fontsize=17)

plt.grid()
plt.show()