import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

N_PLAYS = 1000
N_BANDITS = 10

normal = np.random.normal
cmap = mpl.cm.get_cmap('tab20')

def run_task(mu, var, policy):
	std_dev = np.sqrt(var)

	# estimates of action values for the arms
	Q_a_est = np.random.uniform(low=-1, high=1, size=(N_BANDITS,))
	
	# track no. of times actions are taken
	a_steps = np.ones((N_BANDITS,))
	# track UCB for all arms
	UCB = np.zeros((N_BANDITS,))

	play_rewards = np.zeros((N_PLAYS,))

	# no. of times optimal action was chosen
	opt_action_chosen = np.zeros((N_PLAYS,))

	# run current task for N_PLAYS
	total_reward = 0
	for i in range(N_PLAYS):
		if policy == "greedy":
			action = np.argmax(Q_a_est)
		elif policy == "eps-greedy":
			epsilon = 0.1
			# exploit
			if np.random.random() > epsilon:
				action = np.argmax(Q_a_est)
			# explore
			else:
				action = np.random.randint(low=0, high=N_BANDITS)
		elif policy == "softmax":
			tau = 0.1
			# calculate softmax activations
			softmax_prob = np.exp(Q_a_est/tau)/(np.sum(np.exp(Q_a_est/tau)))
			action = np.random.choice(range(N_BANDITS), 1, p=softmax_prob)
		elif policy == "UCB1":
			# choose action according max UCB
			action = np.argmax(Q_a_est + np.sqrt(2*np.log(i+1)/(a_steps)))
			# print("HI!")
		
		# get rewards for each action
		reward = normal(mu, var)
		# check if optimal action was taken
		if action == np.argmax(mu):
			opt_action_chosen[i] += 1
		# no. of times action was taken
		k_a = a_steps[action]
		# update action value estmate
		Q_a_est[action] = (Q_a_est[action]*k_a + reward[action])/(k_a + 1)
		# update action steps tracker
		a_steps[action] = k_a + 1

		play_rewards[i] = reward[action]

	return play_rewards, opt_action_chosen

def run_test(policy):
	avg_rewards = np.zeros((N_PLAYS,))
	avg_opt_choose = np.zeros((N_PLAYS))
	rewards = np.zeros((2000, N_PLAYS))
	
	# run 2000 tasks
	for i in range(2000):
		# get true action value params
		mu = normal(0, 1, (N_BANDITS,))
		var = np.ones((N_BANDITS,))

		# get rewards over N_PLAYS for the current task
		task_rewards, opt_action_choose = run_task(mu, var, policy)

		rewards[i,:] = task_rewards
		avg_rewards += task_rewards
		avg_opt_choose += opt_action_choose

	return avg_rewards/2000, rewards, avg_opt_choose/2000

def main():
	plays = np.linspace(1, N_PLAYS, N_PLAYS)
	greedy, _, opt_greedy = run_test("greedy")
	eps_greedy, _, opt_eps_greedy = run_test("eps-greedy")
	softmax, _, opt_softmax = run_test("softmax")
	UCB1, _, opt_UCB1 = run_test("UCB1")


	# plotting
	plt.figure(figsize=(12, 9))
	ax = plt.subplot(111)
	plt.plot(plays, greedy, color=cmap(0.4), label="Greedy")
	plt.plot(plays, eps_greedy, color=cmap(0.3), label="Eps-Greedy")
	plt.plot(plays, softmax, color=cmap(0.2), label="Softmax")
	plt.plot(plays, UCB1, color=cmap(0.1), label="UCB1")
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)    
	ax.get_xaxis().tick_bottom()    
	ax.get_yaxis().tick_left()  
	plt.yticks(fontsize=12)
	plt.xticks(fontsize=12)
	plt.xlabel("Plays", fontsize=14)
	plt.ylabel("Average Reward", fontsize=14)
	plt.legend(loc="upper right")
	plt.show()

	# plotting
	plt.figure(figsize=(12, 9))
	ax = plt.subplot(111)
	plt.plot(plays, opt_greedy, color=cmap(0.4), label="Greedy")
	plt.plot(plays, opt_eps_greedy, color=cmap(0.3), label="Eps-Greedy")
	plt.plot(plays, opt_softmax, color=cmap(0.2), label="Softmax")
	plt.plot(plays, opt_UCB1, color=cmap(0.1), label="UCB1")
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)    
	ax.get_xaxis().tick_bottom()    
	ax.get_yaxis().tick_left()  
	plt.yticks(fontsize=12)
	plt.xticks(fontsize=12)
	plt.xlabel("Plays", fontsize=14)
	plt.ylabel("% Optimal action", fontsize=14)
	plt.legend(loc="upper right")
	plt.show()

if __name__ == '__main__':
	main()