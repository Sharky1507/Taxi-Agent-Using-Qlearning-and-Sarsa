import gymnasium as gym
from sarsa_agent import SARSAAgent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# WRITE YOUR CODE HERE

def train_agent(agent,env,episodes,eval_interval):
    best_average = -np.inf
    rewards = []    
    for episode in range(episodes):
        obs,_ = env.reset()
        done = 0
        steps = 0
        total_reward = 0
        while not done:
            action = agent.get_action(obs)
            next_obs,reward,terminated,truncated,_ = env.step(action)
            agent.update(obs,action,reward,terminated,next_obs)
            obs= next_obs
            done = terminated or truncated
            total_reward+=reward
            steps+=1
        rewards.append(total_reward)
        agent.decay_epsilon()
        if episode >= eval_interval:
            average = np.mean(rewards[episode-eval_interval:episode])
            if average > best_average:
                best_average = average
        if episode % eval_interval == 0:
            print(f'Episode {episode} : length = {steps}, reward: {best_average}')
    return rewards
def plot_returns(plot_reward,file_name):
    plt.plot(np.arange(len(plot_reward)),plot_reward)
    plt.title("Episode rewards")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Rewards")
    plt.savefig(file_name)
    plt.show()
def show_policy(agent,env):
    agent.epsilon = 0
    obs,_ = env.reset()
    env.render()
    done = False
    while not done:
        
            action = agent.get_action(obs)
            next_obs,reward,terminated,truncated,_ = env.step(action)
            env.render()
            done = truncated or terminated
            obs = next_obs
            
    
episodes = 1000
learning_rate = 0.5
initial_epsilon = 1.0
final_epsilon = 0.2
epsilon_decay = (initial_epsilon-final_epsilon)/(episodes/2)
env = gym.make('Taxi-v3')
agent = SARSAAgent(env=env, learning_rate=learning_rate, initial_epsilon = initial_epsilon, epsilon_decay = epsilon_decay, final_epsilon= final_epsilon)
plot_reward = train_agent(agent,env,episodes,eval_interval=100)
#plot_returns(plot_reward,file_name='sarsa_learning_curve.png')
#env = gym.make('Taxi-v3',render_mode = 'human')
#show_policy(agent,env)