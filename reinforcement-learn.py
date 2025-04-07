import numpy as np

# Grid world setup
grid_size = 4  # 4x4 grid
actions = ['up', 'down', 'left', 'right']  # Possible moves
goal = (3, 3)  # Goal is at the bottom-right corner

# Q-table: Stores the "goodness" of each action in each state
Q = np.zeros((grid_size, grid_size, len(actions)))

# Hyperparameters
alpha = 0.1  # Learning rate (how fast the agent learns)
gamma = 0.9  # Discount factor (how much the agent cares about future rewards)
epsilon = 1.0  # Exploration rate (start by exploring a lot)
epsilon_decay = 0.995  # Reduce exploration over time
epsilon_min = 0.01  # Minimum exploration rate
num_episodes = 1000  # Number of training rounds

# Function to choose an action (explore or exploit)
def choose_action(state):
    if np.random.rand() < epsilon:  # Explore: choose a random action
        return np.random.choice(len(actions))
    else:  # Exploit: choose the best action from the Q-table
        return np.argmax(Q[state[0], state[1]])

# Function to move the agent
def move(state, action):
    x, y = state
    if action == 0:  # Up
        x = max(x - 1, 0)
    elif action == 1:  # Down
        x = min(x + 1, grid_size - 1)
    elif action == 2:  # Left
        y = max(y - 1, 0)
    elif action == 3:  # Right
        y = min(y + 1, grid_size - 1)
    return (x, y)

# Training the agent
for episode in range(num_episodes):
    state = (0, 0)  # Start at the top-left corner
    done = False
    
    while not done:
        # Choose an action
        action = choose_action(state)
        
        # Move to the next state
        next_state = move(state, action)
        
        # Calculate reward
        reward = 10 if next_state == goal else -1
        
        # Update Q-table
        Q[state[0], state[1], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])
        
        # Move to the next state
        state = next_state
        
        # Check if the goal is reached
        if state == goal:
            done = True
    
    # Reduce exploration over time
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

# Test the trained agent
state = (0, 0)
steps = 0
while state != goal:
    action = np.argmax(Q[state[0], state[1]])  # Choose the best action
    state = move(state, action)  # Move to the next state
    steps += 1
    print(f"Step {steps}: Move {actions[action]} to {state}")

print(f"Reached the goal in {steps} steps!")