import os
import pickle
import random

from elevator_environment import ElevatorEnvironment
from q_learning_agent import QLearningAgent


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 10000

MIN_PEOPLE = 10
MAX_PEOPLE = 20

MODEL_DIR = "models"
Q_TABLE_FILE = os.path.join(MODEL_DIR, "q_table.pkl")
REWARD_HISTORY_FILE = os.path.join(MODEL_DIR, "reward_history.pkl")


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = ElevatorEnvironment(
    floors=10,
    number_of_elevators=3,
    capacities=[500, 500, 500]
)


# ============================================================
# CREATE Q-LEARNING AGENT
# ============================================================

agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05
)


# ============================================================
# TRAINING
# ============================================================

reward_history = []

print("=" * 70)
print("              SAFE Q-LEARNING ELEVATOR TRAINING")
print("=" * 70)

print(f"Episodes        : {EPISODES}")
print(f"People          : {MIN_PEOPLE} - {MAX_PEOPLE}")
print(f"Floors          : 10")
print(f"Elevators       : 3")
print(f"Model file      : {Q_TABLE_FILE}")
print("=" * 70)

print()


for episode in range(1, EPISODES + 1):

    # --------------------------------------------------------
    # Random number of passengers
    # --------------------------------------------------------

    number_of_people = random.randint(
        MIN_PEOPLE,
        MAX_PEOPLE
    )

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    state = env.reset(
        number_of_people=number_of_people
    )

    done = False
    total_reward = 0.0

    # --------------------------------------------------------
    # Run one episode
    # --------------------------------------------------------

    while not done:

        # Get only safe actions
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            break

        # ----------------------------------------------------
        # Select action using Q-learning
        # ----------------------------------------------------

        action = agent.choose_action(
            state,
            valid_actions
        )

        if action is None:
            break

        # ----------------------------------------------------
        # Perform action
        # ----------------------------------------------------

        next_state, reward, done, info = env.step(action)

        # ----------------------------------------------------
        # Get next safe actions
        # ----------------------------------------------------

        if not done:
            next_valid_actions = env.get_valid_actions()
        else:
            next_valid_actions = []

        # ----------------------------------------------------
        # Update Q-table
        # ----------------------------------------------------

        agent.update(
            state,
            action,
            reward,
            next_state,
            next_valid_actions,
            done
        )

        # ----------------------------------------------------
        # Move to next state
        # ----------------------------------------------------

        state = next_state

        total_reward += reward

    # --------------------------------------------------------
    # Reduce exploration
    # --------------------------------------------------------

    agent.decay_epsilon()

    # --------------------------------------------------------
    # Store episode reward
    # --------------------------------------------------------

    reward_history.append(total_reward)

    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    if episode == 1 or episode % 100 == 0:

        average_reward = (
            sum(reward_history[-100:]) /
            len(reward_history[-100:])
        )

        print(
            f"Episode: {episode:5d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Avg Reward: {average_reward:8.2f} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"States: {agent.number_of_states()}"
        )


# ============================================================
# SAVE TRAINED Q-TABLE
# ============================================================

print()
print("=" * 70)
print("                 SAVING TRAINED MODEL")
print("=" * 70)

agent.save(Q_TABLE_FILE)


# ============================================================
# SAVE REWARD HISTORY
# ============================================================

with open(REWARD_HISTORY_FILE, "wb") as file:
    pickle.dump(reward_history, file)


print()
print("=" * 70)
print("                 TRAINING COMPLETED")
print("=" * 70)

print(f"Q-table saved       : {Q_TABLE_FILE}")
print(f"Reward history      : {REWARD_HISTORY_FILE}")
print(f"Learned states      : {agent.number_of_states()}")
print(f"Final epsilon       : {agent.epsilon:.4f}")
print(f"Total episodes      : {EPISODES}")

print("=" * 70)