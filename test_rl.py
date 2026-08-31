from elevator_environment import ElevatorEnvironment
from q_learning_agent import QLearningAgent


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = ElevatorEnvironment(
    floors=10,
    number_of_elevators=3,
    capacities=[500, 500, 500]
)


# ============================================================
# CREATE AGENT
# ============================================================

agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.0,
    epsilon_min=0.05,
    epsilon_decay=0.995
)


# ============================================================
# LOAD TRAINED Q-TABLE
# ============================================================

# IMPORTANT:
# This assumes your q_learning_agent.py has a saved Q-table.
# If your current agent does NOT save/load the Q-table,
# we will add that next.


print("=" * 60)
print("             RL AGENT TEST")
print("=" * 60)

print("Testing Q-Learning agent...")
print()


# ============================================================
# GENERATE PEAK DEMAND
# ============================================================

state = env.generate_peak_demand()

print("Peak demand generated.")
print()


# ============================================================
# TEST AGENT
# ============================================================

done = False
total_reward = 0

while not done:

    valid_actions = env.get_valid_actions()

    if not valid_actions:

        print("No safe elevator available.")
        break

    # Choose learned action
    action = agent.choose_action(
        state,
        valid_actions
    )

    # Execute action
    next_state, reward, done, info = env.step(
        action
    )

    total_reward += reward

    print("-" * 60)
    print("Passenger served:")
    print(info)

    print(
        "Selected Elevator:",
        action + 1
    )

    print(
        "Reward:",
        reward
    )

    state = next_state


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("              RL TEST RESULTS")
print("=" * 60)

print(
    "Total Reward:",
    total_reward
)

print(
    "People Served:",
    env.total_served
)

print(
    "Total Distance:",
    env.total_distance
)

print(
    "Safety Violations:",
    env.safety_violations
)

print(
    "Overload Attempts:",
    env.overload_attempts
)

if env.total_served > 0:

    print(
        "Average Waiting Time:",
        env.total_waiting_time / env.total_served
    )

print("=" * 60)
print("RL TEST COMPLETED")
print("=" * 60)