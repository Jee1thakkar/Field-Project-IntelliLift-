from elevator_environment import ElevatorEnvironment
from q_learning_agent import QLearningAgent


# ============================================================
# SETTINGS
# ============================================================

EPISODES = 500


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment():

    return ElevatorEnvironment(
        floors=10,
        number_of_elevators=3,
        capacities=[500, 500, 500]
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_agent(episodes=500):

    # Create environment
    env = create_environment()

    # Create Q-learning agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    )

    reward_history = []

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("              Q-LEARNING TRAINING")
    print("=" * 60)

    print("Episodes :", episodes)
    print("Floors   : 10")
    print("Elevators: 3")
    print("Capacity : 500 kg each")

    print("=" * 60)
    print()

    # ========================================================
    # TRAINING EPISODES
    # ========================================================

    for episode in range(1, episodes + 1):

        # ----------------------------------------------------
        # Generate peak demand
        # ----------------------------------------------------
        # IMPORTANT:
        # Your current environment uses:
        #
        #     env.generate_peak_demand()
        #
        # Do NOT write num_people=...
        # ----------------------------------------------------

        state = env.generate_peak_demand()

        done = False
        episode_reward = 0

        # ====================================================
        # ONE EPISODE
        # ====================================================

        while not done:

            # Get safe/valid elevator actions
            valid_actions = env.get_valid_actions()

            # ------------------------------------------------
            # If no safe elevator is available
            # ------------------------------------------------

            if not valid_actions:
                break

            # ------------------------------------------------
            # Q-learning chooses an action
            # ------------------------------------------------

            action = agent.choose_action(
                state,
                valid_actions
            )

            # ------------------------------------------------
            # Execute selected elevator
            # ------------------------------------------------

            next_state, reward, done, info = env.step(
                action
            )

            # ------------------------------------------------
            # Find valid actions for next state
            # ------------------------------------------------

            if done:

                next_valid_actions = []

            else:

                next_valid_actions = env.get_valid_actions()

            # ------------------------------------------------
            # Q-learning update
            # ------------------------------------------------

            agent.update(
                state,
                action,
                reward,
                next_state,
                next_valid_actions,
                done
            )

            # ------------------------------------------------
            # Move to next state
            # ------------------------------------------------

            state = next_state

            # Add reward
            episode_reward += reward

        # ====================================================
        # REDUCE EXPLORATION
        # ====================================================

        agent.decay_epsilon()

        reward_history.append(
            episode_reward
        )

        # ====================================================
        # DISPLAY TRAINING PROGRESS
        # ====================================================

        if episode % 50 == 0:

            recent_rewards = reward_history[-50:]

            average_reward = (
                sum(recent_rewards)
                / len(recent_rewards)
            )

            print(
                f"Episode {episode:3d} | "
                f"Average Reward: {average_reward:8.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"States: {len(agent.q_table)}"
            )

    # ========================================================
    # TRAINING COMPLETED
    # ========================================================

    print()
    print("=" * 60)
    print("              TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Total Episodes :",
        episodes
    )

    print(
        "Learned States :",
        len(agent.q_table)
    )

    print(
        "Final Epsilon  :",
        round(agent.epsilon, 3)
    )

    print("=" * 60)

    return agent, reward_history


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("Starting Safe Elevator Reinforcement Learning...")
    print()

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    agent, reward_history = train_agent(
        episodes=EPISODES
    )

    # --------------------------------------------------------
    # Save trained Q-table
    # --------------------------------------------------------

    agent.save("q_table.pkl")

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("        RL TRAINING FINISHED SUCCESSFULLY")
    print("=" * 60)

    print(
        "Q-table saved as: q_table.pkl"
    )

    print(
        "Learned states:",
        len(agent.q_table)
    )

    print("=" * 60)