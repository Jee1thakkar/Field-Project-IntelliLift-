from elevator_environment import ElevatorEnvironment
from q_learning_agent import QLearningAgent


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 500
PEOPLE_PER_EPISODE = 10

MODEL_PATH = "models/q_table.pkl"


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment():

    env = ElevatorEnvironment(
        floors=10,
        number_of_elevators=3,
        capacities=[
            500,
            500,
            500
        ]
    )

    return env


# ============================================================
# CREATE Q-LEARNING AGENT
# ============================================================

def create_agent():

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    return agent


# ============================================================
# TRAIN Q-LEARNING AGENT
# ============================================================

def train_agent(
    episodes=EPISODES,
    people_per_episode=PEOPLE_PER_EPISODE
):

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = create_environment()

    # --------------------------------------------------------
    # Create RL agent
    # --------------------------------------------------------

    agent = create_agent()

    # --------------------------------------------------------
    # Store rewards for analysis
    # --------------------------------------------------------

    reward_history = []

    # ========================================================
    # TRAINING INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("                 Q-LEARNING TRAINING")
    print("=" * 70)

    print(
        f"Episodes            : {episodes}"
    )

    print(
        f"People per episode  : {people_per_episode}"
    )

    print(
        f"Learning rate       : {agent.alpha}"
    )

    print(
        f"Discount factor     : {agent.gamma}"
    )

    print(
        f"Initial epsilon     : {agent.epsilon}"
    )

    print(
        f"Epsilon decay       : {agent.epsilon_decay}"
    )

    print(
        f"Minimum epsilon     : {agent.epsilon_min}"
    )

    print("=" * 70)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(1, episodes + 1):

        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        state = env.reset(
            number_of_people=people_per_episode
        )

        done = False

        episode_reward = 0

        episode_steps = 0

        # ----------------------------------------------------
        # Episode loop
        # ----------------------------------------------------

        while not done:

            # Get safe/valid elevators
            valid_actions = env.get_valid_actions()

            # ------------------------------------------------
            # No safe elevator available
            # ------------------------------------------------

            if not valid_actions:

                break

            # ------------------------------------------------
            # Agent selects elevator
            # ------------------------------------------------

            action = agent.choose_action(
                state,
                valid_actions
            )

            # Safety check
            if action is None:

                break

            # ------------------------------------------------
            # Environment executes action
            # ------------------------------------------------

            (
                next_state,
                reward,
                done,
                info
            ) = env.step(action)

            # ------------------------------------------------
            # Get next valid actions
            # ------------------------------------------------

            if done:

                next_valid_actions = []

            else:

                next_valid_actions = (
                    env.get_valid_actions()
                )

            # ------------------------------------------------
            # Update Q-table
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

            episode_reward += reward

            episode_steps += 1

        # ----------------------------------------------------
        # Reduce exploration
        # ----------------------------------------------------

        agent.decay_epsilon()

        # ----------------------------------------------------
        # Save episode reward
        # ----------------------------------------------------

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
                f"Episode {episode:4d} | "
                f"Avg Reward: {average_reward:9.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"States: {agent.number_of_states():6d}"
            )

    # ========================================================
    # TRAINING COMPLETED
    # ========================================================

    print()
    print("=" * 70)
    print("                 TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Total episodes       : {episodes}"
    )

    print(
        f"Learned states       : "
        f"{agent.number_of_states()}"
    )

    print(
        f"Final epsilon        : "
        f"{agent.epsilon:.4f}"
    )

    if reward_history:

        final_rewards = reward_history[-50:]

        final_average_reward = (
            sum(final_rewards)
            / len(final_rewards)
        )

        print(
            f"Final 50-episode avg : "
            f"{final_average_reward:.2f}"
        )

    print("=" * 70)

    # ========================================================
    # SAVE TRAINED Q-TABLE
    # ========================================================

    print()
    print("Saving trained RL model...")

    agent.save(
        MODEL_PATH
    )

    print()
    print("Training and model saving completed successfully.")

    return (
        agent,
        env,
        reward_history
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    agent, env, rewards = train_agent(
        episodes=EPISODES,
        people_per_episode=PEOPLE_PER_EPISODE
    )

    print()
    print("=" * 70)
    print("             RL TRAINING PROGRAM FINISHED")
    print("=" * 70)

    print(
        "Trained model:"
    )

    print(
        MODEL_PATH
    )

    print(
        f"Learned Q-table states: "
        f"{agent.number_of_states()}"
    )

    print("=" * 70)