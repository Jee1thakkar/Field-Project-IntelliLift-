import random
from collections import defaultdict

from elevator_environment import ElevatorEnvironment


# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    ):

        # Learning rate
        self.alpha = learning_rate

        # Importance of future rewards
        self.gamma = discount_factor

        # Exploration probability
        self.epsilon = epsilon

        # Reduce exploration after every episode
        self.epsilon_decay = epsilon_decay

        # Minimum exploration
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = defaultdict(
            lambda: defaultdict(float)
        )

    # ========================================================
    # CHOOSE ACTION
    # ========================================================

    def choose_action(self, state, valid_actions):

        if not valid_actions:
            return None

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if random.random() < self.epsilon:

            return random.choice(valid_actions)

        # ----------------------------------------------------
        # Exploitation
        # ----------------------------------------------------

        q_values = [
            self.q_table[state][action]
            for action in valid_actions
        ]

        max_q = max(q_values)

        best_actions = [
            action
            for action in valid_actions
            if self.q_table[state][action] == max_q
        ]

        return random.choice(best_actions)

    # ========================================================
    # UPDATE Q-TABLE
    # ========================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_valid_actions,
        done
    ):

        current_q = self.q_table[state][action]

        # ----------------------------------------------------
        # If episode is finished
        # ----------------------------------------------------

        if done or not next_valid_actions:

            future_q = 0.0

        else:

            future_q = max(
                self.q_table[next_state][next_action]
                for next_action in next_valid_actions
            )

        # ----------------------------------------------------
        # Q-learning formula
        # ----------------------------------------------------

        target = (
            reward
            + self.gamma * future_q
        )

        updated_q = (
            current_q
            + self.alpha
            * (target - current_q)
        )

        self.q_table[state][action] = updated_q

    # ========================================================
    # REDUCE EXPLORATION
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment():

    return ElevatorEnvironment(
        floors=10,
        number_of_elevators=3,
        capacities=[
            500,
            500,
            500
        ]
    )


# ============================================================
# TRAIN Q-LEARNING AGENT
# ============================================================

def train_agent(
    episodes=500,
    people_per_episode=10
):

    env = create_environment()

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    reward_history = []

    print("\n" + "=" * 60)
    print("           Q-LEARNING TRAINING")
    print("=" * 60)

    print(
        f"Training episodes : {episodes}"
    )

    print(
        f"People per episode: {people_per_episode}"
    )

    print("=" * 60)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(1, episodes + 1):

        # Reset environment
        state = env.reset(
            number_of_people=people_per_episode
        )

        done = False

        episode_reward = 0

        # ----------------------------------------------------
        # Episode loop
        # ----------------------------------------------------

        while not done:

            valid_actions = (
                env.get_valid_actions()
            )

            # No safe elevator
            if not valid_actions:

                break

            # Choose elevator
            action = agent.choose_action(
                state,
                valid_actions
            )

            # Execute action
            (
                next_state,
                reward,
                done,
                info
            ) = env.step(action)

            # Get actions available from new state
            next_valid_actions = (
                env.get_valid_actions()
            )

            # Update Q-table
            agent.update(
                state,
                action,
                reward,
                next_state,
                next_valid_actions,
                done
            )

            # Move to next state
            state = next_state

            episode_reward += reward

        # Reduce exploration
        agent.decay_epsilon()

        reward_history.append(
            episode_reward
        )

        # ----------------------------------------------------
        # Show training progress
        # ----------------------------------------------------

        if episode % 50 == 0:

            recent_rewards = reward_history[-50:]

            average_reward = (
                sum(recent_rewards)
                / len(recent_rewards)
            )

            print(
                f"Episode {episode:3d} | "
                f"Average Reward: "
                f"{average_reward:8.2f} | "
                f"Epsilon: "
                f"{agent.epsilon:.3f}"
            )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Final epsilon       : "
        f"{agent.epsilon:.3f}"
    )

    print(
        f"Learned states      : "
        f"{len(agent.q_table)}"
    )

    print(
        f"Total episodes      : "
        f"{episodes}"
    )

    return agent, env, reward_history


# ============================================================
# TEST TRAINED AGENT
# ============================================================

def test_agent(
    agent,
    people=10
):

    env = create_environment()

    # --------------------------------------------------------
    # Disable exploration
    # --------------------------------------------------------

    agent.epsilon = 0.0

    # --------------------------------------------------------
    # Create test scenario
    # --------------------------------------------------------

    state = env.generate_peak_demand()

    done = False

    total_reward = 0

    print("\n" + "=" * 60)
    print("          TESTING TRAINED RL AGENT")
    print("=" * 60)

    print("Demand: PEAK")

    print("=" * 60)

    # ========================================================
    # TEST LOOP
    # ========================================================

    while not done:

        valid_actions = (
            env.get_valid_actions()
        )

        if not valid_actions:

            print(
                "\nNo safe elevator available."
            )

            break

        # Choose learned best action
        action = agent.choose_action(
            state,
            valid_actions
        )

        # Execute action
        (
            next_state,
            reward,
            done,
            info
        ) = env.step(action)

        # ----------------------------------------------------
        # Display decision
        # ----------------------------------------------------

        print(
            f"\nPassenger P"
            f"{info['passenger_id']}"
        )

        print(
            f"Selected Elevator : "
            f"{info['elevator']}"
        )

        print(
            f"Waiting Time      : "
            f"{info['waiting_time']} floors"
        )

        print(
            f"Travel Distance   : "
            f"{info['travel_distance']} floors"
        )

        print(
            f"Reward            : "
            f"{reward:.2f}"
        )

        print(
            f"Safety            : "
            f"{'SAFE' if info['safe'] else 'UNSAFE'}"
        )

        total_reward += reward

        state = next_state

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("             RL TEST RESULTS")
    print("=" * 60)

    print(
        f"Total Reward       : "
        f"{total_reward:.2f}"
    )

    print(
        f"People Served      : "
        f"{env.total_served}"
    )

    if env.total_served > 0:

        average_waiting = (
            env.total_waiting_time
            / env.total_served
        )

    else:

        average_waiting = 0

    print(
        f"Average Waiting    : "
        f"{average_waiting:.2f} floors"
    )

    print(
        f"Total Distance     : "
        f"{env.total_distance} floors"
    )

    print(
        f"Overload Attempts  : "
        f"{env.overload_attempts}"
    )

    print(
        f"Safety Violations  : "
        f"{env.safety_violations}"
    )

    print("=" * 60)

    return {
        "total_reward": total_reward,
        "people_served": env.total_served,
        "average_waiting": average_waiting,
        "total_distance": env.total_distance,
        "overload_attempts": env.overload_attempts,
        "safety_violations": env.safety_violations
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    agent, env, rewards = train_agent(
        episodes=500,
        people_per_episode=10
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    results = test_agent(
        agent,
        people=10
    )

    print("\nRL PROGRAM FINISHED SUCCESSFULLY.")