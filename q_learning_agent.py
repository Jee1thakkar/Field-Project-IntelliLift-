import os
import pickle
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

        # ----------------------------------------------------
        # Learning parameters
        # ----------------------------------------------------

        self.alpha = learning_rate
        self.gamma = discount_factor

        # Exploration probability
        self.epsilon = epsilon

        # Exploration reduction after every episode
        self.epsilon_decay = epsilon_decay

        # Minimum exploration probability
        self.epsilon_min = epsilon_min

        # ----------------------------------------------------
        # Q-table
        # ----------------------------------------------------
        #
        # State -> Action -> Q-value
        #
        # Example:
        #
        # q_table[state][action] = value
        #
        # ----------------------------------------------------

        self.q_table = defaultdict(
            lambda: defaultdict(float)
        )

    # ========================================================
    # CHOOSE ACTION
    # ========================================================

    def choose_action(self, state, valid_actions):

        # No valid action available
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

        # Randomly select between tied best actions
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

        # Current Q-value
        current_q = self.q_table[state][action]

        # ----------------------------------------------------
        # Calculate future Q-value
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
        #
        # Q(s,a) = Q(s,a) +
        #          alpha * [reward +
        #          gamma * max Q(s',a') - Q(s,a)]
        #
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

    # ========================================================
    # EVALUATION MODE
    # ========================================================

    def set_evaluation_mode(self):

        """
        Disable exploration.

        During testing the agent should use its
        learned Q-values instead of making random choices.
        """

        self.epsilon = 0.0

    # ========================================================
    # NUMBER OF LEARNED STATES
    # ========================================================

    def number_of_states(self):

        """
        Return the number of states stored in the Q-table.
        """

        return len(self.q_table)

    # ========================================================
    # SAVE Q-TABLE
    # ========================================================

    def save(self, file_path):

        """
        Save the trained Q-table to a file.

        The defaultdict contains lambda functions, which should
        not be directly pickled. Therefore the Q-table is first
        converted into normal dictionaries.
        """

        # Create directory if necessary
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        # Convert defaultdict structure to normal dictionaries
        q_table_data = {
            state: dict(action_values)
            for state, action_values in self.q_table.items()
        }

        # Store additional training parameters
        model_data = {
            "q_table": q_table_data,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min
        }

        # Save model
        with open(file_path, "wb") as file:

            pickle.dump(
                model_data,
                file
            )

        print(
            f"Q-table saved successfully: {file_path}"
        )

    # ========================================================
    # LOAD Q-TABLE
    # ========================================================

    def load(self, file_path):

        """
        Load a previously trained Q-table.
        """

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Trained model not found: {file_path}"
            )

        # Load model
        with open(file_path, "rb") as file:

            model_data = pickle.load(file)

        # ----------------------------------------------------
        # Restore Q-table
        # ----------------------------------------------------

        self.q_table = defaultdict(
            lambda: defaultdict(float)
        )

        for state, action_values in model_data["q_table"].items():

            self.q_table[state] = defaultdict(
                float,
                action_values
            )

        # ----------------------------------------------------
        # Restore training parameters
        # ----------------------------------------------------

        self.alpha = model_data.get(
            "alpha",
            self.alpha
        )

        self.gamma = model_data.get(
            "gamma",
            self.gamma
        )

        self.epsilon = model_data.get(
            "epsilon",
            self.epsilon
        )

        self.epsilon_decay = model_data.get(
            "epsilon_decay",
            self.epsilon_decay
        )

        self.epsilon_min = model_data.get(
            "epsilon_min",
            self.epsilon_min
        )

        print(
            f"Q-table loaded successfully: {file_path}"
        )

        print(
            f"Learned states: {self.number_of_states()}"
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

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = create_environment()

    # --------------------------------------------------------
    # Create agent
    # --------------------------------------------------------

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    # Store rewards
    reward_history = []

    print()
    print("=" * 60)
    print("           Q-LEARNING TRAINING")
    print("=" * 60)

    print(
        f"Training episodes : {episodes}"
    )

    print(
        f"People per episode: {people_per_episode}"
    )

    print(
        f"Learning rate     : {agent.alpha}"
    )

    print(
        f"Discount factor   : {agent.gamma}"
    )

    print(
        f"Initial epsilon   : {agent.epsilon}"
    )

    print(
        f"Epsilon decay     : {agent.epsilon_decay}"
    )

    print(
        f"Minimum epsilon   : {agent.epsilon_min}"
    )

    print("=" * 60)

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

        # ----------------------------------------------------
        # Episode loop
        # ----------------------------------------------------

        while not done:

            # Get safe elevators
            valid_actions = env.get_valid_actions()

            # No safe elevator
            if not valid_actions:

                break

            # ------------------------------------------------
            # Select elevator
            # ------------------------------------------------

            action = agent.choose_action(
                state,
                valid_actions
            )

            if action is None:

                break

            # ------------------------------------------------
            # Execute action
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

            # Move to next state
            state = next_state

            # Add reward
            episode_reward += reward

        # ----------------------------------------------------
        # Reduce exploration
        # ----------------------------------------------------

        agent.decay_epsilon()

        # Save reward
        reward_history.append(
            episode_reward
        )

        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        if episode % 50 == 0:

            recent_rewards = reward_history[-50:]

            average_reward = (
                sum(recent_rewards)
                / len(recent_rewards)
            )

            print(
                f"Episode {episode:4d} | "
                f"Average Reward: {average_reward:9.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"States: {agent.number_of_states():6d}"
            )

    # ========================================================
    # TRAINING COMPLETED
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Total episodes : {episodes}"
    )

    print(
        f"Learned states : {agent.number_of_states()}"
    )

    print(
        f"Final epsilon  : {agent.epsilon:.4f}"
    )

    if reward_history:

        final_rewards = reward_history[-50:]

        final_average_reward = (
            sum(final_rewards)
            / len(final_rewards)
        )

        print(
            f"Final 50-episode average reward: "
            f"{final_average_reward:.2f}"
        )

    print("=" * 60)

    return (
        agent,
        env,
        reward_history
    )


# ============================================================
# TEST TRAINED AGENT
# ============================================================

def test_agent(
    agent,
    people=10
):

    # --------------------------------------------------------
    # Create new environment
    # --------------------------------------------------------

    env = create_environment()

    # --------------------------------------------------------
    # Disable exploration
    # --------------------------------------------------------

    agent.set_evaluation_mode()

    # --------------------------------------------------------
    # Create test scenario
    # --------------------------------------------------------

    state = env.reset(
        number_of_people=people
    )

    total_passengers = len(
        env.passengers
    )

    done = False

    total_reward = 0

    print()
    print("=" * 60)
    print("          TESTING TRAINED RL AGENT")
    print("=" * 60)

    print(
        f"Passengers: {total_passengers}"
    )

    print("=" * 60)

    # ========================================================
    # TEST LOOP
    # ========================================================

    while not done:

        # Get safe elevators
        valid_actions = env.get_valid_actions()

        if not valid_actions:

            print()
            print(
                "No safe elevator available."
            )

            break

        # ----------------------------------------------------
        # Select best learned action
        # ----------------------------------------------------

        action = agent.choose_action(
            state,
            valid_actions
        )

        if action is None:

            print()
            print(
                "Agent could not select an action."
            )

            break

        # ----------------------------------------------------
        # Execute action
        # ----------------------------------------------------

        (
            next_state,
            reward,
            done,
            info
        ) = env.step(action)

        # ----------------------------------------------------
        # Display decision
        # ----------------------------------------------------

        print()
        print("-" * 60)

        print(
            f"Passenger ID      : "
            f"P{info['passenger_id']}"
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

        # Move to next state
        state = next_state

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    if env.total_served > 0:

        average_waiting = (
            env.total_waiting_time
            / env.total_served
        )

    else:

        average_waiting = 0

    print()
    print("=" * 60)
    print("             RL TEST RESULTS")
    print("=" * 60)

    print(
        f"Total Reward      : "
        f"{total_reward:.2f}"
    )

    print(
        f"People Generated  : "
        f"{total_passengers}"
    )

    print(
        f"People Served     : "
        f"{env.total_served}"
    )

    print(
        f"Average Waiting   : "
        f"{average_waiting:.2f} floors"
    )

    print(
        f"Total Distance    : "
        f"{env.total_distance} floors"
    )

    print(
        f"Overload Attempts : "
        f"{env.overload_attempts}"
    )

    print(
        f"Safety Violations : "
        f"{env.safety_violations}"
    )

    print(
        f"Learned States    : "
        f"{agent.number_of_states()}"
    )

    print("=" * 60)

    return {
        "total_reward": total_reward,
        "people_generated": total_passengers,
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
    # Train the agent
    # --------------------------------------------------------

    agent, env, rewards = train_agent(
        episodes=500,
        people_per_episode=10
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    model_path = "models/q_table.pkl"

    agent.save(
        model_path
    )

    # --------------------------------------------------------
    # Test trained agent
    # --------------------------------------------------------

    results = test_agent(
        agent,
        people=10
    )

    print()
    print("=" * 60)
    print("          RL PROGRAM FINISHED")
    print("=" * 60)

    print(
        f"Model saved at: {model_path}"
    )

    print("=" * 60)