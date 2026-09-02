import random
import pickle
import os


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
        # Q-LEARNING PARAMETERS
        # ----------------------------------------------------

        # Learning rate
        # Controls how quickly the agent learns
        self.alpha = learning_rate

        # Discount factor
        # Controls importance of future rewards
        self.gamma = discount_factor

        # ----------------------------------------------------
        # EXPLORATION PARAMETERS
        # ----------------------------------------------------

        # Probability of selecting a random action
        self.epsilon = epsilon

        # Epsilon reduction after each episode
        self.epsilon_decay = epsilon_decay

        # Minimum exploration probability
        self.epsilon_min = epsilon_min

        # ----------------------------------------------------
        # Q-TABLE
        # ----------------------------------------------------

        # Format:
        #
        # {
        #     state: {
        #         action: q_value
        #     }
        # }
        #
        # Example:
        #
        # {
        #     (1, 0, 500, 0, ...): {
        #          0: 5.2,
        #          1: 8.7,
        #          2: 3.1
        #     }
        # }

        self.q_table = {}

    # ========================================================
    # GET Q VALUE
    # ========================================================

    def get_q_value(self, state, action):

        # Create state if it doesn't exist
        if state not in self.q_table:
            self.q_table[state] = {}

        # Create action if it doesn't exist
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        return self.q_table[state][action]

    # ========================================================
    # CHOOSE ACTION
    # ========================================================

    def choose_action(self, state, valid_actions):

        # ----------------------------------------------------
        # No valid/safe action
        # ----------------------------------------------------

        if not valid_actions:
            return None

        # ----------------------------------------------------
        # EXPLORATION
        # ----------------------------------------------------
        #
        # Select a random SAFE elevator.
        #
        # This is important because the environment provides
        # only safe/valid actions.
        # ----------------------------------------------------

        if random.random() < self.epsilon:

            return random.choice(valid_actions)

        # ----------------------------------------------------
        # EXPLOITATION
        # ----------------------------------------------------
        #
        # Select the elevator with the highest learned
        # Q-value.
        # ----------------------------------------------------

        q_values = []

        for action in valid_actions:

            q_value = self.get_q_value(
                state,
                action
            )

            q_values.append(q_value)

        # Find highest Q-value
        max_q_value = max(q_values)

        # ----------------------------------------------------
        # Handle ties
        # ----------------------------------------------------
        #
        # If multiple elevators have the same Q-value,
        # randomly select one of them.
        # ----------------------------------------------------

        best_actions = []

        for action, q_value in zip(
            valid_actions,
            q_values
        ):

            if q_value == max_q_value:

                best_actions.append(action)

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

        # ----------------------------------------------------
        # Get current Q-value
        # ----------------------------------------------------

        current_q = self.get_q_value(
            state,
            action
        )

        # ----------------------------------------------------
        # TERMINAL STATE
        # ----------------------------------------------------

        if done:

            future_q = 0.0

        # ----------------------------------------------------
        # No valid action in next state
        # ----------------------------------------------------

        elif not next_valid_actions:

            future_q = 0.0

        # ----------------------------------------------------
        # NON-TERMINAL STATE
        # ----------------------------------------------------

        else:

            next_q_values = []

            for next_action in next_valid_actions:

                next_q = self.get_q_value(
                    next_state,
                    next_action
                )

                next_q_values.append(next_q)

            # Best future Q-value
            future_q = max(next_q_values)

        # ----------------------------------------------------
        # Q-LEARNING TARGET
        # ----------------------------------------------------

        target = (
            reward
            + self.gamma * future_q
        )

        # ----------------------------------------------------
        # Q-LEARNING UPDATE FORMULA
        # ----------------------------------------------------
        #
        # Q(s,a) ← Q(s,a) +
        #          α [target - Q(s,a)]
        #
        # ----------------------------------------------------

        updated_q = (
            current_q
            + self.alpha
            * (target - current_q)
        )

        self.q_table[state][action] = updated_q

    # ========================================================
    # EPSILON DECAY
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # ========================================================
    # EVALUATION MODE
    # ========================================================
    #
    # During testing we do NOT want random exploration.
    #
    # epsilon = 0
    #
    # means the agent always selects the best learned action.
    # ========================================================

    def set_evaluation_mode(self):

        self.epsilon = 0.0

    # ========================================================
    # SAVE TRAINED Q-TABLE
    # ========================================================

    def save(self, filename="q_table.pkl"):

        # ----------------------------------------------------
        # Data to save
        # ----------------------------------------------------

        data = {

            "q_table":
                self.q_table,

            "epsilon":
                self.epsilon,

            "learning_rate":
                self.alpha,

            "discount_factor":
                self.gamma,

            "epsilon_decay":
                self.epsilon_decay,

            "epsilon_min":
                self.epsilon_min
        }

        # ----------------------------------------------------
        # Create directory if necessary
        # ----------------------------------------------------

        directory = os.path.dirname(filename)

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        # ----------------------------------------------------
        # Save using pickle
        # ----------------------------------------------------

        with open(
            filename,
            "wb"
        ) as file:

            pickle.dump(
                data,
                file
            )

        # ----------------------------------------------------
        # Confirmation
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("              Q-TABLE SAVED")
        print("=" * 60)

        print(
            f"File           : {filename}"
        )

        print(
            f"Learned States : "
            f"{len(self.q_table)}"
        )

        print(
            f"Epsilon        : "
            f"{self.epsilon:.4f}"
        )

        print("=" * 60)

    # ========================================================
    # LOAD TRAINED Q-TABLE
    # ========================================================

    def load(self, filename="q_table.pkl"):

        # ----------------------------------------------------
        # Check whether model exists
        # ----------------------------------------------------

        if not os.path.exists(filename):

            raise FileNotFoundError(
                "\nTrained Q-table was not found.\n"
                f"Expected file: {filename}\n\n"
                "Run train_rl.py first to create "
                "the trained model."
            )

        # ----------------------------------------------------
        # Load saved data
        # ----------------------------------------------------

        with open(
            filename,
            "rb"
        ) as file:

            data = pickle.load(file)

        # ----------------------------------------------------
        # Validate saved data
        # ----------------------------------------------------

        if not isinstance(data, dict):

            raise ValueError(
                "Invalid Q-table file."
            )

        if "q_table" not in data:

            raise ValueError(
                "The saved file does not contain "
                "a Q-table."
            )

        # ----------------------------------------------------
        # Restore Q-table
        # ----------------------------------------------------

        self.q_table = data["q_table"]

        # ----------------------------------------------------
        # Restore training parameters if available
        # ----------------------------------------------------

        if "learning_rate" in data:

            self.alpha = data[
                "learning_rate"
            ]

        if "discount_factor" in data:

            self.gamma = data[
                "discount_factor"
            ]

        if "epsilon_decay" in data:

            self.epsilon_decay = data[
                "epsilon_decay"
            ]

        if "epsilon_min" in data:

            self.epsilon_min = data[
                "epsilon_min"
            ]

        if "epsilon" in data:

            self.epsilon = data[
                "epsilon"
            ]

        # ----------------------------------------------------
        # Confirmation
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("              Q-TABLE LOADED")
        print("=" * 60)

        print(
            f"File           : {filename}"
        )

        print(
            f"Learned States : "
            f"{len(self.q_table)}"
        )

        print(
            f"Epsilon        : "
            f"{self.epsilon:.4f}"
        )

        print("=" * 60)

    # ========================================================
    # NUMBER OF LEARNED STATES
    # ========================================================

    def number_of_states(self):

        return len(self.q_table)
