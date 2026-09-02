import os
from elevator_environment import ElevatorEnvironment
from q_learning_agent import QLearningAgent


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "q_table.pkl"
)


# ============================================================
# LOAD TRAINED AGENT
# ============================================================

def load_trained_agent():

    print("=" * 70)
    print("                 LOADING TRAINED MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not os.path.exists(MODEL_FILE):

        print()
        print("ERROR: Trained Q-table was not found.")
        print()
        print(f"Expected file:")
        print(MODEL_FILE)
        print()
        print("First run:")
        print("py train_rl.py")
        print()

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    # --------------------------------------------------------
    # Create agent
    # --------------------------------------------------------

    agent = QLearningAgent()

    # --------------------------------------------------------
    # Load trained Q-table
    # --------------------------------------------------------

    agent.load(MODEL_FILE)

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------
    # No random exploration during testing.

    agent.set_evaluation_mode()

    print()
    print("Evaluation mode enabled.")
    print("Epsilon:", agent.epsilon)
    print()

    return agent


# ============================================================
# TEST TRAINED AGENT
# ============================================================

def test_agent(agent, number_of_people=15):

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = ElevatorEnvironment(
        floors=10,
        number_of_elevators=3,
        capacities=[500, 500, 500]
    )

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    state = env.reset(
        number_of_people=number_of_people
    )

    done = False

    total_reward = 0.0
    total_waiting_time = 0.0
    total_distance = 0.0
    total_served = 0

    safety_violations = 0
    overload_attempts = 0

    step_number = 0

    # ========================================================
    # TEST EPISODE
    # ========================================================

    print("=" * 70)
    print("                 TESTING RL AGENT")
    print("=" * 70)

    print(f"Passengers: {number_of_people}")
    print()

    while not done:

        step_number += 1

        # ----------------------------------------------------
        # Get safe actions
        # ----------------------------------------------------

        valid_actions = env.get_valid_actions()

        if not valid_actions:

            print("No safe action available.")
            break

        # ----------------------------------------------------
        # Select best learned action
        # ----------------------------------------------------

        action = agent.choose_action(
            state,
            valid_actions
        )

        if action is None:
            print("No valid action selected.")
            break

        # ----------------------------------------------------
        # Perform action
        # ----------------------------------------------------

        next_state, reward, done, info = env.step(action)

        # ----------------------------------------------------
        # Collect metrics
        # ----------------------------------------------------

        total_reward += reward

        # These keys are expected from the environment.
        waiting_time = info.get(
            "waiting_time",
            0
        )

        travel_distance = info.get(
            "travel_distance",
            0
        )

        safety_violation = info.get(
            "safety_violation",
            False
        )

        overload = info.get(
            "overload_attempt",
            False
        )

        passenger_served = info.get(
            "passenger_served",
            True
        )

        total_waiting_time += waiting_time
        total_distance += travel_distance

        if passenger_served:
            total_served += 1

        if safety_violation:
            safety_violations += 1

        if overload:
            overload_attempts += 1

        # ----------------------------------------------------
        # Display step
        # ----------------------------------------------------

        elevator_number = action + 1

        print(
            f"Step {step_number:3d} | "
            f"Elevator {elevator_number} | "
            f"Wait: {waiting_time:5.1f} | "
            f"Distance: {travel_distance:5.1f} | "
            f"Reward: {reward:7.2f}"
        )

        # ----------------------------------------------------
        # Move to next state
        # ----------------------------------------------------

        state = next_state

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("                    TEST RESULTS")
    print("=" * 70)

    print(
        f"Passengers requested : {number_of_people}"
    )

    print(
        f"Passengers served    : {total_served}"
    )

    print(
        f"Total waiting time   : {total_waiting_time:.2f}"
    )

    print(
        f"Total distance       : {total_distance:.2f}"
    )

    print(
        f"Total reward         : {total_reward:.2f}"
    )

    print(
        f"Safety violations    : {safety_violations}"
    )

    print(
        f"Overload attempts    : {overload_attempts}"
    )

    print(
        f"Steps                : {step_number}"
    )

    # --------------------------------------------------------
    # Average waiting time
    # --------------------------------------------------------

    if total_served > 0:

        average_waiting = (
            total_waiting_time /
            total_served
        )

    else:

        average_waiting = 0.0

    print(
        f"Average waiting time : {average_waiting:.2f}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("             SAFE ELEVATOR RL TEST")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    agent = load_trained_agent()

    # --------------------------------------------------------
    # Test the trained model
    # --------------------------------------------------------

    test_agent(
        agent,
        number_of_people=15
    )

    print()
    print("Testing completed.")