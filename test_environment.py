from elevator_environment import ElevatorEnvironment


env = ElevatorEnvironment(
    floors=10,
    number_of_elevators=3,
    capacities=[500, 500, 500]
)

print("=" * 60)
print("FULL RL ENVIRONMENT TEST")
print("=" * 60)

state = env.generate_peak_demand()

print("\nInitial State:")
print(state)

done = False
total_reward = 0

while not done:

    valid_actions = env.get_valid_actions()

    if not valid_actions:
        print("\nNo safe elevator available.")
        break

    # Temporary random policy for testing
    action = valid_actions[0]

    state, reward, done, info = env.step(action)

    total_reward += reward

    print("\nPassenger served:")
    print(info)

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)

print("Total Reward:", total_reward)
print("People Served:", env.total_served)
print("Total Distance:", env.total_distance)
print("Safety Violations:", env.safety_violations)
print("Overload Attempts:", env.overload_attempts)

env.show_results()