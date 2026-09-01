import random


# ============================================================
# PASSENGER CLASS
# ============================================================

class Passenger:

    def __init__(self, passenger_id, start, destination, weight):
        self.id = passenger_id
        self.start = start
        self.destination = destination
        self.weight = weight
        self.status = "Waiting"
        self.waiting_time = 0

    def get_state(self):
        return {
            "id": self.id,
            "start": self.start,
            "destination": self.destination,
            "weight": self.weight,
            "status": self.status,
            "waiting_time": self.waiting_time
        }


# ============================================================
# ELEVATOR CLASS
# ============================================================

class Elevator:

    def __init__(self, elevator_id, capacity):

        self.id = elevator_id
        self.name = f"Elevator {elevator_id}"

        self.floor = 1
        self.capacity = capacity
        self.load = 0
        self.passengers = []

        self.distance = 0

    # --------------------------------------------------------
    # Check whether passenger can enter
    # --------------------------------------------------------

    def can_accept(self, passenger):

        return (
            self.load + passenger.weight
            <= self.capacity
        )

    # --------------------------------------------------------
    # Distance from elevator to passenger
    # --------------------------------------------------------

    def distance_to(self, floor):

        return abs(self.floor - floor)

    # --------------------------------------------------------
    # Add passenger
    # --------------------------------------------------------

    def add_passenger(self, passenger):

        if not self.can_accept(passenger):
            return False

        self.load += passenger.weight
        self.passengers.append(passenger.id)

        return True

    # --------------------------------------------------------
    # Remove passenger
    # --------------------------------------------------------

    def remove_passenger(self, passenger):

        if passenger.id in self.passengers:

            self.passengers.remove(passenger.id)
            self.load -= passenger.weight

            return True

        return False

    # --------------------------------------------------------
    # Move elevator
    # --------------------------------------------------------

    def move_to(self, target_floor):

        distance = abs(
            self.floor - target_floor
        )

        self.floor = target_floor
        self.distance += distance

        return distance

    # --------------------------------------------------------
    # Reset elevator
    # --------------------------------------------------------

    def reset(self):

        self.floor = 1
        self.load = 0
        self.passengers = []
        self.distance = 0

    # --------------------------------------------------------
    # Get elevator state
    # --------------------------------------------------------

    def get_state(self):

        return [
            self.floor,
            self.load,
            self.capacity,
            len(self.passengers)
        ]


# ============================================================
# BUILDING CLASS
# ============================================================

class Building:

    def __init__(self, floors):

        if floors < 2:
            raise ValueError(
                "Building must have at least 2 floors."
            )

        self.floors = floors

    # --------------------------------------------------------
    # Validate floor
    # --------------------------------------------------------

    def valid_floor(self, floor):

        return 1 <= floor <= self.floors

    # --------------------------------------------------------
    # Generate random passenger
    # --------------------------------------------------------

    def create_random_passenger(self, passenger_id):

        start = random.randint(1, self.floors)

        destination = random.randint(1, self.floors)

        while destination == start:

            destination = random.randint(
                1,
                self.floors
            )

        weight = random.randint(40, 100)

        return Passenger(
            passenger_id,
            start,
            destination,
            weight
        )


# ============================================================
# ELEVATOR ENVIRONMENT
# ============================================================

class ElevatorEnvironment:

    """
    RL-ready virtual elevator environment.

    The RL agent selects an elevator.
    The environment checks safety,
    moves the selected elevator,
    serves the passenger,
    calculates reward,
    and returns the next state.
    """

    def __init__(
        self,
        floors=10,
        number_of_elevators=3,
        capacities=None
    ):

        # ----------------------------------------------------
        # Building
        # ----------------------------------------------------

        self.building = Building(floors)

        # ----------------------------------------------------
        # Elevator capacities
        # ----------------------------------------------------

        if capacities is None:

            capacities = [500] * number_of_elevators

        if len(capacities) != number_of_elevators:

            raise ValueError(
                "Number of capacities must match "
                "number of elevators."
            )

        # ----------------------------------------------------
        # Create elevators
        # ----------------------------------------------------

        self.elevators = []

        for i in range(number_of_elevators):

            elevator = Elevator(
                elevator_id=i + 1,
                capacity=capacities[i]
            )

            self.elevators.append(elevator)

        # ----------------------------------------------------
        # Passenger information
        # ----------------------------------------------------

        self.passengers = []
        self.current_passenger = None

        # ----------------------------------------------------
        # Simulation statistics
        # ----------------------------------------------------

        self.total_served = 0
        self.total_waiting_time = 0
        self.total_distance = 0
        self.overload_attempts = 0
        self.safety_violations = 0

        self.step_count = 0

    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    def reset(self, number_of_people=10):

        # Reset elevators

        for elevator in self.elevators:
            elevator.reset()

        # Reset statistics

        self.total_served = 0
        self.total_waiting_time = 0
        self.total_distance = 0
        self.overload_attempts = 0
        self.safety_violations = 0
        self.step_count = 0

        # Create passengers

        self.passengers = []

        for i in range(1, number_of_people + 1):

            passenger = (
                self.building.create_random_passenger(i)
            )

            self.passengers.append(passenger)

        self.current_passenger = None

        # Get initial state

        return self.get_state()

    # ========================================================
    # CREATE PEAK DEMAND
    # ========================================================

    def generate_peak_demand(self):

        number_of_people = random.randint(11, 20)

        return self.reset(number_of_people)

    # ========================================================
    # GET WAITING PASSENGER
    # ========================================================

    def get_next_passenger(self):

        for passenger in self.passengers:

            if passenger.status == "Waiting":

                return passenger

        return None

    # ========================================================
    # GET STATE
    # ========================================================

    def get_state(self):

        state = []

        # Elevator information

        for elevator in self.elevators:

            state.extend(
                elevator.get_state()
            )

        # Current passenger

        passenger = self.get_next_passenger()

        if passenger is not None:

            state.extend([
                passenger.start,
                passenger.destination,
                passenger.weight
            ])

        else:

            state.extend([
                0,
                0,
                0
            ])

        return tuple(state)

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    def safety_check(
        self,
        elevator,
        passenger
    ):

        # Check passenger floor

        if not self.building.valid_floor(
            passenger.start
        ):

            return False

        # Check destination

        if not self.building.valid_floor(
            passenger.destination
        ):

            return False

        # Check elevator capacity

        if not elevator.can_accept(
            passenger
        ):

            self.overload_attempts += 1

            return False

        return True

    # ========================================================
    # GET VALID ACTIONS
    # ========================================================

    def get_valid_actions(self):

        passenger = self.get_next_passenger()

        if passenger is None:
            return []

        valid_actions = []

        for index, elevator in enumerate(
            self.elevators
        ):

            if self.safety_check(
                elevator,
                passenger
            ):

                valid_actions.append(index)

        return valid_actions

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        passenger = self.get_next_passenger()

        # ----------------------------------------------------
        # No passenger remaining
        # ----------------------------------------------------

        if passenger is None:

            return (
                self.get_state(),
                0,
                True,
                {
                    "message":
                    "All passengers served."
                }
            )

        # ----------------------------------------------------
        # Validate action
        # ----------------------------------------------------

        if not isinstance(action, int):

            raise ValueError(
                "Action must be an integer."
            )

        if action < 0 or action >= len(
            self.elevators
        ):

            self.safety_violations += 1

            return (
                self.get_state(),
                -100,
                False,
                {
                    "error":
                    "Invalid elevator action."
                }
            )

        elevator = self.elevators[action]

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not self.safety_check(
            elevator,
            passenger
        ):

            self.safety_violations += 1

            return (
                self.get_state(),
                -100,
                False,
                {
                    "error":
                    "Unsafe elevator action."
                }
            )

        # ----------------------------------------------------
        # Move to passenger
        # ----------------------------------------------------

        distance_to_passenger = (
            elevator.move_to(
                passenger.start
            )
        )

        # Waiting time is based on distance

        waiting_time = distance_to_passenger

        passenger.waiting_time = waiting_time

        self.total_waiting_time += waiting_time

        # ----------------------------------------------------
        # Passenger enters
        # ----------------------------------------------------

        passenger.status = "Inside Elevator"

        elevator.add_passenger(
            passenger
        )

        # ----------------------------------------------------
        # Move to destination
        # ----------------------------------------------------

        travel_distance = (
            elevator.move_to(
                passenger.destination
            )
        )

        # ----------------------------------------------------
        # Passenger exits
        # ----------------------------------------------------

        elevator.remove_passenger(
            passenger
        )

        passenger.status = "Completed"

        self.total_served += 1

        total_distance_for_passenger = (
            distance_to_passenger
            + travel_distance
        )

        self.total_distance += (
            total_distance_for_passenger
        )

        self.step_count += 1

        # ====================================================
        # REWARD
        # ====================================================

        # Smaller distance = better
        # Smaller waiting time = better

        reward = 0

        reward -= waiting_time

        reward -= travel_distance * 0.5

        # Successful service reward

        reward += 10

        # Safety bonus

        reward += 5

        # ----------------------------------------------------
        # Check completion
        # ----------------------------------------------------

        done = (
            self.get_next_passenger()
            is None
        )

        info = {

            "passenger_id":
                passenger.id,

            "elevator":
                elevator.name,

            "waiting_time":
                waiting_time,

            "travel_distance":
                travel_distance,

            "total_distance":
                total_distance_for_passenger,

            "reward":
                reward,

            "safe":
                True
        }

        return (
            self.get_state(),
            reward,
            done,
            info
        )

    # ========================================================
    # RUN RANDOM POLICY
    # ========================================================

    def run_random_policy(self):

        state = self.reset()

        total_reward = 0

        done = False

        while not done:

            valid_actions = (
                self.get_valid_actions()
            )

            if not valid_actions:

                break

            action = random.choice(
                valid_actions
            )

            (
                state,
                reward,
                done,
                info
            ) = self.step(action)

            total_reward += reward

        return {
            "total_reward":
                total_reward,

            "people_served":
                self.total_served,

            "average_waiting_time":
                (
                    self.total_waiting_time
                    / self.total_served
                    if self.total_served > 0
                    else 0
                ),

            "total_distance":
                self.total_distance,

            "overload_attempts":
                self.overload_attempts,

            "safety_violations":
                self.safety_violations
        }

    # ========================================================
    # SHOW RESULTS
    # ========================================================

    def show_results(self):

        print("\n" + "=" * 60)
        print("ELEVATOR ENVIRONMENT RESULTS")
        print("=" * 60)

        print(
            f"People served       : "
            f"{self.total_served}"
        )

        if self.total_served > 0:

            average_wait = (
                self.total_waiting_time
                / self.total_served
            )

        else:

            average_wait = 0

        print(
            f"Average waiting time: "
            f"{average_wait:.2f}"
        )

        print(
            f"Total distance      : "
            f"{self.total_distance}"
        )

        print(
            f"Overload attempts   : "
            f"{self.overload_attempts}"
        )

        print(
            f"Safety violations   : "
            f"{self.safety_violations}"
        )