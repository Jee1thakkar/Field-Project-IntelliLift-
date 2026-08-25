
import random


class Elevator:
    def __init__(self, elevator_id, capacity):
        self.elevator_id = elevator_id
        self.capacity = capacity
        self.current_floor = 0
        self.current_load = 0
        self.passengers = []
        self.total_distance = 0

    def move_to(self, target_floor):
        distance = abs(self.current_floor - target_floor)

        print(
            f"Elevator {self.elevator_id}: "
            f"Floor {self.current_floor} -> Floor {target_floor}"
        )

        self.current_floor = target_floor
        self.total_distance += distance

        return distance

    def can_accept(self, passenger):
        return (
            self.current_load + passenger.weight
            <= self.capacity
        )

    def add_passenger(self, passenger):
        if self.can_accept(passenger):
            self.passengers.append(passenger)
            self.current_load += passenger.weight
            passenger.status = "Inside Elevator"

            return True

        return False

    def remove_passenger(self, passenger):
        if passenger in self.passengers:
            self.passengers.remove(passenger)
            self.current_load -= passenger.weight
            passenger.status = "Completed"

            return True

        return False

    def show_status(self):
        print(
            f"Elevator {self.elevator_id} | "
            f"Floor: {self.current_floor} | "
            f"Load: {self.current_load}/{self.capacity} kg | "
            f"Passengers: {len(self.passengers)} | "
            f"Distance: {self.total_distance} floors"
        )


class Passenger:
    def __init__(self, passenger_id, num_floors):
        self.passenger_id = passenger_id

        # Floor 0 = Ground Floor
        self.start_floor = random.randint(
            0, num_floors - 1
        )

        self.destination_floor = random.randint(
            0, num_floors - 1
        )

        # Start and destination must be different
        while self.destination_floor == self.start_floor:
            self.destination_floor = random.randint(
                0, num_floors - 1
            )

        self.weight = random.randint(40, 100)

        self.waiting_time = 0
        self.travel_time = 0
        self.status = "Waiting"

    def show_request(self):
        print(
            f"Passenger {self.passenger_id}: "
            f"Floor {self.start_floor} -> "
            f"Floor {self.destination_floor} | "
            f"Weight: {self.weight} kg | "
            f"Status: {self.status}"
        )


class Building:
    def __init__(
        self,
        num_floors,
        num_elevators,
        elevator_capacity
    ):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.elevator_capacity = elevator_capacity

        self.elevators = []

        for i in range(num_elevators):
            elevator = Elevator(
                elevator_id=i + 1,
                capacity=elevator_capacity
            )

            self.elevators.append(elevator)

        self.passengers = []
        self.next_passenger_id = 1

    # ------------------------------------------
    # Generate one random passenger
    # ------------------------------------------

    def generate_passenger(self):

        passenger = Passenger(
            passenger_id=self.next_passenger_id,
            num_floors=self.num_floors
        )

        self.passengers.append(passenger)

        self.next_passenger_id += 1

        return passenger

    # ------------------------------------------
    # Generate multiple passengers
    # ------------------------------------------

    def generate_passengers(self, number):

        for _ in range(number):
            self.generate_passenger()

    # ------------------------------------------
    # Find nearest elevator
    # ------------------------------------------

    def select_elevator(self, passenger):

        selected_elevator = None
        shortest_distance = float("inf")

        for elevator in self.elevators:

            distance = abs(
                elevator.current_floor
                - passenger.start_floor
            )

            # Check capacity
            if not elevator.can_accept(passenger):
                continue

            if distance < shortest_distance:
                shortest_distance = distance
                selected_elevator = elevator

        return selected_elevator

    # ------------------------------------------
    # Serve one passenger
    # ------------------------------------------

    def serve_passenger(self, passenger):

        if passenger.status != "Waiting":
            return

        print("\n" + "-" * 50)

        print(
            f"Passenger {passenger.passenger_id} "
            f"is waiting at Floor "
            f"{passenger.start_floor}"
        )

        print(
            f"Destination: Floor "
            f"{passenger.destination_floor}"
        )

        print(
            f"Weight: {passenger.weight} kg"
        )

        # Find elevator
        elevator = self.select_elevator(passenger)

        if elevator is None:

            print(
                "No available elevator for this passenger."
            )

            return

        print(
            f"Selected Elevator "
            f"{elevator.elevator_id}"
        )

        # --------------------------------------
        # Elevator goes to passenger
        # --------------------------------------

        waiting_distance = elevator.move_to(
            passenger.start_floor
        )

        passenger.waiting_time = waiting_distance

        # --------------------------------------
        # Passenger enters
        # --------------------------------------

        if elevator.add_passenger(passenger):

            print(
                f"Passenger {passenger.passenger_id} "
                f"entered Elevator "
                f"{elevator.elevator_id}"
            )

        else:

            print("Passenger could not enter.")
            return

        # --------------------------------------
        # Elevator goes to destination
        # --------------------------------------

        travel_distance = elevator.move_to(
            passenger.destination_floor
        )

        passenger.travel_time = travel_distance

        # --------------------------------------
        # Passenger exits
        # --------------------------------------

        elevator.remove_passenger(passenger)

        print(
            f"Passenger {passenger.passenger_id} "
            f"reached Floor "
            f"{passenger.destination_floor}"
        )

        print(
            f"Passenger {passenger.passenger_id} "
            f"exited Elevator "
            f"{elevator.elevator_id}"
        )

        print(
            f"Waiting time: "
            f"{passenger.waiting_time} floors"
        )

        print(
            f"Travel distance: "
            f"{passenger.travel_time} floors"
        )

    # ------------------------------------------
    # Run complete simulation
    # ------------------------------------------

    def run_simulation(self):

        print("\n")
        print("=" * 60)
        print("        INTELLILIFT SIMULATION")
        print("=" * 60)

        for passenger in self.passengers:

            self.serve_passenger(passenger)

    # ------------------------------------------
    # Show complete status
    # ------------------------------------------

    def show_status(self):

        print("\n" + "=" * 60)
        print("BUILDING STATUS")
        print("=" * 60)

        print(f"Floors: {self.num_floors}")
        print(f"Elevators: {self.num_elevators}")
        print(
            f"Elevator Capacity: "
            f"{self.elevator_capacity} kg"
        )

        print("\n--- Elevators ---")

        for elevator in self.elevators:
            elevator.show_status()

        print("\n--- Passengers ---")

        if not self.passengers:
            print("No passengers.")

        else:
            for passenger in self.passengers:
                passenger.show_request()


# ============================================================
# USER INPUT
# ============================================================

print("=" * 60)
print("                    IntelliLift")
print("=" * 60)

while True:

    try:
        num_floors = int(
            input("Enter number of floors: ")
        )

        if num_floors >= 2:
            break

        print("Enter at least 2 floors.")

    except ValueError:
        print("Please enter a valid number.")


while True:

    try:
        num_elevators = int(
            input("Enter number of elevators: ")
        )

        if num_elevators >= 1:
            break

        print("Enter at least 1 elevator.")

    except ValueError:
        print("Please enter a valid number.")


while True:

    try:
        elevator_capacity = float(
            input("Enter elevator capacity (kg): ")
        )

        if elevator_capacity > 0:
            break

        print("Capacity must be greater than 0.")

    except ValueError:
        print("Please enter a valid number.")


# ============================================================
# CREATE BUILDING
# ============================================================

building = Building(
    num_floors=num_floors,
    num_elevators=num_elevators,
    elevator_capacity=elevator_capacity
)


# ============================================================
# AUTOMATICALLY GENERATE PASSENGERS
# ============================================================

# Temporary value for testing.
# Later this will be controlled by simulation time
# and peak-demand conditions.

building.generate_passengers(5)


# ============================================================
# SHOW INITIAL STATE
# ============================================================

building.show_status()


# ============================================================
# RUN SIMULATION
# ============================================================

building.run_simulation()


# ============================================================
# SHOW FINAL STATE
# ============================================================

print("\n")
print("=" * 60)
print("        FINAL INTELLILIFT STATUS")
print("=" * 60)

building.show_status()
