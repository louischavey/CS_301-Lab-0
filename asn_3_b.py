import time
# import Board
from sonar import *
# import Mpu6050
import ros_robot_controller_sdk as rrc
import threading
import csv
import ast
import json

print('''
**********************************************************
********CS/ME 301 Assignment Template*******
**********************************************************
----------------------------------------------------------
Usage:
    sudo python3 asn_template.py
----------------------------------------------------------
Tips:
 * Press Ctrl+C to close the program. If it fails,
      please try multiple times！
----------------------------------------------------------
''')

board = rrc.Board()

print('''Assignment 3 for Group B''')
time.sleep(1) 

#mpu = Mpu6050.mpu6050()
#mpu.set_gyro_range(mpu.GYRO_RANGE_2000DEG)
#mpu.set_accel_range(mpu.ACCEL_RANGE_2G) 


MAX_MOVEMENT = 45  # demo speed + acc: 100

LEG_NEUTRAL = 500

LEG_FORWARD = 500 - MAX_MOVEMENT
LEG_BACK = 500 + MAX_MOVEMENT

latTime = 500

UP_LEFT = 400
UP_RIGHT = 600

vertTime = 500

LEG_IN = 400
LEG_OUT = 600

extendTime = 500

oddLegs = [1, 7, 13]
evenLegs = [4, 10, 16]

evenVerts = [5, 11, 17]
oddVerts = [2, 8, 14]

runTimeForwardBack = 0.4
runTimeUpDown = 0.2
runTimeSonar = 0.05

WALKS_PER_GRID = 7

TURN_ADJUST = 3
OFFSET_ADJUST = 5

ADJUST_THRESHOLD = 665 - LEG_BACK  # 665 is roughly max we can move leg before it touches other one

jointIDs = list(range(1, 19))

heading = 1

directions = {(-1, 0): 3, (1, 0): 1, (0, -1): 4, (0, 1): 2}

SCAN_SLEEP = 0.5
WALL_THRESHOLD = 400
sonar = Sonar()

def legsUp(legID):
    for id in legID:
        if id < 10:
            board.bus_servo_set_position(runTimeUpDown, [[id, UP_LEFT]])
        else:
            board.bus_servo_set_position(runTimeUpDown, [[id, UP_RIGHT]])
    return

def legsDown(legID):
    for id in legID:
            board.bus_servo_set_position(runTimeUpDown, [[id, LEG_NEUTRAL]])
    return


def legsForward(legID, adjust=0):
    if adjust > ADJUST_THRESHOLD:
        adjust = ADJUST_THRESHOLD
    elif adjust < -ADJUST_THRESHOLD:
        adjust = -ADJUST_THRESHOLD
    for id in legID:
        if adjust > 0:
            if id < 10:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD - adjust]])
            else:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK + OFFSET_ADJUST + adjust]])
                #board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK]])
        elif adjust < 0:
            if id < 10:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD - adjust]])
            else:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK + OFFSET_ADJUST + adjust]])
                #board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK - adjust]])

        else:
            if id < 10:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD]])
            else:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK + OFFSET_ADJUST]])
                #board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK]])

    return


def legsBack(legID, adjust=0):
    if adjust > ADJUST_THRESHOLD:
        adjust = ADJUST_THRESHOLD
    elif adjust < -ADJUST_THRESHOLD:
        adjust = -ADJUST_THRESHOLD
    for id in legID:
        if adjust > 0:
            if id < 10:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK + adjust]])
            else:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD + OFFSET_ADJUST - adjust]])
        elif adjust < 0:
            if id < 10:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK + adjust]])
            else:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD + OFFSET_ADJUST - adjust]])
        else:
            if id < 10:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK]])
            else:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD]])
        
    return

def walkForward(adjust=0):
    #Gets the first set of legs ready
    legsUp(evenVerts)
    legsForward(evenLegs, adjust)

    time.sleep(runTimeUpDown)

    legsDown(evenVerts)

    time.sleep(runTimeUpDown)

    legsUp(oddVerts)
    legsForward(oddLegs, adjust)
    time.sleep(runTimeForwardBack)
    legsBack(evenLegs)

    time.sleep(runTimeUpDown)

    legsDown(oddVerts)

    time.sleep(runTimeUpDown)

    legsUp(evenVerts)
    legsBack(oddLegs)
    time.sleep(runTimeUpDown)
    return

def wallFollowLeft(s):
    dis = s.getDistance()
    error = 350 - dis

    P = error * kp
    #print(f"Distance {dis} has P of {P}")
    print(dis)
    walkForward(int(P))

    return 

def wallFollowRight(s):
    dis = s.getDistance()
    error = 350 - dis

    P = error * kp
    print(f"Distance {dis} has P of {P}")
    print(dis)
    walkForward(-int(P))

    return 


def walkBack():
    legsUp(evenVerts)
    legsBack(evenLegs)

    time.sleep(runTimeUpDown)

    legsDown(evenVerts)

    time.sleep(runTimeUpDown)

    legsForward(evenLegs)
    legsUp(oddVerts)
    time.sleep(runTimeUpDown)
    legsBack(oddLegs)

    time.sleep(runTimeUpDown)

    legsDown(oddVerts)

    time.sleep(runTimeUpDown)

    legsUp(evenVerts)
    legsForward(oddLegs)
    time.sleep(runTimeUpDown)
    return

def turnRightNew(adjust=0):
    legsDown(evenVerts)

    
    time.sleep(runTimeUpDown)
    legsUp(oddVerts)

   
    time.sleep(runTimeUpDown)

    legsForward([1,7,10,16], adjust)
    legsBack([13, 4], adjust)

    time.sleep(runTimeUpDown)

    legsDown(oddVerts)
    time.sleep(runTimeUpDown)
    legsUp(evenVerts)

    time.sleep(runTimeUpDown)
    #legsBack([1, 7, 4])
    #legsForward([13, 10, 16])

    legsBack([1, 7, 10, 16], adjust)
    legsForward([13, 4], adjust)
    time.sleep(runTimeUpDown)
    
    return

def turnLeftNew(adjust=0):
    legsDown(evenVerts)

    time.sleep(runTimeUpDown)

    legsUp(oddVerts)
    
    time.sleep(runTimeUpDown)

    legsBack([1,7,10,16], adjust)
    legsForward([13, 4], adjust)

    time.sleep(runTimeUpDown)

    legsDown(oddVerts)
    time.sleep(runTimeUpDown)
    legsUp(evenVerts)

    time.sleep(runTimeUpDown)
    #legsBack([1, 7, 4])
    #legsForward([13, 10, 16])

    legsForward([1, 7, 10, 16], adjust)
    legsBack([13, 4], adjust)

    time.sleep(runTimeUpDown)

    return

def turn(degrees, adjust):  # clockwise means positive, counter-clockwise means negative degrees
    if degrees < 0:
        for _ in range(int(-degrees//22.5)):
            turnLeftNew(adjust)  # need to calculate exactly how many degrees each call is
    elif degrees > 0:
        for _ in range(int(degrees//22.5)):
            turnRightNew(adjust)

def abs_turn(start_heading, end_heading):
    #end_heading-start_heading
    #if abs(
    turn(min(end_heading-start_heading, 360 - abs(end_heading-start_heading)))

def forward(seconds):
    startForward = time.time()
    while time.time() - startForward < seconds:
        walkForward()

def sonarRight():
    board.bus_servo_set_position(runTimeSonar, [[21, 110]])
    #print("sonar right")
    return

def sonarLeft():
    board.bus_servo_set_position(runTimeSonar, [[21, 890]])
    #print("sonar left")
    return

def sonarForward():
    board.bus_servo_set_position(runTimeSonar, [[21, 500]])
    #print("sonar fwd")
    return

def initRobot():
    for id in jointIDs:
        if id in [3, 6, 9, 12, 15, 18]:
            if id > 10:
                board.bus_servo_set_position(runTimeUpDown, [[id, 650]])
            else:
                board.bus_servo_set_position(runTimeUpDown, [[id, 350]])
        else:
            board.bus_servo_set_position(runTimeUpDown, [[id, 500]])

    sonarForward()
                
    return




# Initialize sensors
s = Sonar()
#initRobot()
# Get trial number and terrain type from user
trial_num = input("Enter trial number: ")
#terrain_type = input("Enter terrain type: ")

# CSV file name
csv_filename = "DemoData.csv"

# Shared storage for collected data
velocity_readings = []
imu_readings = []
stop_event = threading.Event()  # Event to signal when recording is done
total_time_elapsed = 0  # Variable to store total elapsed time

# Walk gait function (runs indefinitely until stopped)
def walk_gait():
    while not stop_event.is_set():
        walkForward()

# Record velocity and IMU data over time
def record_motion_data():
    global total_time_elapsed  # Access the global variable
    start_time = time.time()
    
    while True:
        # Get velocity from sonar
        velocity = s.getDistance()
        velocity_readings.append(velocity)

        # Get IMU data
        imu_data = board.get_imu()
        if imu_data:
            print("Got IMU data: ", imu_data)
            imu_readings.append([imu_data[0], imu_data[1], imu_data[2], imu_data[3], imu_data[4], imu_data[5]])

        # Exit condition: Stop both threads if velocity is below 500
        if velocity < 350:
            total_time_elapsed = time.time() - start_time  # Calculate total elapsed time
            print(f"Velocity below threshold. Stopping threads. Total time elapsed: {total_time_elapsed:.2f} seconds.")
            stop_event.set()  # Signal the walk_gait thread to stop
            break

        time.sleep(0.05)  # Sample every second

# Function to save data to CSV
def save_to_csv():
    # Read existing data (if any) to preserve previous entries
    try:
        with open(csv_filename, "r") as file:
            reader = csv.reader(file)
            data = list(reader)
    except FileNotFoundError:
        data = [["Trial #", "Velocity Readings", "IMU Readings", "Time to End"]]

    # Convert arrays to JSON format for single-cell storage
    velocity_str = json.dumps(velocity_readings)
    imu_str = json.dumps(imu_readings)

    # Append new trial data
    data.append([trial_num, velocity_str, imu_str, f"{total_time_elapsed:.2f}"])

    # Write back to CSV
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"Trial {trial_num} data saved to {csv_filename}")

# Main function with threading
if __name__ == "__main__":
    # Create and start threads
    walk_thread = threading.Thread(target=walk_gait)
    record_thread = threading.Thread(target=record_motion_data)

    walk_thread.start()
    record_thread.start()

    # Wait for recording thread to finish (since it contains the stop condition)
    record_thread.join()

    # Now we ensure CSV writing is done before exiting
    save_to_csv()

    # Wait for walking thread to finish (it should stop when stop_event is set)
    walk_thread.join()
    
    print("Data collection finished. Exiting program.")
    initRobot()

