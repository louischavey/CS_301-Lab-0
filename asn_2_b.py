import time
# import Board
from sonar import *
# import Mpu6050
import ros_robot_controller_sdk as rrc
from map_ import *


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

print('''Assignment 1 for Group B''')
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

runTimeForwardBack = 0.15
runTimeUpDown = 0.1

WALKS_PER_GRID = 3

ADJUST_THRESHOLD = 665 - LEG_BACK  # 665 is roughly max we can move leg before it touches other one

jointIDs = list(range(1, 19))

directions = {(-1, 0): 3, (1, 0): 1, (0, -1): 4, (0, 1): 2}


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
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK + 5 + adjust]])
                #board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK]])
        elif adjust < 0:
            if id < 10:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD + adjust]])
            else:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK + 5 - adjust]])
                #board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK - adjust]])

        else:
            if id < 10:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD]])
            else:
                board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK + 5]])
                #board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK]])

    return


def legsBack(legID):
    for id in legID:
        if id < 10:
            board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_BACK]])
        else:
            board.bus_servo_set_position(runTimeForwardBack, [[id, LEG_FORWARD]])
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
    print(f"Distance {dis} has P of {P}")
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

    time.sleep(1)

    legsDown(evenVerts)

    time.sleep(1)

    legsForward(evenLegs)
    legsUp(oddVerts)
    time.sleep(0.5)
    legsBack(oddLegs)

    time.sleep(1)

    legsDown(oddVerts)

    time.sleep(1)

    legsUp(evenVerts)
    legsForward(oddLegs)
    time.sleep(0.5)
    return

def turnRightNew(adjust=0):
    legsDown(evenVerts)

    
    time.sleep(runTimeUpDown)
    legsUp(oddVerts)

   
    time.sleep(runTimeUpDown)

    legsForward([1,7,10,16], adjust)
    legsBack([13, 4])

    time.sleep(runTimeUpDown)

    legsDown(oddVerts)
    time.sleep(runTimeUpDown)
    legsUp(evenVerts)

    time.sleep(runTimeUpDown)
    #legsBack([1, 7, 4])
    #legsForward([13, 10, 16])

    legsBack([1, 7, 10, 16])
    legsForward([13, 4], adjust)
    time.sleep(runTimeUpDown)
    
    return

def turnLeftNew(adjust=0):
    legsDown(evenVerts)

    time.sleep(runTimeUpDown)

    legsUp(oddVerts)
    
    time.sleep(runTimeUpDown)

    legsBack([1,7,10,16])
    legsForward([13, 4], adjust)

    time.sleep(runTimeUpDown)

    legsDown(oddVerts)
    time.sleep(runTimeUpDown)
    legsUp(evenVerts)

    time.sleep(runTimeUpDown)
    #legsBack([1, 7, 4])
    #legsForward([13, 10, 16])

    legsForward([1, 7, 10, 16], adjust)
    legsBack([13, 4])

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
    board.bus_servo_set_position(runTimeUpDown, [[21, 110]])
    print("sonar right")
    return

def sonarLeft():
    board.bus_servo_set_position(runTimeUpDown, [[21, 890]])
    print("sonar left")
    return

def sonarForward():
    board.bus_servo_set_position(runTimeUpDown, [[21, 500]])
    print("sonar fwd")
    return

def gridWalk(grids):
    if grids < 0:
        for _ in range(-grids * WALKS_PER_GRID + (grids // 6)): #Change 11.5 to be accurate
            walkBack()
    elif grids > 0:
        for _ in range(grids * WALKS_PER_GRID): #- (grids // 6)): #Change 11.5 to be accurate
            #print(grids * WALKS_PER_GRID - (grids // 6))
            walkForward()

def getCoords():
    start = input("Enter starting location: ")
    print("\n")
    end = input("Enter end location: ")
    print("\n")

    initCoords = []
    finalCoords = []
    #currNum = ""

    start_x, start_y, start_o = start.split(",")
    initCoords.append(int(start_x))
    initCoords.append(int(start_y))
    initCoords.append(int(start_o))

    end_x, end_y, end_o = end.split(",")
    finalCoords.append(int(end_x))
    finalCoords.append(int(end_y))
    finalCoords.append(int(end_o))

    
    #for char in start:
    #    if char is not (" " or ","):
    #        currNum += char
    #    else: 
    #        initCoords.append(int(currNum))
    #        currNum = ""
    
    #currNum = ""
    
    #for char in end:
    #    if char is not (" " or ","):
    #        currNum += char
    #    else: 
    #        finalCoords.append(int(currNum))
    #        currNum = ""
            
    print("Starting at: ", initCoords)
    print("Ending at: ", finalCoords)

    return [initCoords, finalCoords]

currHeading = 3

def coordMove(init, final):
    deltaI = final[0] - init[0]
    deltaJ = final[1] - init[1]

    currHeading = init[2]
    
    if deltaJ is not 0:
        gridWalk(deltaJ)

    turn(90)

    if deltaI is not 0:
        gridWalk(deltaI)

    if final[2] is 1:
        turn(180)
    elif final[2] is 3:
        turn(-90)
    elif final[2] is 4:
        turn(90)

    return

heading = 1

def pathStep(init, final):
    deltaI = final[0] - init[0]
    deltaJ = final[1] - init[1]

    if deltaI is not 0:
        heading = globalTurn(heading, 2)
        gridWalk(deltaI)
    if deltaJ is not 0:
        heading = globalTurn(heading, 3)
        gridWalk(deltaJ)   

    return 


def globalTurn(start, end):

    if start is 1:
        if end is 2:
            turn(180)
        elif end is 3:
            turn(90)
        elif end is 4: 
            turn(-90)
    elif start is 2:
        if end is 1:
            turn(180)
        elif end is 3:
            turn(-90)
        elif end is 4:
            turn(90)
    elif start is 3:
        if end is 1:
            turn(-90)
        elif end is 2:
            turn(90)
        elif end is 4:
            turn(180)
    elif start is 4:
        if end is 1:
            turn(90)
        elif end is 2:
            turn(-90)
        elif end is 3:
            turn(180)

    return end


def priceGrid(init, grid):

    rows, cols = grid.costmap_size_row, grid.costmap_size_col

    initI = init[0]
    initJ = init[1]

    for i_ in range(rows):
        for j_ in range(cols):
            if grid.getCost(i_, j_) == 0:
                grid.setCost(i_, j_, -1)


    seekList = [(initI, initJ)]
    grid.setCost(initI, initJ, 0)
    while seekList:
        for i in range(len(seekList)):
            coord = seekList[i]
            i, j = coord[0], coord[1]
            for key in directions.keys():
                di = key[0]
                dj = key[1]
                nextI = i - di
                nextJ = j + dj
                if nextI < rows and nextI >= 0 and nextJ >= 0 and nextJ < cols:
                    nextPosCost = grid.getCost(nextI, nextJ)
                    if nextPosCost == -1 and grid.getNeighborObstacle(i, j, directions[key]) == 0:
                        grid.setCost(nextI, nextJ, grid.getCost(i, j) + 1)
                        seekList.append((nextI, nextJ))

        seekList.pop(0)
    return

def pathGen(init, final, grid):

    rows, cols = grid.costmap_size_row, grid.costmap_size_col

    initI = init[0]
    initJ = init[1]

    finalI = final[0]
    finalJ = final[1]

    #directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visitList = [(finalI, finalJ)]

    i, j = finalI, finalJ
    visited = set()
    visited.add((i, j))

    while i != initI or j != initJ:
        minCost = 100

        for d in directions.keys():
            di = d[0]
            dj = d[1]
            nextI = i - di
            nextJ = j + dj
            #print("di: " , di)
            #print("dj: ", dj)
            #print("nextI: ", nextI)
            #print("nextJ: ", nextJ)
            
            #print(visited)

            if nextI < rows and nextJ < cols and nextI >= 0 and nextJ >= 0 and (nextI, nextJ) not in visited and grid.getNeighborObstacle(i, j, directions[d]) == 0:
                if grid.getCost(nextI, nextJ) < minCost:
                    minCost = grid.getCost(nextI, nextJ)
                    nextCoord = (nextI, nextJ)
                    visited.add(nextCoord)
        #print(nextCoord)
        visitList.insert(0, nextCoord)
        i = nextCoord[0]
        j = nextCoord[1]
    print("visitList in pathGen: ", visitList)            
    return visitList


def pathFollow(coordList, initHeading, finalHeading):

    heading = initHeading

    for i in range(coordList-1):
        pathStep(coordList[i], [coordList[i+1]])

    heading = globalTurn(heading, finalHeading)

    return


initRobot()
turn(90, 10)
grid = CSME301Map()

#1. Be able to accept a starting position xs and end position xg
coords = getCoords()

#2. Set the cost of each grid cell
priceGrid(coords[0], grid)

#3. Generate a path from xs to xg. 
steps = pathGen(coords[0], coords[1], grid)
print(steps)

#4. Generate command sequence... call the commands developed in Step 1
#5. Walk the path. 
#pathFollow(steps, coords[0][2], coords[1][2])


#initRobot()

#coords = getCoords()

#coordMove(coords[0], coords[1])
grid.printCostMap()
grid.printObstacleMap()

#coords = getCoords()


#sonarLeft()
#test_map = CSME301Map()
#test_map.printObstacleMap()
#gridWalk(3)
