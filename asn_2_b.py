import time
# import Board
from sonar import *
# import Mpu6050
import ros_robot_controller_sdk as rrc
from map_301 import *


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
runTimeSonar = 0.05

WALKS_PER_GRID = 4 * 2

TURN_ADJUST = -5
OFFSET_ADJUST = 0

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

def gridWalk(grids):
    if grids < 0:
        for _ in range(-grids * WALKS_PER_GRID - (grids // 6)): #Change 11.5 to be accurate
            walkBack()

    elif grids > 0:
        for _ in range(grids * WALKS_PER_GRID + (grids // 6)): #- (grids // 6)): #Change 11.5 to be accurate
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



def pathStep(init, final, heading):
    deltaI = final[0] - init[0]
    deltaJ = final[1] - init[1]

    if deltaI is not 0:
        if deltaI > 0:
            heading = globalTurn(heading, 2)
        elif deltaJ < 0:
            heading = globalTurn(heading, 1)
        gridWalk(abs(deltaI))
    if deltaJ is not 0:
        if deltaJ > 0:
            heading = globalTurn(heading, 3)
        elif deltaJ < 0:
            heading = globalTurn(heading, 4)
        gridWalk(abs(deltaJ))   

    return heading


def globalTurn(start, end):

    if start == 1:
        if end == 2:
            turn(180, TURN_ADJUST)
        elif end == 3:
            turn(90, TURN_ADJUST)
        elif end == 4: 
            turn(-90, TURN_ADJUST)
    elif start == 2:
        if end == 1:
            turn(180, TURN_ADJUST)
        elif end == 3:
            turn(-90, TURN_ADJUST)
        elif end == 4:
            turn(90, TURN_ADJUST)
    elif start == 3:
        if end == 1:
            turn(-90, TURN_ADJUST)
        elif end == 2:
            turn(90, TURN_ADJUST)
        elif end == 4:
            turn(180, TURN_ADJUST)
    elif start == 4:
        if end == 1:
            turn(90, TURN_ADJUST)
        elif end == 2:
            turn(-90, TURN_ADJUST)
        elif end == 3:
            turn(180, TURN_ADJUST)

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
    return visitList


def pathFollow(coordList, initHeading, finalHeading):

    heading = initHeading

    for i in range(len(coordList[:-1])):
        print(coordList[i])
        heading = pathStep(coordList[i], coordList[i+1], heading)
        

    heading = globalTurn(heading, finalHeading)

    return

def wallDetect(grid, position, currHeading):

    i = position[0]
    j = position[1]
    wallForward = False
    wallLeft = False
    wallRight = False

    sonarForward()
    time.sleep(SCAN_SLEEP)
    print("forward sonar read in wallDetect: ", sonar.getDistance())
    if sonar.getDistance() < WALL_THRESHOLD:
        wallForward = True

    time.sleep(SCAN_SLEEP)

    sonarLeft()
    time.sleep(SCAN_SLEEP)
    print("left sonar read in wallDetect: ", sonar.getDistance())
    if sonar.getDistance() < WALL_THRESHOLD:
        wallLeft = True

    time.sleep(SCAN_SLEEP * 2)

    sonarRight()
    time.sleep(SCAN_SLEEP * 2)
    print("right sonar read in wallDetect: ", sonar.getDistance())
    if sonar.getDistance() < WALL_THRESHOLD:
        wallRight = True

    time.sleep(SCAN_SLEEP // 2)

    if currHeading == 1: # Facing North
        if wallForward:
            #set wall i - 1
            grid.setObstacle(i, j, 1, 1)
        if wallLeft:
            #set wall j - 1
            grid.setObstacle(i, j, 1, 4)
        if wallRight:
            #set wall j + 1
            grid.setObstacle(i, j, 1, 2)

    elif currHeading == 2: # Facing South
        if wallForward:
            #set wall i + 1
            grid.setObstacle(i, j, 1, 3)
        if wallLeft:
            #set wall j + 1
            grid.setObstacle(i, j, 1, 2)
        if wallRight:
            #set wall j - 1
            grid.setObstacle(i, j, 1, 4)

    elif currHeading == 3: # Facing East
        if wallForward:
            #set wall j + 1
            grid.setObstacle(i, j, 1, 2)
        if wallLeft:
            #set wall i - 1
            grid.setObstacle(i, j, 1, 1)
        if wallRight:
            #set wall i + 1
            grid.setObstacle(i, j, 1, 3)

    elif currHeading == 4: # Facing West
        if wallForward:
            #set wall j - 1
            grid.setObstacle(i, j, 1, 4)
        if wallLeft:
            #set wall i + 1
            grid.setObstacle(i, j, 1, 3)
        if wallRight:
            #set wall i - 1
            grid.setObstacle(i, j, 1, 1)
    return


def mapSeek(grid, finalPosition):


    i = 0
    j = 0

    initI = 0
    initJ = 0

    finalI = finalPosition[0]
    finalJ = finalPosition[1]
    finalHeading = finalPosition[2]

    heading = 1

    rows, cols = grid.costmap_size_row, grid.costmap_size_col

    for i_ in range(rows):
        for j_ in range(cols):
            if grid.getCost(i_, j_) == 0:
                grid.setCost(i_, j_, -1)

    grid.setCost(initI, initJ, 0)

    forks = {}
    path = [(initI, initJ)]

    wallDetect(grid, (i, j), heading)

    heading = globalTurn(heading, 3)

    while i != finalI or j != finalJ:

        wallDetect(grid, (i,j), heading)
        viableNeighbors = []

        #Set neighbor costs
        for key in directions.keys():
                di = key[0]
                dj = key[1]
                nextI = i - di
                nextJ = j + dj

                #Check if in map bounds
                if nextI < rows and nextI >= 0 and nextJ >= 0 and nextJ < cols:
                    nextPosCost = grid.getCost(nextI, nextJ)

                    #Check if next cell has been visited or if there is a wall
                    if nextPosCost == -1 and grid.getNeighborObstacle(i, j, directions[key]) == 0:

                        #Price next cell
                        grid.setCost(nextI, nextJ, grid.getCost(i, j) + 1)

                        viableNeighbors.append((nextI, nextJ))

                        #Next cell to visit is whatever is priced last
                        nextCoord = (nextI, nextJ)



        #if all paths are blocked:

        if not viableNeighbors: #This means new coord was not found
            print("no viable neighbors")

            #FIGURE OUT HOW TO TURN AROUND HERE WITH GLOBAL TURN
            #THIS MAY NOT BE NEEDED IDK
            #turn(180, TURN_ADJUST)

            #If the path is blocked and there are no more forks, end
            if forks == {}:
                print("Coordinate could not be found!\n")
                break


            while (i, j) not in forks.keys():
                #Follow the most recent path backwards until a fork is reached
                heading = pathStep((i,j), path[-1], heading)
                i, j = path[-1][0], path[-1][1]
                path.pop()

            nextI, nextJ = forks[(i,j)][-1][0], forks[(i,j)][-1][1]
            heading = pathStep((i, j), (nextI, nextJ), heading)
            forks[(i,j)].pop()

            #If all forks are searched, remove fork
            if len(forks[(i, j)]) == 0:
                forks.pop((i, j))

            i, j = nextI, nextJ

        else:
            #Remove next coord from neighbors
            viableNeighbors.pop()
            print("looking at this neighbor: ", viableNeighbors)

            #if there are multiple available cells:
            #if len(viableNeighbors) >= 1:
            forks[(i, j)] = viableNeighbors

            #Move to next cell, add in P controller to this potentially
            heading = pathStep((i, j), nextCoord, heading)

            #Update i and j
            i, j = nextCoord[0], nextCoord[1]

            path.append((i,j))

        #grid.printCostMap()
        print("intermediate obstacle map")
        grid.printObstacleMap()
        print("intermediate cost map")
        grid.printCostMap()


    globalTurn(heading, finalHeading)

    return


initRobot()
#turn(90, -25)
grid = CSME301Map()
grid.clearObstacleMap()
grid.clearCostMap()
mapSeek(grid, [8, 8, 1]) 
print("obstacle grid")
grid.printObstacleMap()
#1. Be able to accept a starting position xs and end position xg
'''
coords = getCoords()

#2. Set the cost of each grid cell
priceGrid(coords[0], grid)

#3. Generate a path from xs to xg. 
steps = pathGen(coords[0], coords[1], grid)
print(steps)

#4. Generate command sequence... call the commands developed in Step 1
#5. Walk the path. 
grid.printCostMap()
grid.printObstacleMap()

pathFollow(steps, coords[0][2], coords[1][2])
'''

#initRobot()

#coords = getCoords()

#coordMove(coords[0], coords[1])


#coords = getCoords()


#sonarLeft()
#test_map = CSME301Map()
#test_map.printObstacleMap()
#gridWalk(3)
