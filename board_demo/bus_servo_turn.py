import sys
import time
import signal
import threading
import ros_robot_controller_sdk as rrc

print('''
**********************************************************
********功能:幻尔科技树莓派扩展板，控制总线舵机转动(Function:Hiwonder Raspberry Pi expansion board, bus servo rotation control)**********
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！(Press "Ctrl+C" to exit the program. If it fails, please try again multiple times!)
----------------------------------------------------------
''')
board = rrc.Board()
start = True

# 关闭前处理(process before closing)
def Stop(signum, frame):
    global start
    start = False
    print('关闭中...')

signal.signal(signal.SIGINT, Stop)

if __name__ == '__main__':
    while True:
       board.bus_servo_set_position(1, [[1, 500], [2, 500]])
       time.sleep(1)
       board.bus_servo_set_position(2, [[1, 0], [2, 0]])
       time.sleep(1)
       board.bus_servo_stop([1, 2])
       time.sleep(1)
       if not start:
        board.bus_servo_stop([1, 2])
        time.sleep(1)
        print('已关闭')
        break