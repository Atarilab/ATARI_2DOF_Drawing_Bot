from delta_utils import plt
from delta_utils import plot_delta
from delta_utils import ik_delta
from delta_utils import plot_box
from math import cos, sin, pi
import time
import serial

ser = serial.Serial('/dev/cu.usbserial-0001', 115200)

ROTATION_SPEED = 2000 # defines how many milliseconds one rotation takes
X0 = 0
Y0 = 0.11
R = 0.03

def millis():
    return time.time()*1000

def calc_new_angle(current_angle, time_passed):
    return (time_passed / ROTATION_SPEED) * 2 * pi + current_angle

def calc_position(circ_angle):
    x = cos(circ_angle) * R + X0
    y = sin(circ_angle) * R + Y0

    return [[x, y], circ_angle]

def mainb():
    clock = millis()
    current_angle = 0

    current_angle = calc_new_angle(current_angle, millis() - clock)
    position = calc_position(current_angle)
    clock = millis()

    motor_angles = ik_delta(position[0])

def main():
    clock = millis()
    current_angle = 0

    time.sleep(15)

    while(1):
        #fig = plt.figure()

        current_angle = calc_new_angle(current_angle, millis() - clock)
        position = calc_position(current_angle)
        clock = millis()

        motor_angles = ik_delta(position[0])
        #plot_delta(motor_angles)

        message1 = f'L{3*float(motor_angles[0])}\n'
        message2 = f'R{3*float(motor_angles[1])}\n'
        '''
        ser.rts = 1
        while not ser.cts:
            print(f'CTS: {ser.cts}')
            time.sleep(0.0001)
            print("waiting...")
        '''
        #if (ser.cts):
        #print("sending...")
        #print(f"\t{message1.encode('utf-8')}\t{message2.encode('utf-8')}")
        #time.sleep(0.001)
        ser.write(message1.encode('utf-8'))
        #time.sleep(0.0005)
        ser.write(message2.encode('utf-8'))
        time.sleep(0.005)
        #ser.rts = 0

        #plt.show()

main()