#include <SimpleFOC.h>
#include <Wire.h>
#include <math.h>

//#define ANGLE_CALIBRATION
//#define DEBUG

#define READY_STATMENT_WAIT_TIME 1000 

// I2C setup
#define SDA_1 33
#define SCL_1 32
#define SDA_2 23
#define SCL_2 22
#define freq 100000

#define IN1_LEFT 26
#define IN2_LEFT 27
#define IN3_LEFT 14
#define EN_LEFT 12
#define GND_LEFT 15

#define IN1_RIGHT 2
#define IN2_RIGHT 0
#define IN3_RIGHT 4
#define EN_RIGHT 16
#define GND_RIGHT 25

TwoWire i2cOne = TwoWire(0);
TwoWire i2cTwo = TwoWire(1);

// BLDC motor & driver instance
BLDCMotor motorLeft = BLDCMotor(7, 11.2);
BLDCDriver3PWM driverLeft = BLDCDriver3PWM(26, 27, 14, 12);

//BLDCMotor motorRight = BLDCMotor(6, 11.2);
BLDCMotor motorRight = BLDCMotor(7, 11.2);
BLDCDriver3PWM driverRight = BLDCDriver3PWM(2, 0, 4, 16);

// Magnetic encoders
MagneticSensorI2C encoderLeft = MagneticSensorI2C(AS5600_I2C);
MagneticSensorI2C encoderRight = MagneticSensorI2C(AS5600_I2C);

int system_is_ready = 0;

// commander communication instance
Commander command = Commander(Serial);
void doMotionLeft(char* cmd){ command.motion(&motorLeft, cmd); }
void doMotionRight(char* cmd){ command.motion(&motorRight, cmd); }
void onPid(char* cmd){ command.pid(&motorLeft.P_angle, cmd); }
// void doMotor(char* cmd){ command.motor(&motor, cmd); }

void restart(char* cmd){
  ESP.restart();
}

void send_ready_statement(char* cmd) {
  Serial.println("Checking status...");
  if (system_is_ready) {
    Serial.println("RDY"); 
  }
}

float find_zero_angle(FOCMotor* motor, int direction) {
  float zero_angle = 0;
  
  motor->target = motor->shaft_angle - (direction * 18.85);

  int timer = millis();
  float previous_position = motor->shaft_angle + direction;

  while(1) {
    motor->loopFOC();
    motor->move();

    if ( fabs(motor->shaft_velocity) < 1) {
      if (millis()-timer > 100) {
        if (direction+1) zero_angle = motor->shaft_angle + 0.3;
        else zero_angle = motor->shaft_angle - 9.42;
        break;
      }
    }
    else timer = millis();

    previous_position = motor->shaft_angle; 
    //Serial.println(previous_position);
  }

  motor->target = zero_angle + 4.71;

  timer = millis();
  while (1) {
    motor->loopFOC();
    motor->move();

    if ((millis() - timer) > 500) {
      if (fabs(motor->shaft_velocity) < 2) break;
    }
  }

  Serial.println("Zero angle found!");
  return zero_angle;

}

void setup() {

  i2cOne.begin(SDA_1, SCL_1, freq);
  i2cTwo.begin(SDA_2, SCL_2, freq);

  pinMode(GND_LEFT, OUTPUT);
  pinMode(GND_RIGHT, OUTPUT);
  digitalWrite(GND_LEFT, 0);
  digitalWrite(GND_RIGHT, 0);
  
  // use monitoring with serial
  //Serial.setPins(-1, -1, 15, 14);
  //Serial.setHwFlowCtrlMode(UART_HW_FLOWCTRL_CTS_RTS, 64);
  delay(10);
  Serial.begin(115200);
  //Serial.begin(921600);
  // enable more verbose output for debugging
  // comment out if not needed
  //SimpleFOCDebug::enable(&Serial);

  // initialize encoder sensor hardware
  encoderLeft.init(&i2cOne);
  encoderRight.init(&i2cTwo);

  // link the motor to the sensor
  motorLeft.linkSensor(&encoderLeft);
  motorRight.linkSensor(&encoderRight);

  // driver config
  // power supply voltage [V]
  driverLeft.voltage_power_supply = 12;
  driverLeft.init();
  driverRight.voltage_power_supply = 12;
  driverRight.init();

  // link driver
  motorLeft.linkDriver(&driverLeft);
  motorRight.linkDriver(&driverRight);

  // aligning voltage [V]
  motorLeft.voltage_sensor_align = 12;
  motorRight.voltage_sensor_align = 12;

  // set control loop type to be used
  motorLeft.controller = MotionControlType::angle;
  motorRight.controller = MotionControlType::angle;

  // contoller configuration based on the control type
  motorLeft.PID_velocity.P = 0.008;
  motorLeft.PID_velocity.I = 0.0;
  motorLeft.PID_velocity.D = 0.0;
  motorLeft.P_angle.P = 100;
  motorLeft.P_angle.I = 0.0;
  motorLeft.P_angle.D = 0.4;

  motorRight.PID_velocity.P = 0.008;
  motorRight.PID_velocity.I = 0.0;
  motorRight.PID_velocity.D = 0.0;
  motorRight.P_angle.P = 100;
  motorRight.P_angle.I = 0.0;
  motorRight.P_angle.D = 0.4;


  // default voltage_power_supply
  motorLeft.voltage_limit = 12;
  motorRight.voltage_limit = 12;

  // velocity low pass filtering time constant
  //motorLeft.LPF_velocity.Tf = 0.05;
  motorLeft.LPF_angle.Tf = 0.02;
  //motorRight.LPF_velocity.Tf = 0.05;
  motorRight.LPF_angle.Tf = 0.02;

  // angle loop velocity limit
  motorLeft.velocity_limit = 25;
  motorRight.velocity_limit = 25;

  // comment out if not needed
  /*
  motorLeft.useMonitoring(Serial);
  motorLeft.monitor_downsample = 0; // disable intially
  motorLeft.monitor_variables = _MON_TARGET | _MON_VEL | _MON_ANGLE; // monitor target velocity and angle

  motorRight.useMonitoring(Serial);
  motorRight.monitor_downsample = 1; // disable intially
  motorRight.monitor_variables = _MON_TARGET | _MON_VEL | _MON_ANGLE; // monitor target velocity and angle
  */
  
  // initialise motor
  motorLeft.init();
  motorRight.init();
#ifndef ANGLE_CALIBRATION
  // align encoder and start FOC
  motorLeft.initFOC();
  motorRight.initFOC();

  Serial.println("Finding zero angle for left motor");
  motorLeft.sensor_offset = find_zero_angle(&motorLeft, 1); //2.39;
  Serial.println("Finding zero angle for right motor");
  motorRight.sensor_offset = find_zero_angle(&motorRight, -1); //-0.87;
#endif
  // subscribe motor to the commander
  command.add('W', doMotionLeft, "motion control");  // "West" - "Left" Motor
  command.add('E', doMotionRight, "motion control"); // "East" - "Right" Motor
  command.add('C', onPid, "pid");
  command.add('R', restart, "restart");
  command.add('I', send_ready_statement, "ready_rtn");

  // set the inital target value
  motorLeft.target =  4.71;
  motorRight.target = 4.71;
  
  // Run user commands to configure and the motor (find the full command list in docs.simplefoc.com)
  Serial.println("Motor ready.");

  system_is_ready = 1;
}


void loop() {

  #if defined ANGLE_CALIBRATION || defined DEBUG
    encoderLeft.update();
    encoderRight.update();
    Serial.print("Left | Precision angle: ");
    Serial.print(encoderLeft.getMechanicalAngle());
    Serial.print("\tAngle and Rotations: ");
    Serial.println(encoderLeft.getAngle());
    Serial.print("Right | Precision angle: ");
    Serial.print(encoderRight.getMechanicalAngle());
    Serial.print("\tAngle and Rotations: ");
    Serial.println(encoderRight.getAngle());
    Serial.println();
    #ifdef ANGLE_CALIBRATION
      delay(200);
    #endif
  #endif

  #ifndef ANGLE_CALIBRATION
    // iterative setting FOC phase voltage
    motorLeft.loopFOC();
    motorRight.loopFOC();

    // iterative function setting the outter loop target
    motorLeft.move();
    motorRight.move();

    // motor monitoring
    //motorLeft.monitor();
    //motorRight.monitor();

    // user communication
    command.run();

  #endif
}