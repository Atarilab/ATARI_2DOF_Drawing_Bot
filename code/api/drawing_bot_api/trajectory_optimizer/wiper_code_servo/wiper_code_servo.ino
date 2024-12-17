//#include <Servo.h>
/*
Servo myservo;  // create servo object to control a servo

int pos = 0;    // variable to store the servo position

void setup() {
  Serial.begin(9600);
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
}


void loop() {
  for (pos = 0; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(100);                       // waits 15ms for the servo to reach the position
    Serial.println(pos);
  }

  for (pos = 180; pos >= 0; pos -= 1) { // goes from 180 degrees to 0 degrees
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(100);                       // waits 15ms for the servo to reach the position
    Serial.println(pos);
  }
}*/

#define DATA_PIN 9

void setup() {
  Serial.begin(9600);
  pinMode(DATA_PIN, OUTPUT);
}

void loop() {
  for(int i=0; i < 255; i ++) {
    analogWrite(DATA_PIN, i);
    delay(500);
    Serial.println(i);
  }

  delay(1000);

    for(int i=255; i > 0; i--) {
    analogWrite(DATA_PIN, 255-i);
    delay(500);
    Serial.println(i);
  }
  
  delay(1000);
}