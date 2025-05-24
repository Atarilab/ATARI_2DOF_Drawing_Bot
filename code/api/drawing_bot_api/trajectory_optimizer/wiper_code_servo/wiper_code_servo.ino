#define DATA_PIN 9
#define START_POINT 250
#define END_POINT 0


void setup() {
  Serial.begin(9600);
  pinMode(DATA_PIN, OUTPUT);
  for(int i=0; i < 5; i++) {
    analogWrite(DATA_PIN, START_POINT);
    delay(500);
    analogWrite(DATA_PIN, END_POINT);
    delay(500);
  }
}

void loop() {
}