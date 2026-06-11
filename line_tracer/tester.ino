void setup() {
  Serial.begin(9600);
}

void loop() {
  for (int i = 0; i <= 5; i++) {
    Serial.print("A");
    Serial.print(i);
    Serial.print(": ");
    Serial.print(analogRead(i));
    Serial.print("  ");
  }
  Serial.println();
  delay(500);
}
