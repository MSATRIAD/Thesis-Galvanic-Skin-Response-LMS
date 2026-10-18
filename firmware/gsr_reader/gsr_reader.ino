const int PIN_GSR = A0;
const unsigned long SAMPLE_INTERVAL_MS = 100;
const unsigned long BLINK_INTERVAL_MS = 1000;

unsigned long lastSampleAt = 0;
unsigned long lastBlinkAt = 0;
bool ledState = false;

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(115200);
  unsigned long waitStart = millis();
  while (!Serial && millis() - waitStart < 3000) { ; }

  analogReadResolution(12);
  pinMode(PIN_GSR, INPUT);

}

void loop() {
  unsigned long now = millis();

  if (now - lastBlinkAt >= BLINK_INTERVAL_MS) {
    lastBlinkAt = now;
    ledState = !ledState;
    digitalWrite(LED_BUILTIN, ledState ? HIGH : LOW);
  }

  if (now - lastSampleAt < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleAt = now;

  int raw12 = analogRead(PIN_GSR);
  int conv10 = map(raw12, 0, 4095, 0, 1023);

  Serial.print("Mentah (12-bit): ");
  Serial.print(raw12);
  Serial.print(" | Hasil Konversi (10-bit): ");
  Serial.println(conv10);
}
