// Syringe pump firmware v2 (Arduino Uno + step/dir driver), for pressure_control.py.
//
// Same wiring and sign convention as syringe_simple_controls_sketch (v1):
//   step pin 2, dir pin 3, enable pin 4 (LOW = enabled)
//   "F" / push in = dir HIGH = raises pressure = position counts DOWN; 0 = empty syringe.
//
// What v2 adds: steps are timed in loop() with micros(), so serial commands are read during a
// move (stop, new target, velocity); a velocity mode; holding torque on request; a position
// stream; the position survives the Uno's reset-on-connect (EEPROM, wear-levelled).
//
// Commands (one per line, 115200 baud):
//   F<n> / B<n>  move n steps in / out (v1 compatible: "OK CMD=F<n>", then "DONE POS=<p>")
//   M<n>         move n steps, + = in (push), - = out (pull); "DONE POS=<p>" when done
//   V<sps>       run at sps steps/s (+ = in); V0 stops. Needs a command at least every 2 s
//                (watchdog: "ERR WATCHDOG", stop) so a crashed PC can't drive the plunger
//                into the end stop
//   S            stop now ("OK STOP POS=<p>")
//   SPEED <sps>  speed for F/B/M moves (default 450 steps/s, as v1's 2 ms per step)
//   HOLD 0|1     1: motor stays enabled when idle (holds against the line pressure)
//   STREAM 0|1   1: "P=<pos> V=<sps>" every 50 ms
//   POS?         "POS=<p>"
//   ZERO         position = 0 ("OK ZERO POS=0")
//   SETPOS <p>   position = p ("OK SETPOS POS=<p>")
//   VER?         "VER syringe_v2 1.0"
// Errors: "ERR <reason>".

#include <EEPROM.h>

const int stepPin = 2;
const int dirPin = 3;
const int enablePin = 4;

const unsigned int stepPulseMicros = 10;
const long MAX_SPS = 1500;
const unsigned long WATCHDOG_MS = 2000;
const unsigned long SAVE_IDLE_MS = 2000;   // save the position after this long without motion
const unsigned long STREAM_MS = 50;

// EEPROM: 64 slots of {magic, seq, pos}; the slot with the highest seq is current.
struct Slot { uint16_t magic; uint16_t seq; long pos; };
const uint16_t MAGIC = 0x5A72;
const int N_SLOTS = 64;
int slotIndex = -1;
uint16_t slotSeq = 0;

long position = 0;
long savedPosition = 0;
long target = 0;            // for F/B/M moves
bool moving = false;        // a move toward target
float velocity = 0;         // steps/s in velocity mode (+ = in); 0 = off
long moveSps = 450;
bool hold = false;
bool streamOn = false;
bool enabled = false;
int dirState = -1;

unsigned long lastStepMicros = 0;
unsigned long lastCmdMillis = 0;
unsigned long lastMotionMillis = 0;
unsigned long lastStreamMillis = 0;
char line[40];
byte lineLen = 0;

void loadPosition() {
  Slot s;
  for (int i = 0; i < N_SLOTS; i++) {
    EEPROM.get(i * sizeof(Slot), s);
    if (s.magic != MAGIC) continue;
    if (slotIndex < 0 || (uint16_t)(s.seq - slotSeq) < 0x8000) {   // newer (wrap-safe)
      slotIndex = i;
      slotSeq = s.seq;
      position = s.pos;
    }
  }
  savedPosition = position;
}

void savePosition() {
  if (position == savedPosition) return;
  Slot s;
  slotIndex = (slotIndex + 1) % N_SLOTS;
  slotSeq++;
  s.magic = MAGIC;
  s.seq = slotSeq;
  s.pos = position;
  EEPROM.put(slotIndex * sizeof(Slot), s);
  savedPosition = position;
}

void setEnabled(bool on) {
  if (on == enabled) return;
  digitalWrite(enablePin, on ? LOW : HIGH);
  enabled = on;
  if (on) delay(10);          // driver wake-up, as v1
}

void setDir(bool in) {
  int d = in ? 1 : 0;
  if (d == dirState) return;
  digitalWrite(dirPin, in ? HIGH : LOW);
  dirState = d;
  delayMicroseconds(20);
}

void stepOnce(bool in) {
  setDir(in);
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(stepPulseMicros);
  digitalWrite(stepPin, LOW);
  position += in ? -1 : 1;
  lastMotionMillis = millis();
}

void stopMotion() {
  moving = false;
  velocity = 0;
  if (!hold) setEnabled(false);
}

void startMove(long steps) {        // + = in
  if (steps == 0) {
    Serial.print("DONE POS=");
    Serial.println(position);
    return;
  }
  velocity = 0;
  target = position - steps;
  moving = true;
  setEnabled(true);
  lastStepMicros = micros();
}

void reply(const char* what) {
  Serial.print(what);
  Serial.println(position);
}

void handleCommand(char* cmd) {
  lastCmdMillis = millis();
  String c = String(cmd);
  c.trim();
  if (c.length() == 0) return;

  if (c == "VER?") { Serial.println("VER syringe_v2 1.0"); return; }
  if (c == "POS?") { reply("POS="); return; }
  if (c == "ZERO") { stopMotion(); position = 0; savePosition(); reply("OK ZERO POS="); return; }
  if (c == "S") { stopMotion(); reply("OK STOP POS="); return; }
  if (c.startsWith("SETPOS ")) {
    stopMotion();
    position = c.substring(7).toInt();
    savePosition();
    reply("OK SETPOS POS=");
    return;
  }
  if (c.startsWith("SPEED ")) {
    long v = c.substring(6).toInt();
    if (v < 1 || v > MAX_SPS) { Serial.println("ERR INVALID_SPEED"); return; }
    moveSps = v;
    Serial.print("OK SPEED=");
    Serial.println(moveSps);
    return;
  }
  if (c.startsWith("HOLD ")) {
    hold = c.substring(5).toInt() != 0;
    if (hold) setEnabled(true);
    else if (!moving && velocity == 0) setEnabled(false);
    Serial.println(hold ? "OK HOLD=1" : "OK HOLD=0");
    return;
  }
  if (c.startsWith("STREAM ")) {
    streamOn = c.substring(7).toInt() != 0;
    Serial.println(streamOn ? "OK STREAM=1" : "OK STREAM=0");
    return;
  }
  char k = c.charAt(0);
  if (k == 'F' || k == 'B') {
    long n = c.substring(1).toInt();
    if (n <= 0) { Serial.println("ERR INVALID_STEPS"); return; }
    Serial.print("OK CMD=");
    Serial.println(c);
    startMove(k == 'F' ? n : -n);
    return;
  }
  if (k == 'M') {
    long n = c.substring(1).toInt();
    if (n == 0 && c != "M0") { Serial.println("ERR INVALID_STEPS"); return; }
    Serial.print("OK CMD=");
    Serial.println(c);
    startMove(n);
    return;
  }
  if (k == 'V') {
    long v = c.substring(1).toInt();
    if (v > MAX_SPS) v = MAX_SPS;
    if (v < -MAX_SPS) v = -MAX_SPS;
    moving = false;
    velocity = v;
    if (v != 0) {
      setEnabled(true);
      lastStepMicros = micros();
    } else if (!hold) {
      setEnabled(false);
    }
    return;                    // no reply: sent many times a second
  }
  Serial.println("ERR UNKNOWN_CMD");
}

void readSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (lineLen > 0) {
        line[lineLen] = 0;
        handleCommand(line);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = ch;
    }
  }
}

void runMotor() {
  unsigned long now = micros();
  if (moving) {
    unsigned long interval = 1000000UL / (unsigned long)moveSps;
    if (now - lastStepMicros >= interval) {
      lastStepMicros += interval;
      if (now - lastStepMicros > interval) lastStepMicros = now;   // don't catch up after a stall
      stepOnce(target < position);
      if (position == target) {
        moving = false;
        if (!hold) setEnabled(false);
        reply("DONE POS=");
      }
    }
  } else if (velocity != 0) {
    unsigned long interval = (unsigned long)(1000000.0 / fabs(velocity));
    if (now - lastStepMicros >= interval) {
      lastStepMicros += interval;
      if (now - lastStepMicros > interval) lastStepMicros = now;
      stepOnce(velocity > 0);
    }
    if (millis() - lastCmdMillis > WATCHDOG_MS) {
      stopMotion();
      Serial.println("ERR WATCHDOG");
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enablePin, OUTPUT);
  digitalWrite(stepPin, LOW);
  digitalWrite(dirPin, LOW);
  digitalWrite(enablePin, HIGH);   // disabled until a move (or HOLD 1)
  loadPosition();
  lastCmdMillis = millis();
  lastMotionMillis = millis();
  Serial.print("READY v2 POS=");
  Serial.println(position);
}

void loop() {
  readSerial();
  runMotor();
  unsigned long ms = millis();
  if (streamOn && ms - lastStreamMillis >= STREAM_MS) {
    lastStreamMillis = ms;
    Serial.print("P=");
    Serial.print(position);
    Serial.print(" V=");
    Serial.println((long)velocity);
  }
  if (!moving && velocity == 0 && ms - lastMotionMillis > SAVE_IDLE_MS) savePosition();
}
