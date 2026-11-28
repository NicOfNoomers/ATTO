// Pump driver with carrier-PWM amplitude control, XIAO nRF52840 on the atto_drvr_dev_kit board (NM, 28 Nov 2026).
//
// Why (capsule driver note 12): the capsule runs the DRV8837 straight off 2 x SR516 cells (117-190 ohm stack). A plain
// +-VBAT square wave would collapse the battery, so the capsule sets the coil amplitude with a PWM carrier inside each
// half-cycle (slow decay; the coil sees +-D x VM). This firmware lets the lab rig drive the pump the same way.
//
// Board (atto_drvr_dev_kit.kicad_pcb): D0 -> DRV8837 IN1 (PWM+), D1 -> IN2 (PWM-), D2 -> nSLEEP (M_EN).
// D10 (M_CTRL_V) reaches VM only through 10 kohm: left as an input. M+/M- to the pump.
//
// Compatibility with the dashboard (protocol freq_only / freq_duty; the dashboard never reads replies):
//   "<integer>\n" or "F <integer>\n"   pump frequency in Hz, 0 = stop                (unchanged)
//   "D <x>\n"                           the dashboard's legacy on/off duty: IGNORED    (it sends D 0 at start/stop)
// New commands:
//   "A <0..1>\n"     coil amplitude as carrier duty D (fraction of each carrier period in drive). Default 1.0 =
//                    today's plain +-VM square wave.
//   "C <hz>\n"       carrier frequency, 10000..250000 (default 50000; DRV8837 allows 0-250 kHz)
//   "R <s>\n"        amplitude ramp at start and stop, 0..1.5 s (default 0)
//   "P <0|1>\n"      PWM edge-polarity convention (default 0); flip only if a bench check shows inverted duty
//   "?\n"            status line
// Scheme (slow decay, as the capsule PLAN 3.2): forward half-cycle IN1 high, IN2 low for D of each carrier period and
// high (brake, both low-side on) for 1 - D; reverse half-cycle mirrored. Hardware PWM0 + EasyDMA: sequence 0 =
// forward, sequence 1 = reverse, each value repeated N times (REFRESH = N - 1), N = carrier / (2 f), looped forever.
// COUNTERTOP is trimmed per frequency so that the pump frequency is exact (the carrier lands within a few % of the set).
// Stop: PWM stopped, IN1 = IN2 = low (coast), nSLEEP low.
#include <Arduino.h>
#include <Adafruit_TinyUSB.h>   // USB CDC Serial on the Seeed (Adafruit-based) nRF52 core

static const uint8_t PIN_IN1 = D0, PIN_IN2 = D1, PIN_NSLEEP = D2, PIN_MCTRL = D10;
static const uint32_t PWM_CLK = 16000000UL;

static volatile uint16_t seqF[4], seqR[4];         // forward / reverse: one value per channel (decoder Individual)
static float duty = 1.0f, ampNow = 0.0f;           // target duty and the duty being output (ramps)
static uint32_t carrierSet = 50000, fSet = 0;
static float rampS = 0.0f;
static uint8_t polFlip = 0;
static bool running = false;
static uint16_t top = 320;
static uint32_t nRep = 1;

static uint16_t valHigh() { return polFlip ? (uint16_t)0x7FFF : (uint16_t)(0x8000 | 0x7FFF); }   // high all period
static uint16_t valLowFor(float d) {                // low for d of the period, then high (brake)
  if (d >= 0.999f) return polFlip ? (uint16_t)(0x8000 | 0x7FFF) : (uint16_t)0x7FFF;           // low all period
  uint16_t c = (uint16_t)(d * top + 0.5f);
  return polFlip ? (uint16_t)(0x8000 | c) : c;     // default: first edge rising (low, then high at c)
}

static void writeSeq(float d) {
  uint16_t hi = valHigh(), lo = valLowFor(d);
  seqF[0] = hi; seqF[1] = lo; seqF[2] = 0; seqF[3] = 0;   // forward: IN1 high, IN2 low for d
  seqR[0] = lo; seqR[1] = hi; seqR[2] = 0; seqR[3] = 0;   // reverse: IN2 high, IN1 low for d
  ampNow = d;
}

static void pinsCoast() {
  NRF_PWM0->ENABLE = 0;
  pinMode(PIN_IN1, OUTPUT); digitalWrite(PIN_IN1, LOW);
  pinMode(PIN_IN2, OUTPUT); digitalWrite(PIN_IN2, LOW);
}

static void pwmStop() {
  if (NRF_PWM0->ENABLE) {
    NRF_PWM0->EVENTS_STOPPED = 0;
    NRF_PWM0->TASKS_STOP = 1;
    uint32_t t0 = micros();
    while (!NRF_PWM0->EVENTS_STOPPED && micros() - t0 < 5000) {}
  }
  pinsCoast();
}

static void pwmStart(uint32_t f) {
  // exact pump frequency: N carrier periods per half-cycle, COUNTERTOP trimmed
  uint32_t n = (uint32_t)((float)carrierSet / (2.0f * f) + 0.5f);
  if (n < 1) n = 1;
  uint32_t t = (uint32_t)((float)PWM_CLK / (2.0f * n * f) + 0.5f);
  if (t > 32767) { t = 32767; }
  if (t < 16) t = 16;
  top = (uint16_t)t;
  nRep = n;
  NRF_PWM0->ENABLE = 0;
  NRF_PWM0->PSEL.OUT[0] = g_ADigitalPinMap[PIN_IN1];
  NRF_PWM0->PSEL.OUT[1] = g_ADigitalPinMap[PIN_IN2];
  NRF_PWM0->PSEL.OUT[2] = 0xFFFFFFFFUL;
  NRF_PWM0->PSEL.OUT[3] = 0xFFFFFFFFUL;
  NRF_PWM0->MODE = PWM_MODE_UPDOWN_Up;
  NRF_PWM0->PRESCALER = PWM_PRESCALER_PRESCALER_DIV_1;
  NRF_PWM0->COUNTERTOP = top;
  NRF_PWM0->DECODER = (PWM_DECODER_LOAD_Individual << PWM_DECODER_LOAD_Pos) | (PWM_DECODER_MODE_RefreshCount << PWM_DECODER_MODE_Pos);
  NRF_PWM0->SEQ[0].PTR = (uint32_t)seqF; NRF_PWM0->SEQ[0].CNT = 4; NRF_PWM0->SEQ[0].REFRESH = n - 1; NRF_PWM0->SEQ[0].ENDDELAY = 0;
  NRF_PWM0->SEQ[1].PTR = (uint32_t)seqR; NRF_PWM0->SEQ[1].CNT = 4; NRF_PWM0->SEQ[1].REFRESH = n - 1; NRF_PWM0->SEQ[1].ENDDELAY = 0;
  NRF_PWM0->LOOP = 1;
  NRF_PWM0->SHORTS = PWM_SHORTS_LOOPSDONE_SEQSTART0_Msk;   // forward, reverse, forward, ... forever
  NRF_PWM0->ENABLE = 1;
  NRF_PWM0->EVENTS_SEQEND[0] = 0;
  NRF_PWM0->TASKS_SEQSTART[0] = 1;
}

static void rampTo(float target) {                  // amplitude ramp over rampS (scaled by the step size)
  float from = ampNow;
  if (rampS <= 0.0f || fabsf(target - from) < 0.005f) { writeSeq(target); return; }
  uint32_t steps = (uint32_t)(rampS * fabsf(target - from) * 200.0f) + 1;   // 5 ms steps, full 0..1 = rampS
  for (uint32_t k = 1; k <= steps; k++) {
    writeSeq(from + (target - from) * (float)k / (float)steps);
    delay(5);
  }
}

static void setFreq(uint32_t f) {
  if (f == 0) {
    if (running) { rampTo(0.0f); pwmStop(); }
    digitalWrite(PIN_NSLEEP, LOW);
    running = false; fSet = 0;
    return;
  }
  if (f > 20000) f = 20000;
  digitalWrite(PIN_NSLEEP, HIGH);
  delayMicroseconds(60);                            // DRV8837 wake-up (tWAKE)
  bool wasRunning = running;
  if (wasRunning) pwmStop();
  writeSeq(wasRunning ? ampNow : (rampS > 0 ? 0.0f : duty));
  pwmStart(f);
  running = true; fSet = f;
  if (ampNow != duty) rampTo(duty);
}

static void status() {
  float fAct = running ? (float)PWM_CLK / (2.0f * nRep * top) : 0.0f;
  Serial.print("PUMP_DRIVER_DUTY v1 mcu=nrf52840 f="); Serial.print(fSet);
  Serial.print(" f_act="); Serial.print(fAct, 2);
  Serial.print(" D="); Serial.print(duty, 3);
  Serial.print(" D_now="); Serial.print(ampNow, 3);
  Serial.print(" carrier="); Serial.print(carrierSet);
  Serial.print(" carrier_act="); Serial.print(running ? (float)PWM_CLK / top : 0.0f, 0);
  Serial.print(" ramp_s="); Serial.print(rampS, 2);
  Serial.print(" pol="); Serial.print(polFlip);
  Serial.print(" running="); Serial.println(running ? 1 : 0);
}

static char line[48];
static uint8_t len = 0;

static void handle(char *s) {
  while (*s == ' ' || *s == '\t') s++;
  if (!*s) return;
  char c = *s;
  if ((c >= '0' && c <= '9') || c == '-') { long f = atol(s); setFreq(f < 0 ? 0 : (uint32_t)f); return; }
  char *v = s + 1;
  while (*v == ' ') v++;
  switch (c) {
    case 'F': case 'f': { long f = atol(v); setFreq(f < 0 ? 0 : (uint32_t)f); break; }
    case 'D': case 'd': break;                                       // legacy on/off duty: ignored
    case 'A': case 'a': { float d = atof(v); if (d < 0) d = 0; if (d > 1) d = 1; duty = d; if (running) rampTo(duty); break; }
    case 'C': case 'c': {                                            // restart only on a real change (the dashboard re-sends C)
      long h = atol(v);
      if (h >= 10000 && h <= 250000 && (uint32_t)h != carrierSet) { carrierSet = h; if (running) setFreq(fSet); }
      break;
    }
    case 'R': case 'r': { float r = atof(v); if (r < 0) r = 0; if (r > 1.5f) r = 1.5f; rampS = r; break; }
    case 'P': case 'p': { polFlip = atoi(v) ? 1 : 0; writeSeq(ampNow); break; }
    case '?': status(); break;
    default: break;
  }
}

void setup() {
  pinMode(PIN_MCTRL, INPUT);
  pinMode(PIN_NSLEEP, OUTPUT); digitalWrite(PIN_NSLEEP, LOW);
  pinsCoast();
  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r' || ch == '?') {
      if (ch == '?') { if (len < sizeof(line) - 1) line[len++] = '?'; }
      line[len] = 0;
      if (len) handle(line);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = ch;
    }
  }
}
