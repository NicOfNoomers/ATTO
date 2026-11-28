// Pump driver with carrier-PWM amplitude control, XIAO SAMD21 on the atto_drvr_dev_kit board (NM, 28 Nov 2026).
// Same command set as firmware/pump_driver_duty_nrf52 (see there for the why: capsule-driver note 12).
//
// HARDWARE CHANGE NEEDED FIRST: on the SAMD21 the board's IN1 pin, D0 = PA02, has no timer output. Solder a wire from
// D9 (PA05 = TCC0/WO[1]) to the D0 / IN1 net. This firmware keeps D0 as an INPUT (high impedance) so the two never
// fight, and drives IN1 from D9. Flash ONLY after the wire is in: without it IN1 floats (DRV8837 pull-down = low) and
// the coil gets one polarity only. (Alternative pin: D3 = PA11 = TCC0/WO[3], function F; set IN1_ON_D3 to 1.)
//
// Pins: IN2 = D1 (PA04 = TCC0/WO[0], function E), IN1 = D9 (PA05 = TCC0/WO[1], function E) via the wire,
// nSLEEP = D2 (PA10), D0 input, D10 (M_CTRL_V) input.
// TCC0 in single-slope PWM at the carrier (48 MHz / (PER + 1)), both outputs inverted: CC = L x (PER + 1) makes the
// pin low for the fraction L of each carrier period, then high. Forward half-cycle: IN1 high (CC 0), IN2 low for D
// (CC = D x (PER + 1)), high (brake) for 1 - D. Reverse: mirrored. D = 1: IN2 (or IN1) low all period = today's plain
// square wave. The overflow interrupt counts N carrier periods per half-cycle (N = carrier / (2 f)) and writes the
// buffered CCB values, so half-cycles change on carrier boundaries; PER is trimmed so that the pump frequency is exact.
//
// Commands (USB serial; the dashboard never reads replies):
//   "<hz>" / "F <hz>"  pump frequency, 0 = stop (coast, nSLEEP low)      "D <x>"  legacy on/off duty: ignored
//   "A <0..1>"  carrier duty (coil amplitude +-D x VM), default 1.0       "C <hz>" carrier 10000..250000, default 50000
//   "R <s>"     start/stop ramp 0..1.5 s, default 0                       "P <0|1>" polarity flip for a bench check
//   "?"         status
//   "T"         self-test while running: samples the IN1 / IN2 pins ~100000 times (PORT input buffer) and reports the
//               fraction of time each is high in the forward and the reverse half-cycle. Expected: forward IN1 100 %,
//               IN2 (1 - D) x 100 %; reverse mirrored. A mismatch means flip the polarity with "P 1".
#include <Arduino.h>
#include "wiring_private.h"

#define IN1_ON_D3 0
static const uint8_t PIN_IN2 = 1;                       // D1 = PA04, TCC0/WO[0] -> CC[0]
#if IN1_ON_D3
static const uint8_t PIN_IN1 = 3;                       // D3 = PA11, TCC0/WO[3] -> CC[3], function F
static const uint8_t CC_IN1 = 3;
#else
static const uint8_t PIN_IN1 = 9;                       // D9 = PA05, TCC0/WO[1] -> CC[1], function E
static const uint8_t CC_IN1 = 1;
#endif
static const uint8_t CC_IN2 = 0;
static const uint8_t PIN_D0 = 0, PIN_NSLEEP = 2, PIN_MCTRL = 10;
static const uint32_t TCC_CLK = 48000000UL;

static volatile uint32_t perP1 = 960;                   // PER + 1
static volatile uint32_t nHalf = 1, cnt = 0;
static volatile uint8_t fwd = 1;
static volatile float dutyNow = 1.0f;
static volatile uint8_t polFlip = 0;
static float duty = 1.0f, rampS = 0.0f;
static uint32_t carrierSet = 50000, fSet = 0;
static bool running = false;

static inline uint32_t ccLow(float d) {                 // CC giving "low for d of the period" (outputs inverted)
  if (polFlip) d = 1.0f - d;
  if (d >= 0.999f) return perP1;                        // never matches: low all period
  if (d <= 0.0f) return 0;
  return (uint32_t)(d * perP1 + 0.5f);
}

static inline void writeCCB() {
  uint32_t lo = ccLow(dutyNow), hi = ccLow(0.0f);
  if (fwd) { TCC0->CCB[CC_IN1].reg = hi; TCC0->CCB[CC_IN2].reg = lo; }
  else     { TCC0->CCB[CC_IN1].reg = lo; TCC0->CCB[CC_IN2].reg = hi; }
}

void TCC0_Handler() {
  TCC0->INTFLAG.reg = TCC_INTFLAG_OVF;
  if (++cnt >= nHalf) { cnt = 0; fwd ^= 1; }
  writeCCB();                                           // applied at the next UPDATE (end of this carrier period)
}

static void pinsCoast() {
  TCC0->CTRLA.bit.ENABLE = 0;
  while (TCC0->SYNCBUSY.bit.ENABLE) {}
  pinMode(PIN_IN1, OUTPUT); digitalWrite(PIN_IN1, LOW);
  pinMode(PIN_IN2, OUTPUT); digitalWrite(PIN_IN2, LOW);
}

static void tccStart(uint32_t f) {
  uint32_t n = (uint32_t)((float)carrierSet / (2.0f * f) + 0.5f);
  if (n < 1) n = 1;
  uint32_t p = (uint32_t)((float)TCC_CLK / (2.0f * n * f) + 0.5f);
  if (p < 48) p = 48;                                    // carrier <= 1 MHz
  if (p > 0xFFFFFF) p = 0xFFFFFF;
  perP1 = p; nHalf = n; cnt = 0; fwd = 1;
  PM->APBCMASK.reg |= PM_APBCMASK_TCC0;
  GCLK->CLKCTRL.reg = GCLK_CLKCTRL_CLKEN | GCLK_CLKCTRL_GEN_GCLK0 | GCLK_CLKCTRL_ID_TCC0_TCC1;
  while (GCLK->STATUS.bit.SYNCBUSY) {}
  TCC0->CTRLA.bit.ENABLE = 0;
  while (TCC0->SYNCBUSY.bit.ENABLE) {}
  TCC0->CTRLA.reg = TCC_CTRLA_PRESCALER_DIV1;
  TCC0->WAVE.reg = TCC_WAVE_WAVEGEN_NPWM;
  while (TCC0->SYNCBUSY.bit.WAVE) {}
  TCC0->DRVCTRL.reg = (1u << (TCC_DRVCTRL_INVEN_Pos + CC_IN2)) | (1u << (TCC_DRVCTRL_INVEN_Pos + CC_IN1));
  TCC0->PER.reg = perP1 - 1;
  while (TCC0->SYNCBUSY.bit.PER) {}
  uint32_t lo = ccLow(dutyNow), hi = ccLow(0.0f);
  TCC0->CC[CC_IN1].reg = hi; TCC0->CC[CC_IN2].reg = lo;
  while (TCC0->SYNCBUSY.reg & (TCC_SYNCBUSY_CC0 | TCC_SYNCBUSY_CC1 | TCC_SYNCBUSY_CC3)) {}
  writeCCB();
  TCC0->INTENSET.reg = TCC_INTENSET_OVF;
  NVIC_SetPriority(TCC0_IRQn, 0);
  NVIC_EnableIRQ(TCC0_IRQn);
#if IN1_ON_D3
  pinPeripheral(PIN_IN1, PIO_TIMER_ALT);
#else
  pinPeripheral(PIN_IN1, PIO_TIMER);
#endif
  pinPeripheral(PIN_IN2, PIO_TIMER);
  TCC0->CTRLA.bit.ENABLE = 1;
  while (TCC0->SYNCBUSY.bit.ENABLE) {}
}

static void rampTo(float target) {
  float from = dutyNow;
  if (rampS <= 0.0f || fabsf(target - from) < 0.005f) { dutyNow = target; return; }
  uint32_t steps = (uint32_t)(rampS * fabsf(target - from) * 200.0f) + 1;   // 5 ms steps, 0 -> 1 takes rampS
  for (uint32_t k = 1; k <= steps; k++) { dutyNow = from + (target - from) * (float)k / (float)steps; delay(5); }
}

static void setFreq(uint32_t f) {
  if (f == 0) {
    if (running) { rampTo(0.0f); NVIC_DisableIRQ(TCC0_IRQn); pinsCoast(); }
    digitalWrite(PIN_NSLEEP, LOW);
    running = false; fSet = 0;
    return;
  }
  if (f > 20000) f = 20000;
  digitalWrite(PIN_NSLEEP, HIGH);
  delayMicroseconds(60);                                 // DRV8837 wake-up
  bool was = running;
  if (was) { NVIC_DisableIRQ(TCC0_IRQn); }
  if (!was) dutyNow = (rampS > 0) ? 0.0f : duty;
  tccStart(f);
  running = true; fSet = f;
  if (dutyNow != duty) rampTo(duty);
}

static void status() {
  float fAct = running ? (float)TCC_CLK / (2.0f * nHalf * perP1) : 0.0f;
  Serial.print("PUMP_DRIVER_DUTY v1 mcu=samd21 in1=D"); Serial.print(PIN_IN1);
  Serial.print(" f="); Serial.print(fSet);
  Serial.print(" f_act="); Serial.print(fAct, 2);
  Serial.print(" D="); Serial.print(duty, 3);
  Serial.print(" D_now="); Serial.print((float)dutyNow, 3);
  Serial.print(" carrier="); Serial.print(carrierSet);
  Serial.print(" carrier_act="); Serial.print(running ? (float)TCC_CLK / perP1 : 0.0f, 0);
  Serial.print(" ramp_s="); Serial.print(rampS, 2);
  Serial.print(" pol="); Serial.print(polFlip);
  Serial.print(" running="); Serial.println(running ? 1 : 0);
}


static void selfTest() {
  if (!running) { Serial.println("TEST needs a running drive (send a frequency first)"); return; }
  const uint32_t b1 = (uint32_t)g_APinDescription[PIN_IN1].ulPin, b2 = (uint32_t)g_APinDescription[PIN_IN2].ulPin;
  PORT->Group[0].PINCFG[b1].bit.INEN = 1;
  PORT->Group[0].PINCFG[b2].bit.INEN = 1;
  uint32_t n[2] = {0, 0}, h1[2] = {0, 0}, h2[2] = {0, 0};
  for (uint32_t k = 0; k < 100000; k++) {
    uint8_t half = fwd;
    uint32_t in = PORT->Group[0].IN.reg;
    n[half]++; h1[half] += (in >> b1) & 1u; h2[half] += (in >> b2) & 1u;
    for (volatile uint8_t w = 0; w < (uint8_t)(k % 7); w++) {}       // de-correlate from the carrier
  }
  Serial.print("TEST D="); Serial.print((float)dutyNow, 3);
  Serial.print(" fwd: IN1 high "); Serial.print(n[1] ? 100.0f * h1[1] / n[1] : -1.0f, 1);
  Serial.print(" % IN2 high "); Serial.print(n[1] ? 100.0f * h2[1] / n[1] : -1.0f, 1);
  Serial.print(" % | rev: IN1 high "); Serial.print(n[0] ? 100.0f * h1[0] / n[0] : -1.0f, 1);
  Serial.print(" % IN2 high "); Serial.print(n[0] ? 100.0f * h2[0] / n[0] : -1.0f, 1);
  Serial.print(" % | expected fwd IN1 100, IN2 "); Serial.print(100.0f * (1.0f - (float)dutyNow), 1);
  Serial.println(" (rev mirrored)");
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
    case 'D': case 'd': break;
    case 'A': case 'a': { float d = atof(v); if (d < 0) d = 0; if (d > 1) d = 1; duty = d; if (running) rampTo(duty); break; }
    case 'C': case 'c': {
      long h = atol(v);
      if (h >= 10000 && h <= 250000 && (uint32_t)h != carrierSet) { carrierSet = h; if (running) setFreq(fSet); }
      break;
    }
    case 'R': case 'r': { float r = atof(v); if (r < 0) r = 0; if (r > 1.5f) r = 1.5f; rampS = r; break; }
    case 'P': case 'p': { polFlip = atoi(v) ? 1 : 0; break; }
    case '?': status(); break;
    case 'T': case 't': selfTest(); break;
    default: break;
  }
}

void setup() {
  pinMode(PIN_D0, INPUT);                               // wired to IN1 together with D9: never drive it
  pinMode(PIN_MCTRL, INPUT);
  pinMode(PIN_NSLEEP, OUTPUT); digitalWrite(PIN_NSLEEP, LOW);
  pinMode(PIN_IN1, OUTPUT); digitalWrite(PIN_IN1, LOW);
  pinMode(PIN_IN2, OUTPUT); digitalWrite(PIN_IN2, LOW);
  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r' || ch == '?') {
      if (ch == '?' && len < sizeof(line) - 1) line[len++] = '?';
      line[len] = 0;
      if (len) handle(line);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = ch;
    }
  }
}
