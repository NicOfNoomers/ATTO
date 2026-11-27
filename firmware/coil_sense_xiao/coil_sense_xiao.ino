// Coil-current probe on a Seeed XIAO SAMD21 (NM, 27 Nov 2026): differential ADC across a 1 ohm shunt in one coil lead.
//
// Why: the PPK2 sits on the supply side, after the driver board's capacitors, so the coil's waveform is smoothed to a
// 2-6 % ripple and cannot read the pump's gas state. Measured at the coil, the armature's motion (loaded by the gas
// inside) should show in the current waveform within each cycle.
//
// Wiring (use a SPARE XIAO, never the pump driver on COM3):
//   1 ohm shunt in ONE coil lead, short wires.
//   shunt end A -- 1 kohm -- XIAO A1 (PA04, AIN4)   (positive input)
//   shunt end B -- 1 kohm -- XIAO A0 (PA02, AIN0)   (negative input; must be AIN0..7)
//   XIAO GND -- driver board GND.  Power the XIAO (USB) BEFORE the drive starts.
//   Both shunt ends stay within 0..2.5 V (the bridge supply), inside the XIAO's 0..3.3 V input range.
//
// ADC: differential, 12 bit signed, internal 1.0 V bandgap reference, gain 8 (default) -> +-125 mV full scale,
// ~61 uV per count = ~61 uA per count through 1 ohm. Both inputs are sampled at the same instant. Free-running,
// polled into RAM; the sample rate is measured (micros) and reported with each capture.
//
// Serial (USB CDC, any baud):
//   c        capture N samples, reply: "CAP n=<N> us=<elapsed_us> gain=<g> ref_mv=1000 t_ms=<millis at start>\n",
//            then N int16 little-endian (2N bytes), then "\nEND\n"
//   n<N>     set N (100..12000), reply "OK n=<N>"
//   g<G>     set gain (1,2,4,8,16), reply "OK gain=<G>"
//   s<S>     set SAMPLEN (0..63; longer sampling for higher source impedance), reply "OK samplen=<S>"
//   ?        reply "COIL_SENSE_XIAO v1 n=<N> gain=<G> samplen=<S>"
#include <Arduino.h>
#include "wiring_private.h"

static const uint16_t NMAX = 12000;
static int16_t buf[NMAX];
static uint16_t nsamp = 10000;
static uint8_t gain = 8;
static uint8_t samplen = 1;

static void adcSync() { while (ADC->STATUS.bit.SYNCBUSY) {} }

static uint32_t gainBits(uint8_t g) {
  switch (g) {
    case 1: return ADC_INPUTCTRL_GAIN_1X;
    case 2: return ADC_INPUTCTRL_GAIN_2X;
    case 4: return ADC_INPUTCTRL_GAIN_4X;
    case 16: return ADC_INPUTCTRL_GAIN_16X;
    default: return ADC_INPUTCTRL_GAIN_8X;
  }
}

static void adcSetup() {
  pinPeripheral(A0, PIO_ANALOG);
  pinPeripheral(A1, PIO_ANALOG);
  SYSCTRL->VREF.reg |= SYSCTRL_VREF_BGOUTEN;          // bandgap for the 1.0 V reference
  ADC->CTRLA.bit.ENABLE = 0;
  adcSync();
  ADC->REFCTRL.reg = ADC_REFCTRL_REFSEL_INT1V;
  ADC->INPUTCTRL.reg = ADC_INPUTCTRL_MUXPOS_PIN4 | ADC_INPUTCTRL_MUXNEG_PIN0 | gainBits(gain);
  adcSync();
  // GCLK0 = 48 MHz (Arduino core) / 32 = 1.5 MHz ADC clock (max 2.1 MHz)
  ADC->CTRLB.reg = ADC_CTRLB_PRESCALER_DIV32 | ADC_CTRLB_RESSEL_12BIT | ADC_CTRLB_DIFFMODE | ADC_CTRLB_FREERUN;
  adcSync();
  ADC->SAMPCTRL.reg = ADC_SAMPCTRL_SAMPLEN(samplen);
  ADC->AVGCTRL.reg = 0;
  adcSync();
  ADC->CTRLA.bit.ENABLE = 1;
  adcSync();
  ADC->SWTRIG.bit.START = 1;
  adcSync();
}

static inline int16_t readSigned12() {
  while (!ADC->INTFLAG.bit.RESRDY) {}
  uint16_t r = ADC->RESULT.reg;                         // reading clears RESRDY
  int16_t v = (int16_t)(r & 0x0FFF);
  if (v & 0x0800) v |= (int16_t)0xF000;                 // sign-extend 12-bit two's complement
  return v;
}

static void capture() {
  for (uint8_t k = 0; k < 4; k++) (void)readSigned12(); // flush results from before the command
  uint32_t t_ms = millis();
  uint32_t t0 = micros();
  for (uint16_t i = 0; i < nsamp; i++) buf[i] = readSigned12();
  uint32_t us = micros() - t0;
  Serial.print("CAP n="); Serial.print(nsamp);
  Serial.print(" us="); Serial.print(us);
  Serial.print(" gain="); Serial.print(gain);
  Serial.print(" ref_mv=1000 t_ms="); Serial.print(t_ms);
  Serial.print("\n");
  Serial.write((const uint8_t *)buf, (size_t)nsamp * 2);
  Serial.print("\nEND\n");
}

void setup() {
  Serial.begin(115200);
  adcSetup();
}

void loop() {
  if (!Serial.available()) return;
  char c = Serial.read();
  if (c == 'c') {
    capture();
  } else if (c == 'n' || c == 'g' || c == 's') {
    long v = Serial.parseInt();
    if (c == 'n' && v >= 100 && v <= NMAX) { nsamp = (uint16_t)v; Serial.print("OK n="); Serial.println(nsamp); }
    else if (c == 'g' && (v == 1 || v == 2 || v == 4 || v == 8 || v == 16)) {
      gain = (uint8_t)v; adcSetup(); Serial.print("OK gain="); Serial.println(gain);
    } else if (c == 's' && v >= 0 && v <= 63) {
      samplen = (uint8_t)v; adcSetup(); Serial.print("OK samplen="); Serial.println(samplen);
    } else { Serial.println("ERR value"); }
  } else if (c == '?') {
    Serial.print("COIL_SENSE_XIAO v1 n="); Serial.print(nsamp);
    Serial.print(" gain="); Serial.print(gain);
    Serial.print(" samplen="); Serial.println(samplen);
  }
}
