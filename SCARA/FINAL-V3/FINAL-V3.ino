#include <Arduino.h>
#include <math.h>

// =====================================================
//  TYPE
// =====================================================
struct Waypoint {         
  float a, b, c;     // a(deg), b(cm), c(deg)
  bool  hasSpeed;
  float speed;       // 0..100
  bool  hasMag;
  int   mag;         // 0/1
};

// =====================================================
//  PROTOTYPES
// =====================================================
void setSpeedPercent(float p);
bool sameAsLastPose(float a, float b, float c);
void updateLastPose(float a, float b, float c);

bool checkLimitSafe(int pin);
void IRAM_ATTR handleEncoderA();
void stepOnce(int stepPin);

void motorForward2(int speed);
void motorBackward2(int speed);
void motorStop2();

void setMagnet(bool on);
void powerOffAll();
void stopAllMotionImmediate();

void homingJoint1_findSwitch_setZero();
void goHome1ByAngle();
void homingJoint3_160();
void homingAll3();

void updateStepper1();
void updateStepper3();
void updateAxis2();

void updateTestMode();
void handleSerial();

// ===== Xử lý Cảm biến băng tải =====
void handleConveyorSensor();

// ===== Kinematics (Pz in cm) =====
void FK_current(float &Px_mm, float &Py_mm, float &Pz_cm);
bool IK_solve(float Px_mm, float Py_mm, float Pz_cm, float &a_deg, float &b_cm, float &c_deg);

// ===== Trajectory profiles =====
void startJointQuinticProfile(float a_deg, float b_cm, float c_deg);
void startCartesianMinJerkProfile(float Px_t_mm, float Py_t_mm, float Pz_t_cm);
void updateProfiles();

// ===== Axis2 smoothing =====
static void updateAxis2SmoothTarget();
static inline void updateAxis2Gains();

// =====================================================
//  DONE / FEEDBACK helpers
// =====================================================
static inline void sendDone()     { Serial.println("DONE"); }
static inline void sendDoneStop() { Serial.println("DONE_STOP"); }
static inline void sendDone2()    { Serial.println("DONE_2"); }
// static inline void sendDone2() {
//   for (int i = 0; i < 5; i++) {
//     Serial.println("DONE_2");
//     delay(300);   // vẫn block nhưng ngắn
//     yield();      // nhường CPU (ESP32/ESP8266)
//   }
// }

// =====================================================
//  TUNING
// =====================================================
static const float DEFAULT_STEP_MM = 4.0f;

static const float MIN_STEP_RATE  = 2500.0f;
// FAST & STABLE
static const float MAX_STEP_RATE1 = 16000.0f;
static const float MAX_STEP_RATE3 = 16000.0f;

static const float MAX_VEL2_CM_S  = 10.0f;

static const uint32_t PROFILE_UPDATE_MS = 8;

static const float TRAJ_VMAX_MM_S   = 300.0f;
static const float TRAJ_AMAX_MM_S2  = 900.0f;

static const float MIN_TOTAL_TIME_S = 0.30f;

// ===== NEW: slew-rate limit for step-rate (giảm khựng/mất step) =====
static float runRate1 = MIN_STEP_RATE;
static float runRate3 = MIN_STEP_RATE;
// tăng accel vừa phải để lên tốc nhanh mà ít mất bước
static const float RATE_ACCEL_1 = 15000.0f;
static const float RATE_ACCEL_3 = 15000.0f;

// =====================================================
//  ROBOT STATE / HARDWARE
// =====================================================
bool isBusy = false;

// ===================== BẬC 1 – STEPPER =====================
#define STEP1_PIN   16
#define DIR1_PIN    17
#define ENA1_PIN    5
#define LIMIT1_PIN  14
#define stephome1   300
const float PULSES_1 = 35200.0f / 360.0f;

// ===================== BẬC 3 – STEPPER =====================
#define STEP3_PIN   18
#define DIR3_PIN    19
#define ENA3_PIN    21
#define LIMIT3_PIN  13
#define stephome3   150
const float PULSES_3 = (4.6f * 6400.0f) / 360.0f;// gear: 4.6, 6400 step/r

// ===== MAGNET =====
#define MAG_PIN 27

// ===== CONVEYOR CONFIG =====
#define CONVEYOR_SENSOR_PIN 23
#define CONVEYOR_RELAY_PIN  25

// debounce non-blocking
static int      conveyorStable = HIGH;
static int      conveyorLastRead = HIGH;
static uint32_t conveyorLastChangeMs = 0;
static const uint32_t CONVEYOR_DEBOUNCE_MS = 10;

// report DONE_STOP only on OFF edge (object detected => relay ON)
static bool conveyorDoneStopSent = false;

// ===== BACKOFF J1 =====
const float BACKOFF_ANGLE_J1_DEG = 3.0f;
const long  BACKOFF_STEPS_1      = (long)(BACKOFF_ANGLE_J1_DEG * PULSES_1 + 0.5f);

// ===== BACKOFF J3 =====
const float BACKOFF_ANGLE_J3_DEG = 163.0f;
const long  BACKOFF_STEPS_3      = (long)(BACKOFF_ANGLE_J3_DEG * PULSES_3 + 0.5f);

// ===================== BẬC 2 – DC + ENCODER =====================
const int pinA = 32;      // Encoder A
const int pinB = 33;      // Encoder B
const int LIMIT2_PIN = 12;

volatile long encoderValue = 0;
const float pulsesPerCm = (3200.0f) / 25.0f; // 128 pulses/cm

// BTS7960
const int RPWM = 4;
const int LPWM = 2;
const int EN   = 15;

// PID (global)
float Kp = 2.0f;
float Ki = 0.05f;
float Kd = 1.0f;

long  prevError = 0;
float integral = 0;
unsigned long lastPidTime = 0;

const int PID_INTERVAL = 10;

// PWM smoothing
static int lastPwmCmd = 0;
static const int PWM_SLEW_PER_TICK = 10;
int minPWM = 45;

// Axis2 setpoint smoothing (slew-rate)
static long     target2_smooth = 0;
static uint32_t lastSmoothMs   = 0;
static const float AXIS2_VEL_LIMIT_CM_S = 6.0f;

// ===== NEW: Axis2 non-blocking backoff state machine =====
enum Axis2BackoffState : uint8_t { B2_BO_NONE=0, B2_BO_RUN=1 };
static Axis2BackoffState b2_backoff = B2_BO_NONE;
static uint32_t b2_backoffStartMs = 0;
static const uint32_t B2_BACKOFF_TIMEOUT_MS = 2000;

// ===================== TRẠNG THÁI ĐỘNG HỌC =====================
float currentAngle1 = 0.0f;
float currentAngle3 = 0.0f;

bool joint1Homed = false;

static long posSteps1 = 0;
static long posSteps3 = 0;

// Axis2 target pulses
long targetPulses2 = 0;

// Flags chạy
bool moving1 = false;
bool moving2 = false;
bool moving3 = false;

bool homing2Active = false;
uint8_t homing2Stage = 0;

const int HOMING2_SPEED_UP   = 100;
const int HOMING2_SPEED_DOWN = 35;

// Deadband axis2
const float CM_DEADBAND = 0.20f;

// speed
int   speedPercent = 100;
float speedScale   = 1.0f;

// remember last pose
bool  hasLastPose = false;
float lastA = 0.0f, lastB = 0.0f, lastC = 0.0f;

// TEST
const int NUM_TEST_POSES = 5;
const float testA[NUM_TEST_POSES] = { 90.0f, 180.0f, 270.0f, 180.0f, 0.0f };
const float testB[NUM_TEST_POSES] = {  5.0f,  10.0f,  15.0f,   5.0f, 0.0f };
const float testC[NUM_TEST_POSES] = { 90.0f,   0.0f, 180.0f,  90.0f, 0.0f };

bool testModeActive = false;
int  testIndex      = 0;

// =====================================================
//  KINEMATICS (Px,Py mm; Pz cm)
// =====================================================
const float L2 = 228.0f;
const float L4 = 144.0f;

const float L1 = 45.0f;
const float L3 = 63.5f;
const float L5 = 30.0f;

const float Z_HOME_OFFSET_CM = (L1 - L3 - L5); // -48.5
// dộng học thuận
void FK_current(float &Px_mm, float &Py_mm, float &Pz_cm) {
  float th1 = currentAngle1 * (PI / 180.0f);
  float th3 = currentAngle3 * (PI / 180.0f);

  float b_cm = (float)encoderValue / pulsesPerCm;

  Px_mm = L4 * cosf(th1 + th3) + L2 * cosf(th1);
  Py_mm = L4 * sinf(th1 + th3) + L2 * sinf(th1);

  Pz_cm = b_cm + Z_HOME_OFFSET_CM;
}

bool IK_solve(float Px_mm, float Py_mm, float Pz_cm, float &a_deg, float &b_cm, float &c_deg) {
  b_cm = Pz_cm - Z_HOME_OFFSET_CM;

  float r2 = Px_mm*Px_mm + Py_mm*Py_mm;
  float c3 = (r2 - L4*L4 - L2*L2) / (2.0f * L2 * L4);

  if (c3 < -1.0005f || c3 > 1.0005f) return false;
  c3 = fmaxf(-1.0f, fminf(1.0f, c3));

  float s3 = sqrtf(fmaxf(0.0f, 1.0f - c3*c3));
  float th3_1 = atan2f( s3, c3);
  float th3_2 = atan2f(-s3, c3);

  auto calc_th1 = [&](float th3)->float {
    float A = L2 + L4 * cosf(th3);
    float B = L4 * sinf(th3);
    return atan2f(Py_mm, Px_mm) - atan2f(B, A);
  };

  float th1_1 = calc_th1(th3_1);
  float th1_2 = calc_th1(th3_2);

  float cur3 = currentAngle3 * (PI / 180.0f);
  float d1 = fabsf(th3_1 - cur3);
  float d2 = fabsf(th3_2 - cur3);

  float th1 = (d1 <= d2) ? th1_1 : th1_2;
  float th3 = (d1 <= d2) ? th3_1 : th3_2;

//Chuyển đổi đơn vị từ Radian sang Độ
  a_deg = th1 * 180.0f / PI;
  c_deg = th3 * 180.0f / PI;
  return true;
}

// =====================================================
//  HELPERS
// =====================================================
void setSpeedPercent(float p) {
  if (p < 0)   p = 0;
  if (p > 100) p = 100;
  speedPercent = (int)p;
  speedScale   = speedPercent / 100.0f;

  Serial.print(">> Set speed = ");
  Serial.print(speedPercent);
  Serial.println(" %");
}

bool sameAsLastPose(float a, float b, float c) {
  if (!hasLastPose) return false;
  const float EPS_A = 0.5f;
  const float EPS_B = 0.2f;
  const float EPS_C = 0.5f;
  return (fabs(a - lastA) < EPS_A && fabs(b - lastB) < EPS_B && fabs(c - lastC) < EPS_C);
}

void updateLastPose(float a, float b, float c) {
  lastA = a; lastB = b; lastC = c;
  hasLastPose = true;
}

bool checkLimitSafe(int pin) {
  if (digitalRead(pin) == LOW) {
    delayMicroseconds(50);
    if (digitalRead(pin) == LOW) {
      delayMicroseconds(50);
      if (digitalRead(pin) == LOW) return true;
    }
  }
  return false;
}

void stepOnce(int stepPin) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(3);
  digitalWrite(stepPin, LOW);
}

void IRAM_ATTR handleEncoderA() {
  if (digitalRead(pinB) == HIGH) encoderValue++;
  else encoderValue--;
}

void motorForward2(int speed) {
  if (speed > 255) speed = 255;
  analogWrite(RPWM, speed);
  analogWrite(LPWM, 0);
  digitalWrite(EN, HIGH);
}
void motorBackward2(int speed) {
  if (speed > 255) speed = 255;
  analogWrite(RPWM, 0);
  analogWrite(LPWM, speed);
  digitalWrite(EN, HIGH);
}
void motorStop2() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
  digitalWrite(EN, LOW);
}

// =====bật/tắt nam châm trả DONE =====
void setMagnet(bool on) {
  digitalWrite(MAG_PIN, on ? HIGH : LOW);
  sendDone();
}

static inline bool robotIdle() {
  return (!moving1 && !moving2 && !moving3 && !homing2Active && b2_backoff==B2_BO_NONE);
}

void powerOffAll() {
  if (robotIdle()) {
    digitalWrite(ENA1_PIN, LOW);
    digitalWrite(ENA3_PIN, LOW);
    motorStop2();
  }
}

// =====================================================
//  EMERGENCY POLL 
//  during homing it may "consume" other commands; emergency has priority.
// =====================================================
static bool pollEmergencyNow() {
  if (!Serial.available()) return false;
  String s = Serial.readStringUntil('\n');
  s.trim();
  s.toLowerCase();
  if (s == "emergency") {
    testModeActive = false;
    stopAllMotionImmediate();
    powerOffAll();
    Serial.println(">> EMERGENCY RECEIVED (forced stop)");
    return true;
  }
  return false;
}

// =====================================================
//  PROFILE: minimum-jerk (quintic) core
// QHQD bậc 5
// =====================================================
static inline float clamp01(float x) { return (x < 0) ? 0 : (x > 1 ? 1 : x); }

static inline float minJerk_u(float tau) {
  tau = clamp01(tau);
  float t2 = tau * tau;
  float t3 = t2 * tau;
  float t4 = t3 * tau;
  float t5 = t4 * tau;
  return 10.0f*t3 - 15.0f*t4 + 6.0f*t5;
}

static inline float minJerk_du(float tau) {
  tau = clamp01(tau);
  float t2 = tau*tau;
  float t3 = t2*tau;
  float t4 = t3*tau;
  return 30.0f*t2 - 60.0f*t3 + 30.0f*t4;
}

// =====================================================
//  PROFILE STATE
// =====================================================
enum ProfileMode : uint8_t { PROF_NONE=0, PROF_JOINT=1, PROF_CART=2 };// đứng yên, không gian khớp, khong gian làm việc
static ProfileMode profMode = PROF_NONE;

static uint32_t profStartMs = 0;
static uint32_t profLastUpdateMs = 0;
static float    profT_s = 0.0f;

// JOINT PROFILE
static long  prof_q1_0 = 0, prof_q1_1 = 0;
static long  prof_q3_0 = 0, prof_q3_1 = 0;
static long  prof_p2_0 = 0, prof_p2_1 = 0;

// CART PROFILE
static float prof_Px0=0, prof_Py0=0, prof_Pz0=0;
static float prof_Pxt=0, prof_Pyt=0, prof_Pzt=0;

// desired setpoints
static volatile long  desSteps1 = 0;
static volatile long  desSteps3 = 0;
static volatile long  desPulses2 = 0;

// desired rates (target)
static volatile float desRate1 = MIN_STEP_RATE;
static volatile float desRate3 = MIN_STEP_RATE;

// stepper timing// tính thời gian để phát xung, không dùng delay vì gây treo hệ thống 
static uint32_t lastStepTime1 = 0;
static uint32_t lastStepTime3 = 0;
static uint32_t stepInterval1 = 200;
static uint32_t stepInterval3 = 200;

// =====================================================
//  STOP (emergency)
// =====================================================
void stopAllMotionImmediate() {
  profMode = PROF_NONE;
  profT_s = 0;
  profStartMs = profLastUpdateMs = 0;

  moving1 = moving2 = moving3 = false;
  homing2Active = false;
  homing2Stage = 0;

  b2_backoff = B2_BO_NONE;

  prevError = 0;
  integral = 0;
  lastPwmCmd = 0;

  motorStop2();
  digitalWrite(ENA1_PIN, LOW);
  digitalWrite(ENA3_PIN, LOW);

  // setMagnet(false) sẽ tự in DONE -> emergency không cần DONE nữa
  digitalWrite(MAG_PIN, LOW);

  target2_smooth = encoderValue;
  targetPulses2  = encoderValue;

  // reset run rates
  runRate1 = MIN_STEP_RATE;
  runRate3 = MIN_STEP_RATE;

  // Serial.println(">> STOP ALL MOTION (immediate) & RESET PID");
}

// =====================================================
//  HOMING
// =====================================================
void homingJoint1_findSwitch_setZero() {
  Serial.println("=== HOME1: find switch -> set 0deg ===");
  pinMode(LIMIT1_PIN, INPUT_PULLUP);
  digitalWrite(ENA1_PIN, HIGH);
  digitalWrite(DIR1_PIN, HIGH);

  if (checkLimitSafe(LIMIT1_PIN)) {
    digitalWrite(ENA1_PIN, LOW);
    currentAngle1 = 0.0f;
    posSteps1 = 0;
    hasLastPose = false;
    joint1Homed = true;
    Serial.println("HOME1: already on switch -> set 0deg");
    return;
  }

  while (!checkLimitSafe(LIMIT1_PIN)) {
    if (pollEmergencyNow()) return;     // <<< EMERGENCY ALWAYS WORKS
    stepOnce(STEP1_PIN);
    delayMicroseconds(400);
  }

  delay(50);
  digitalWrite(ENA1_PIN, LOW);

  currentAngle1 = 0.0f;
  posSteps1 = 0;
  hasLastPose = false;
  joint1Homed = true;

  Serial.println("HOME1 DONE: set 0deg at switch");
}

void goHome1ByAngle() {
  if (!joint1Homed) {
    homingJoint1_findSwitch_setZero();
    return;
  }
  if (fabs(currentAngle1) < 0.001f) {
    Serial.println("HOME1: already at 0deg");
    return;
  }
  startJointQuinticProfile(0.0f, (float)encoderValue / pulsesPerCm, currentAngle3);
}

void homingJoint1() {
  Serial.println("=== HOME1: hit switch then backoff deg -> set 0deg ===");
  pinMode(LIMIT1_PIN, INPUT_PULLUP);
  digitalWrite(ENA1_PIN, HIGH);

  digitalWrite(DIR1_PIN, HIGH);
  while (!checkLimitSafe(LIMIT1_PIN)) {
    if (pollEmergencyNow()) return;     // <<< ADD
    stepOnce(STEP1_PIN);
    delayMicroseconds(stephome1);
  }
  delay(100);

  digitalWrite(DIR1_PIN, LOW);
  for (long k = 0; k < BACKOFF_STEPS_1; k++) {
    if (pollEmergencyNow()) return;     // <<< ADD
    stepOnce(STEP1_PIN);
    delayMicroseconds(stephome1);
  }

  digitalWrite(ENA1_PIN, LOW);
  currentAngle1 = 0.0f;
  posSteps1 = 0;
  hasLastPose = false;

  joint1Homed = true;
  Serial.println("HOME1 DONE: set 0deg");
}

void homingJoint3_160() {
  Serial.println("=== HOME3: hit switch then backoff 163deg -> set 0deg ===");
  pinMode(LIMIT3_PIN, INPUT_PULLUP);
  digitalWrite(ENA3_PIN, HIGH);

  digitalWrite(DIR3_PIN, LOW);
  while (!checkLimitSafe(LIMIT3_PIN)) {
    if (pollEmergencyNow()) return;     // <<< ADD
    stepOnce(STEP3_PIN);
    delayMicroseconds(stephome3);
  }
  delay(100);

  digitalWrite(DIR3_PIN, HIGH);
  for (long k = 0; k < BACKOFF_STEPS_3; k++) {
    if (pollEmergencyNow()) return;     // <<< ADD
    stepOnce(STEP3_PIN);
    delayMicroseconds(stephome3);
  }

  digitalWrite(ENA3_PIN, LOW);
  currentAngle3 = 0.0f;
  posSteps3 = 0;
  hasLastPose = false;

  Serial.println("HOME3 DONE: set 0deg");
}

void homingAll3() {
  Serial.println("=== HOMEALL (2 -> 1 -> 3) ===");

  // không gọi setMagnet ở đây để khỏi in DONE
  digitalWrite(MAG_PIN, LOW);

  pinMode(LIMIT2_PIN, INPUT_PULLUP);

  motorBackward2(HOMING2_SPEED_UP);
  while (!checkLimitSafe(LIMIT2_PIN)) {
    if (pollEmergencyNow()) { motorStop2(); return; }  // <<< ADD
    delay(5);
  }
  motorStop2();
  encoderValue = 0;
  delay(100);

  motorForward2(HOMING2_SPEED_DOWN);
  while (digitalRead(LIMIT2_PIN) == LOW) {
    if (pollEmergencyNow()) { motorStop2(); return; }  // <<< ADD
    delay(5);
  }
  motorStop2();

  Serial.println(">>> Axis2 HOMED");

  target2_smooth = encoderValue;
  targetPulses2  = encoderValue;
  lastSmoothMs   = millis();

  homingJoint1();
  homingJoint3_160();

  hasLastPose = false;
  prevError = 0;
  integral = 0;
  lastPwmCmd = 0;

  runRate1 = MIN_STEP_RATE;
  runRate3 = MIN_STEP_RATE;

  Serial.println("=== HOMEALL DONE ===");
}

// =====================================================
//  PROFILE STARTERS
// =====================================================
static float computeJointDuration(float a_deg, float b_cm, float c_deg) {
  long s1 = labs(lroundf(a_deg * PULSES_1) - posSteps1);
  long s3 = labs(lroundf(c_deg * PULSES_3) - posSteps3);
  long p2 = labs(lroundf(b_cm * pulsesPerCm) - encoderValue);

  float v2 = fmaxf(0.5f, MAX_VEL2_CM_S * speedScale);
  float t2 = (p2 / pulsesPerCm) / v2;

  float maxR1 = fmaxf(500.0f, MAX_STEP_RATE1 * speedScale);
  float maxR3 = fmaxf(500.0f, MAX_STEP_RATE3 * speedScale);

  float t1 = (s1 > 0) ? (1.4f * (float)s1 / maxR1) : 0.0f;
  float t3 = (s3 > 0) ? (1.4f * (float)s3 / maxR3) : 0.0f;

  float T = fmaxf(t2, fmaxf(t1, t3));
  if (T < MIN_TOTAL_TIME_S) T = MIN_TOTAL_TIME_S;
  return T;
}

void startJointQuinticProfile(float a_deg, float b_cm, float c_deg) {
  if (!joint1Homed) { Serial.println("Need home1/homeall first."); return; }
  if (!robotIdle()) { Serial.println("Busy."); return; }

  moving1 = moving2 = moving3 = true;
  isBusy = true;

  profT_s = computeJointDuration(a_deg, b_cm, c_deg);

  prof_q1_0 = posSteps1;
  prof_q1_1 = lroundf(a_deg * PULSES_1);

  prof_q3_0 = posSteps3;
  prof_q3_1 = lroundf(c_deg * PULSES_3);

  prof_p2_0 = encoderValue;
  prof_p2_1 = lroundf(b_cm * pulsesPerCm);

  desSteps1  = prof_q1_0;
  desSteps3  = prof_q3_0;
  desPulses2 = prof_p2_0;

  targetPulses2  = encoderValue;
  target2_smooth = encoderValue;

  integral = 0;
  prevError = 0;
  lastPwmCmd = 0;

  lastSmoothMs   = millis();

  desRate1 = MIN_STEP_RATE;
  desRate3 = MIN_STEP_RATE;

  runRate1 = MIN_STEP_RATE;
  runRate3 = MIN_STEP_RATE;

  digitalWrite(ENA1_PIN, HIGH);
  digitalWrite(ENA3_PIN, HIGH);

  profMode = PROF_JOINT;
  profStartMs = millis();
  profLastUpdateMs = profStartMs;

  lastStepTime1 = micros();
  lastStepTime3 = micros();

  Serial.print("JOINT PROFILE start, T(s)=");
  Serial.println(profT_s, 3);
}

static float computeCartesianDuration(float Px0, float Py0, float Pz0_cm,
                                      float Pxt, float Pyt, float Pzt_cm) {
  float dx = Pxt - Px0;
  float dy = Pyt - Py0;
  float dz = (Pzt_cm - Pz0_cm) * 10.0f; // cm->mm
  float L = sqrtf(dx*dx + dy*dy + dz*dz);
  if (L < 0.01f) return MIN_TOTAL_TIME_S;

  float V = fmaxf(20.0f, TRAJ_VMAX_MM_S  * speedScale);
  float A = fmaxf(80.0f, TRAJ_AMAX_MM_S2 * speedScale);

  float T1 = (1.875f * L) / V;
  float T2 = sqrtf((5.7735f * L) / A);
  float T  = fmaxf(T1, T2);
  if (T < MIN_TOTAL_TIME_S) T = MIN_TOTAL_TIME_S;
  return T;
}

void startCartesianMinJerkProfile(float Px_t_mm, float Py_t_mm, float Pz_t_cm) {
  if (!joint1Homed) { Serial.println("Need home1/homeall first."); return; }
  if (!robotIdle()) { Serial.println("Busy."); return; }

  float Px0_mm, Py0_mm, Pz0_cm;
  FK_current(Px0_mm, Py0_mm, Pz0_cm);

  prof_Px0 = Px0_mm; prof_Py0 = Py0_mm; prof_Pz0 = Pz0_cm;
  prof_Pxt = Px_t_mm; prof_Pyt = Py_t_mm; prof_Pzt = Pz_t_cm;

  profT_s = computeCartesianDuration(prof_Px0, prof_Py0, prof_Pz0, prof_Pxt, prof_Pyt, prof_Pzt);

  moving1 = moving2 = moving3 = true;
  isBusy = true;

  digitalWrite(ENA1_PIN, HIGH);
  digitalWrite(ENA3_PIN, HIGH);

  desSteps1  = posSteps1;
  desSteps3  = posSteps3;
  desPulses2 = encoderValue;

  targetPulses2  = encoderValue;
  target2_smooth = encoderValue;

  integral = 0;
  prevError = 0;
  lastPwmCmd = 0;

  lastSmoothMs   = millis();

  desRate1 = MIN_STEP_RATE;
  desRate3 = MIN_STEP_RATE;

  runRate1 = MIN_STEP_RATE;
  runRate3 = MIN_STEP_RATE;

  profMode = PROF_CART;
  profStartMs = millis();
  profLastUpdateMs = profStartMs;

  lastStepTime1 = micros();
  lastStepTime3 = micros();

  Serial.print("CART PROFILE (L) start, T(s)=");
  Serial.println(profT_s, 3);
}

// =====================================================
//  PROFILE UPDATE
// =====================================================
static void finishProfile() {
  profMode = PROF_NONE;

  currentAngle1 = (float)posSteps1 / PULSES_1;
  currentAngle3 = (float)posSteps3 / PULSES_3;

  moving1 = moving2 = moving3 = false;

  powerOffAll();

  // ===== yêu cầu: chạy xong trả DONE =====
  sendDone();

  isBusy = false;
}

void updateProfiles() {
  if (profMode == PROF_NONE) return;

  uint32_t nowMs = millis();
  if (nowMs - profLastUpdateMs < PROFILE_UPDATE_MS) return;
  profLastUpdateMs = nowMs;

  float t = (nowMs - profStartMs) / 1000.0f;
  if (t < 0) t = 0;
  if (t > profT_s) t = profT_s;

  float tau = (profT_s > 1e-6f) ? (t / profT_s) : 1.0f;
  float u   = minJerk_u(tau);
  float du  = minJerk_du(tau);

  if (profMode == PROF_JOINT) {
    long q1 = lroundf((float)prof_q1_0 + (float)(prof_q1_1 - prof_q1_0) * u);
    long q3 = lroundf((float)prof_q3_0 + (float)(prof_q3_1 - prof_q3_0) * u);
    long p2 = lroundf((float)prof_p2_0 + (float)(prof_p2_1 - prof_p2_0) * u);

    desSteps1  = q1;
    desSteps3  = q3;
    desPulses2 = p2;

    float invT = (profT_s > 1e-6f) ? (1.0f / profT_s) : 0.0f;
    float q1dot = fabsf((float)(prof_q1_1 - prof_q1_0) * du * invT);
    float q3dot = fabsf((float)(prof_q3_1 - prof_q3_0) * du * invT);

    if (q1dot < MIN_STEP_RATE) q1dot = MIN_STEP_RATE;
    if (q3dot < MIN_STEP_RATE) q3dot = MIN_STEP_RATE;
    if (q1dot > MAX_STEP_RATE1) q1dot = MAX_STEP_RATE1;
    if (q3dot > MAX_STEP_RATE3) q3dot = MAX_STEP_RATE3;

    desRate1 = q1dot;
    desRate3 = q3dot;

  } else if (profMode == PROF_CART) {
    float Px = prof_Px0 + (prof_Pxt - prof_Px0) * u;
    float Py = prof_Py0 + (prof_Pyt - prof_Py0) * u;
    float Pz = prof_Pz0 + (prof_Pzt - prof_Pz0) * u;

    float a_deg, b_cm, c_deg;
    if (!IK_solve(Px, Py, Pz, a_deg, b_cm, c_deg)) {
      Serial.println("L: IK fail -> STOP");
      stopAllMotionImmediate();
      return;
    }
//====Hàm lroundf (Làm tròn số)
    long q1 = lroundf(a_deg * PULSES_1);
    long q3 = lroundf(c_deg * PULSES_3);
    long p2 = lroundf(b_cm * pulsesPerCm);

    desSteps1  = q1;
    desSteps3  = q3;
    desPulses2 = p2;

    static long lastQ1 = 0, lastQ3 = 0;
    static bool hasLast = false;
    float dt = PROFILE_UPDATE_MS / 1000.0f;
    if (!hasLast) { lastQ1 = q1; lastQ3 = q3; hasLast = true; }

    float r1 = fabsf((float)(q1 - lastQ1)) / fmaxf(1e-3f, dt);
    float r3 = fabsf((float)(q3 - lastQ3)) / fmaxf(1e-3f, dt);
    lastQ1 = q1; lastQ3 = q3;

    if (r1 < MIN_STEP_RATE) r1 = MIN_STEP_RATE;
    if (r3 < MIN_STEP_RATE) r3 = MIN_STEP_RATE;
    if (r1 > MAX_STEP_RATE1) r1 = MAX_STEP_RATE1;
    if (r3 > MAX_STEP_RATE3) r3 = MAX_STEP_RATE3;

    desRate1 = r1;
    desRate3 = r3;
  }

  // slew desRate -> runRate
  {
    float dt = PROFILE_UPDATE_MS / 1000.0f;

    float maxD1 = RATE_ACCEL_1 * dt;// Giới hạn thay đổi tốc độ trong 8ms
    float d1 = desRate1 - runRate1;// Chênh lệch giữa Mong muốn và Thực tế
    if (d1 >  maxD1) d1 =  maxD1;// không cho tăng quá nhanh
    if (d1 < -maxD1) d1 = -maxD1;//k cho giảm quá nhanh
    runRate1 += d1;//Cập nhật tốc độ thực tế

    float maxD3 = RATE_ACCEL_3 * dt;
    float d3 = desRate3 - runRate3;
    if (d3 >  maxD3) d3 =  maxD3;
    if (d3 < -maxD3) d3 = -maxD3;
    runRate3 += d3;

    runRate1 = fmaxf(MIN_STEP_RATE, fminf(MAX_STEP_RATE1, runRate1));
    runRate3 = fmaxf(MIN_STEP_RATE, fminf(MAX_STEP_RATE3, runRate3));
  }

  if (t >= profT_s) {
    bool ok1 = (labs(desSteps1 - posSteps1) <= 0);
    bool ok3 = (labs(desSteps3 - posSteps3) <= 0);
    bool ok2 = (labs(desPulses2 - encoderValue) <= (long)(CM_DEADBAND * pulsesPerCm));
    if (ok1 && ok3 && ok2) finishProfile();
  }
}

// =====================================================
//  Axis2 setpoint smoothing (slew-rate)
// =====================================================
static void updateAxis2SmoothTarget() {
  uint32_t now = millis();
  float dt = (now - lastSmoothMs) / 1000.0f;
  if (dt <= 0) return;
  lastSmoothMs = now;

  long goal = (profMode != PROF_NONE) ? (long)desPulses2 : targetPulses2;

  float v_cm_s = AXIS2_VEL_LIMIT_CM_S * speedScale;

  float diffCm = fabsf((goal - target2_smooth) / pulsesPerCm);
  if (diffCm < 4.0f) v_cm_s *= 0.5f;
  if (diffCm < 2.0f) v_cm_s *= 0.4f;

  float maxStep = v_cm_s * pulsesPerCm * dt;
  if (maxStep < 1.0f) maxStep = 1.0f;

  long diff = goal - target2_smooth;
  if (labs(diff) <= (long)maxStep) target2_smooth = goal;
  else target2_smooth += (diff > 0) ? (long)maxStep : -(long)maxStep;

  targetPulses2 = target2_smooth;
}

// =====================================================
//  STEPPER UPDATE
// =====================================================
void updateStepper1() {
  if (!moving1) return;

//so sánh vị trí mong muốn (desSteps1) với vị trí hiện tại (posSteps1).
  long err = desSteps1 - posSteps1;
  if (err == 0) return;
  if (err > 0) digitalWrite(DIR1_PIN, HIGH);//quay thuận
  else         digitalWrite(DIR1_PIN, LOW);//quay ngược

  float rate = runRate1;
  if (rate < MIN_STEP_RATE) rate = MIN_STEP_RATE;
  if (rate > MAX_STEP_RATE1) rate = MAX_STEP_RATE1;
  stepInterval1 = (uint32_t)(1000000.0f / rate);

  uint32_t now = micros();
  if (now - lastStepTime1 >= stepInterval1) {
    lastStepTime1 = now;
    stepOnce(STEP1_PIN);
    if (err > 0) posSteps1++;
    else         posSteps1--;
    currentAngle1 = (float)posSteps1 / PULSES_1;// đổi từ bước sang độ
  }
}

void updateStepper3() {
  if (!moving3) return;

  long err = desSteps3 - posSteps3;
  if (err == 0) return;

  if (err > 0) digitalWrite(DIR3_PIN, HIGH);
  else         digitalWrite(DIR3_PIN, LOW);

  float rate = runRate3;
  if (rate < MIN_STEP_RATE) rate = MIN_STEP_RATE;
  if (rate > MAX_STEP_RATE3) rate = MAX_STEP_RATE3;
  stepInterval3 = (uint32_t)(1000000.0f / rate);

  uint32_t now = micros();
  if (now - lastStepTime3 >= stepInterval3) {
    lastStepTime3 = now;
    stepOnce(STEP3_PIN);
    if (err > 0) posSteps3++;
    else         posSteps3--;
    currentAngle3 = (float)posSteps3 / PULSES_3;
  }
}

// =====================================================
//  PID gain schedule
// =====================================================
static inline void updateAxis2Gains() {
  if (speedScale <= 0.4f) {
    Kp = 3.0f;   Ki = 0.05f;    Kd = 1.0f;//Tốc độ thấp (dưới 40%).
  } else if (speedScale <= 0.7f) {
    Kp = 2.0f;   Ki = 0.0015f;  Kd = 0.5f;//Tốc độ vừa phải (40% - 70%).
  } else {
    Kp = 1.5f;   Ki = 0.0010f;  Kd = 0.08f;//Tốc độ cao (trên 70%).
  }
}

// =====================================================
//  AXIS2 PID UPDATE (NON-BLOCKING backoff)
// =====================================================
void updateAxis2() {
  if (b2_backoff == B2_BO_RUN) {
    if (digitalRead(LIMIT2_PIN) == HIGH) {
      motorStop2();
      b2_backoff = B2_BO_NONE;
      //reset PID
      target2_smooth = encoderValue;
      targetPulses2  = encoderValue;
      desPulses2     = encoderValue;

      prevError = 0;
      integral = 0;
      lastPwmCmd = 0;
      return;
    }
    if (millis() - b2_backoffStartMs > B2_BACKOFF_TIMEOUT_MS) {
      motorStop2();
      b2_backoff = B2_BO_NONE;//kiểm tra đã lùi quá thời gian quy định chưa
      Serial.println("Axis2 backoff TIMEOUT!");//nếu quá thì ép phải dừng
      return;
    }
    return;
  }

  if (homing2Active) {
    if (homing2Stage == 1) {
      if (checkLimitSafe(LIMIT2_PIN)) {
        motorStop2(); encoderValue = 0; delay(50);
        homing2Stage = 2; motorForward2(HOMING2_SPEED_DOWN);
      }//Trục 2 đang di chuyển về phía công tắc hành trình
    } else if (homing2Stage == 2) {
      if (digitalRead(LIMIT2_PIN) == HIGH) {
        motorStop2();
        homing2Active = false;
        moving2 = false;
        encoderValue = 0;

        target2_smooth = encoderValue;
        targetPulses2  = encoderValue;

        //Serial.println("Axis2 homed.");
        powerOffAll();
        sendDone2();
      }
    }
    return;
  }

  if (!moving2 && profMode == PROF_NONE) return;

  unsigned long now = millis();
  if (now - lastPidTime < (unsigned long)PID_INTERVAL) return;
  lastPidTime = now;

  updateAxis2Gains();

  long currentPulses = encoderValue;
  long error = targetPulses2 - currentPulses;
  float errorCm = error / pulsesPerCm;

//sai số nằm trong khoảng này (-0.05 đến +0.05), coi như Về đích
  if (fabs(errorCm) <= CM_DEADBAND) {
    motorStop2();
    prevError = 0;
    integral = 0;
    lastPwmCmd = 0;
    if (profMode == PROF_NONE) moving2 = false;
    return;
  }

  moving2 = true;

  float P = Kp * (float)error;

  static uint32_t lastI_ms = 0;
  uint32_t nowI = millis();
  float dt = (nowI - lastI_ms) / 1000.0f;
  if (dt <= 0) dt = PID_INTERVAL / 1000.0f;
  lastI_ms = nowI;

  float errAbsCm = fabsf(errorCm);

  if (errAbsCm < 4.0f) integral += (float)error * dt;
  else                 integral *= 0.98f;

  integral = fmaxf(-300.0f, fminf(300.0f, integral));
  float I = Ki * integral;

  float derr = (float)(error - prevError);
  float D = Kd * (derr / fmaxf(dt, 1e-3f));
  prevError = error;

  float output = P + I + D;

  int maxPWM = (int)(255.0f * speedScale);
  maxPWM = (int)fminf(255.0f, maxPWM * 1.10f);
  if (maxPWM < minPWM) maxPWM = minPWM;

  int pwmVal = (int)output;
  if (pwmVal >  maxPWM) pwmVal =  maxPWM;
  if (pwmVal < -maxPWM) pwmVal = -maxPWM;

  int hi = lastPwmCmd + PWM_SLEW_PER_TICK;
  int lo = lastPwmCmd - PWM_SLEW_PER_TICK;
  if (pwmVal > hi) pwmVal = hi;
  if (pwmVal < lo) pwmVal = lo;
  lastPwmCmd = pwmVal;

  if (pwmVal > 0) {
    if (pwmVal < minPWM) pwmVal = minPWM;
    motorForward2(pwmVal);
  }
  else if (pwmVal < 0) {
    if (checkLimitSafe(LIMIT2_PIN)) {
      motorStop2();
      encoderValue = 0;

      Serial.println("Axis2 hit limit during Run -> Auto Backoff (NON-BLOCKING)");

      motorForward2(HOMING2_SPEED_DOWN);
      b2_backoff = B2_BO_RUN;
      b2_backoffStartMs = millis();

      target2_smooth = encoderValue;
      targetPulses2  = encoderValue;
      desPulses2     = encoderValue;

      prevError = 0;
      integral = 0;
      lastPwmCmd = 0;
      return;
    }

    int absPwm = abs(pwmVal);
    if (absPwm < minPWM) absPwm = minPWM;
    motorBackward2(absPwm);
  }
  else {
    motorStop2();
  }
}

// =====================================================
//  TEST MODE
// =====================================================
void updateTestMode() {
  if (!testModeActive) return;
  if (!robotIdle()) return;

  if (testIndex < NUM_TEST_POSES) {
    startJointQuinticProfile(testA[testIndex], testB[testIndex], testC[testIndex]);
    testIndex++;
  } else {
    Serial.println("TEST DONE.");
    testModeActive = false;
  }
}

// =====================================================
//  CONVEYOR SENSOR HANDLER (NON-BLOCKING debounce)
//  Yêu cầu: nếu cảm biến convey nhận -> gửi DONE_STOP
// =====================================================
void handleConveyorSensor() {
  int r = digitalRead(CONVEYOR_SENSOR_PIN);

  if (r != conveyorLastRead) {
    conveyorLastRead = r;
    conveyorLastChangeMs = millis();
  }

  if ((millis() - conveyorLastChangeMs) >= CONVEYOR_DEBOUNCE_MS) {
    if (conveyorStable != conveyorLastRead) {
      conveyorStable = conveyorLastRead;

      if (conveyorStable == LOW) {
        digitalWrite(CONVEYOR_RELAY_PIN, LOW);
        Serial.println(">> TRANG THAI: CAM BIEN = 0 (Co vat) | BANG TAI = 1 (ON)");

        // ONLY send once per detection (LOW)
        if (!conveyorDoneStopSent) {
          conveyorDoneStopSent = true;
          sendDoneStop();
        }
      } else {
        digitalWrite(CONVEYOR_RELAY_PIN, HIGH);
        Serial.println(">> TRANG THAI: CAM BIEN = 1 (Khong vat) | BANG TAI = 0 (OFF)");
        conveyorDoneStopSent = false; // reset when back HIGH
      }
    }
  }
}

// =====================================================
//  SERIAL
// =====================================================
void handleSerial() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  String low = line;
  low.toLowerCase();

  // emergency ưu tiên tuyệt đối (kể cả busy)
  if (low == "emergency") {
    testModeActive = false;
    stopAllMotionImmediate();
    powerOffAll();
    return;
  }

  if (low.startsWith("mag")) {
    int val = low.substring(3).toInt();
    setMagnet(val != 0); // <<< sẽ tự in DONE
    return;
  }

  if (low.startsWith("test")) {
    int commaPos = low.indexOf(',');
    if (commaPos > 0) setSpeedPercent(low.substring(commaPos + 1).toFloat());
    testModeActive = true; testIndex = 0;
    stopAllMotionImmediate();
    hasLastPose = false;
    return;
  }

  if (low == "homeall") {
    testModeActive = false;
    stopAllMotionImmediate();
    hasLastPose = false;
    homingAll3();
    powerOffAll();
    return;
  }

  if (low == "home1") {
    testModeActive = false;
    stopAllMotionImmediate();
    hasLastPose = false;
    if (!joint1Homed) homingJoint1_findSwitch_setZero();
    else              goHome1ByAngle();
    powerOffAll();
    return;
  }

  if (low == "home2") {
    testModeActive = false;
    stopAllMotionImmediate();
    hasLastPose = false;
    homing2Active = true; homing2Stage = 1; moving2 = true;
    motorBackward2(HOMING2_SPEED_UP + 70);
    return;
  }

  if (low == "home3") {
    testModeActive = false;
    stopAllMotionImmediate();
    hasLastPose = false;
    homingJoint3_160();
    powerOffAll();
    return;
  }

  // L command
  if (line[0] == 'L' || line[0] == 'l') {
    if (!robotIdle()) { Serial.println("Busy."); return; }
    if (!joint1Homed) { Serial.println("Need home1/homeall first."); return; }

    String payload = line.substring(1);
    payload.trim();
    if (payload.startsWith(":")) payload = payload.substring(1);
    payload.trim();

    int c1 = payload.indexOf(',');
    int c2 = payload.indexOf(',', c1 + 1);
    int c3 = payload.indexOf(',', c2 + 1);
    int c4 = payload.indexOf(',', c3 + 1);

    if (c1 < 0 || c2 < 0) {
      Serial.println("Use: L Px(mm),Py(mm),Pz(cm)  or  L Px,Py,Pz,speed,mag");
      return;
    }

    float Px = payload.substring(0, c1).toFloat();
    float Py = payload.substring(c1 + 1, c2).toFloat();

    float Pz_cm = 0.0f, spd = 0.0f;
    int mag = 0;
    bool hasSpeed = false, hasMag = false;

    if (c3 < 0) {
      Pz_cm = payload.substring(c2 + 1).toFloat();
    } else if (c4 < 0) {
      Pz_cm = payload.substring(c2 + 1, c3).toFloat();
      spd   = payload.substring(c3 + 1).toFloat();
      hasSpeed = true;
    } else {
      Pz_cm = payload.substring(c2 + 1, c3).toFloat();
      spd   = payload.substring(c3 + 1, c4).toFloat();
      hasSpeed = true;
      mag   = payload.substring(c4 + 1).toInt();
      hasMag = true;
    }

    if (hasSpeed) setSpeedPercent(spd);
    if (hasMag)   setMagnet(mag != 0); // <<< in DONE ngay lúc set mag

    startCartesianMinJerkProfile(Px, Py, Pz_cm);
    return;
  }

  // Joint command
  if (!robotIdle()) { Serial.println("Busy."); return; }

  int c1 = line.indexOf(',');
  int c2 = line.indexOf(',', c1 + 1);
  int c3 = line.indexOf(',', c2 + 1);
  int c4 = line.indexOf(',', c3 + 1);
  if (c1 < 0 || c2 < 0) return;

  float a = line.substring(0, c1).toFloat();
  float b = line.substring(c1 + 1, c2).toFloat();
  float c, d;
  int e = -1;

  if (c3 < 0) {
    c = line.substring(c2 + 1).toFloat();
  } else if (c4 < 0) {
    c = line.substring(c2 + 1, c3).toFloat();
    d = line.substring(c3 + 1).toFloat();
    setSpeedPercent(d);
  } else {
    c = line.substring(c2 + 1, c3).toFloat();
    d = line.substring(c3 + 1, c4).toFloat();
    setSpeedPercent(d);
    String eStr = line.substring(c4 + 1);
    eStr.trim();
    e = eStr.toInt();
    setMagnet(e != 0); // <<< in DONE ngay lúc set mag
  }

  startJointQuinticProfile(a, b, c);
}

// =====================================================
//  SETUP & LOOP
// =====================================================
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(3);

  pinMode(STEP1_PIN, OUTPUT); pinMode(DIR1_PIN, OUTPUT); pinMode(ENA1_PIN, OUTPUT);
  pinMode(STEP3_PIN, OUTPUT); pinMode(DIR3_PIN, OUTPUT); pinMode(ENA3_PIN, OUTPUT);

  pinMode(pinA, INPUT_PULLUP);
  pinMode(pinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(pinA), handleEncoderA, RISING);

  pinMode(LIMIT2_PIN, INPUT_PULLUP);
  pinMode(RPWM, OUTPUT); pinMode(LPWM, OUTPUT); pinMode(EN, OUTPUT);
  motorStop2();

  pinMode(MAG_PIN, OUTPUT);
  digitalWrite(MAG_PIN, LOW);

  // Conveyor
  pinMode(CONVEYOR_SENSOR_PIN, INPUT_PULLUP);
  pinMode(CONVEYOR_RELAY_PIN, OUTPUT);
  digitalWrite(CONVEYOR_RELAY_PIN, LOW);

  conveyorStable = digitalRead(CONVEYOR_SENSOR_PIN);
  conveyorLastRead = conveyorStable;
  conveyorLastChangeMs = millis();
  conveyorDoneStopSent = false;

  // init smoothing axis2
  target2_smooth = encoderValue;
  targetPulses2  = encoderValue;
  lastSmoothMs   = millis();

  runRate1 = MIN_STEP_RATE;
  runRate3 = MIN_STEP_RATE;

  Serial.println("SYSTEM BOOTED.");
  Serial.println("Commands: homeall / home1..3 / emergency / test / a,b,c / L x,y,z / mag0|mag1");
}

void loop() {
  handleSerial();
  handleConveyorSensor();

  updateProfiles();

  updateAxis2SmoothTarget();

  updateStepper1();
  updateStepper3();

  updateAxis2();

  updateTestMode();
}
