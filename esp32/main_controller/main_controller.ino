#include <Bluepad32.h>
#include "esp_task_wdt.h"

// ============================================================================
// MCTR 474 - Macadamia Harvesting Robot - Main ESP32 Controller
// PS5 DualSense input handling skeleton
// ============================================================================

ControllerPtr myController = nullptr;  // single controller, no need for an array of 4

// --- System state ---
enum RobotMode { MODE_MANUAL, MODE_AUTONOMOUS };
RobotMode currentMode = MODE_MANUAL;

bool eStopActive = false;
bool sweeperRunning = false;

// --- Bluetooth timeout safe-stop (NF-06) ---
unsigned long lastControllerUpdateMs = 0;
const unsigned long CONTROLLER_TIMEOUT_MS = 1500;

// --- Joystick thresholds, derived from bench measurement, see axis_calibration_template.txt ---
const int JOYSTICK_DEADZONE = 25;   // clears measured idle noise band of +/-12 raw units
const int JOYSTICK_MAX      = 500;  // symmetric clamp, measured extremes were -508 / +512, asymmetric

// --- PWM output range, standard placeholder values, confirm against final build ---
const int PWM_FREQ = 20000;        // 20kHz target, per handover doc Section 4
const int PWM_RESOLUTION_BITS = 8; // 8-bit (0-255), plenty of headroom at 20kHz
                                    // (max achievable freq at 8-bit res on ESP32's
                                    // 80MHz APB clock is ~312kHz, 20kHz is well inside range)
const int PWM_MAX = 255;           // matches PWM_RESOLUTION_BITS, NOT independent of it

// --- Motor driver pin/channel map (CORRECTED, see chat note: BTS7960 needs two PWM
// inputs per side, R_PWM and L_PWM, NOT one PWM pin + one digital DIR pin) ---
const int LEFT_RPWM_PIN = 4,   LEFT_RPWM_CHANNEL = 0;   // left forward
const int LEFT_LPWM_PIN = 25,  LEFT_LPWM_CHANNEL = 1;   // left reverse
const int RIGHT_RPWM_PIN = 27, RIGHT_RPWM_CHANNEL = 2;  // right forward
const int RIGHT_LPWM_PIN = 26, RIGHT_LPWM_CHANNEL = 3;  // right reverse

// --- Sweeper motor, PINS REVISED 2 Jul 2026 ---
// GPIO14 (original SWEEPER_RPWM) is the ESP32 HSPI_CLK pin. The boot ROM drives
// it during flash startup before setup() runs, causing a ~1s transient at 3.4V
// observed on scope. No firmware fix is possible since it happens before Arduino
// setup() is even called. Moved to GPIO19 which has no boot or strapping function.
// GPIO12 (original SWEEPER_LPWM) is a strapping pin for flash voltage selection.
// Moved to GPIO23 which is similarly clean.
// GPIO2 (original GREEN LED) is a strapping pin, must be low at boot.
// GPIO15 (original YELLOW LED) is a strapping pin, controls boot log verbosity.
// Both LED pins moved to clean GPIOs (GPIO21, GPIO22) to eliminate boot risk entirely.
const int SWEEPER_RPWM_PIN = 19, SWEEPER_RPWM_CHANNEL = 4;  // was GPIO14
const int SWEEPER_LPWM_PIN = 23, SWEEPER_LPWM_CHANNEL = 5;  // was GPIO12

// --- Status LEDs (330 ohm series resistor each), PINS REVISED 2 Jul 2026 ---
const int LED_GREEN_PIN  = 21;  // manual mode active,    was GPIO2  (strapping pin)
const int LED_YELLOW_PIN = 22;  // autonomous mode active, was GPIO15 (strapping pin)
const int LED_RED_PIN    = 33;  // fault / e-stop,         unchanged

// --- Sweeper ramp parameters, non-blocking (no delay()), updated once per loop() ---
const int SWEEPER_START_PWM  = PWM_MAX / 4;  // 25% initial speed on turn-on (63/255)
const int SWEEPER_MIN_PWM    = PWM_MAX / 10;  // 10% start speed  ( 25/255)
const int SWEEPER_TARGET_PWM = PWM_MAX / 2;   // 50% target speed (127/255)
const unsigned long SWEEPER_RAMP_MS = 1500;   // 1.5s ramp from 10% to 50%

// --- Software trim for straight-line drift correction ---
// Both set to 0 until empirical field testing determines the offset needed.
// Positive trim = that motor runs slightly faster, negative = slightly slower.
// Adjust one side only until robot drives straight on flat ground.
const int LEFT_TRIM  = 0;
const int RIGHT_TRIM = 0;

unsigned long sweeperRampStartMs = 0;
int sweeperRampStartValue = 0;
int sweeperRampTargetValue = 0;
int g_sweeperPWM = 0;  // current sweeper PWM, also used for telemetry

// --- Telemetry print interval, continuous one-line status like the original demo ---
const unsigned long PRINT_INTERVAL_MS = 100;
unsigned long lastPrintMs = 0;

// --- Last-computed values, stored globally purely so printStatus() can report them ---
int g_axisYRaw = 0, g_axisRXRaw = 0, g_throttle = 0, g_turn = 0;
int g_leftPWM = 0, g_rightPWM = 0;

// --- UART packet from Pi ---
// 8-byte structure: [bearing_error: int16][ground_distance: float32][mode_flag: uint8][checksum: uint8]
// Big-endian byte order throughout. Transmitted at 10Hz from Pi vision pipeline.
const int   UART_PACKET_SIZE      = 8;    // 7 payload bytes + 1 checksum
const float AUTONOMOUS_KP         = 0.5;   // proportional gain, tune empirically in field
const int   AUTONOMOUS_SPEED      = 191;   // 75% PWM approach/capture/search speed
const int   AUTO_MAX_PWM          = 191;   // hard cap on ALL autonomous motor outputs, 75% PWM
const int   CAPTURE_SPEED         = 191;   // 75% PWM capture drive
const int   TURN_RAMP_START_PWM   = 128;   // 50% turn authority at approach entry
const unsigned long TURN_RAMP_MS  = 1500;  // 1.5s ramp to full 75% turn authority
const unsigned long UART_TIMEOUT_MS    = 2000;
const unsigned long CAPTURE_DRIVE_MS  = 1200;
const unsigned long SEARCH_TIMEOUT_MS = 10000;
const unsigned long DETECTION_DROPOUT_MS = 800;  // grace period before switching to search on dropout

// UART state
int16_t  g_bearingMrad    = 0;
float    g_groundDistM    = 0.0f;
uint8_t  g_modeFlag       = 0;
unsigned long g_lastValidPacketMs = 0;
unsigned long g_lastDetectionMs   = 0;  // last time mode_flag=1 was received

// Capture zone state
bool          g_captureDriving  = false;
unsigned long g_captureStartMs  = 0;
bool          g_inApproach      = false;   // true while Zone 2 is active
unsigned long g_approachEntryMs = 0;       // timestamp when Zone 2 last started

// Forward declaration, needed because onDisconnectedController() (below) calls this
// before its full definition appears later in the file
void triggerSafeStop(const char* reason);

// Forward declaration, driveMotor() (below) needs this before its full definition appears
int clampToMax(int value, int maxVal);

// Forward declaration, onConnectedController() calls this before its definition appears
void updateModeLEDs();

// MotorOutput struct must be defined here, above every function, NOT further down
// the file. Arduino auto-generates function prototypes and inserts them before this
// point, if a function (like computeSkidSteer) returns MotorOutput and the struct is
// defined after that auto-inserted prototype, the compiler fails with "does not name
// a type." Defining it this early guarantees it's always seen first.
struct MotorOutput {
    int leftSpeed;   // -PWM_MAX to +PWM_MAX, sign indicates direction
    int rightSpeed;  // -PWM_MAX to +PWM_MAX, sign indicates direction
};

// ============================================================================
// Bluepad32 connection callbacks
// ============================================================================

void onConnectedController(ControllerPtr ctl) {
    if (myController == nullptr) {
        Serial.printf("Controller connected, index=%d\n", ctl->index());
        myController = ctl;
        lastControllerUpdateMs = millis();
        updateModeLEDs();
        // NOTE: no digitalWrite(LED_RED_PIN, LOW) here. If an e-stop was
        // active when the controller disconnected, updateModeLEDs() returns
        // early and leaves the RED LED on. The operator must press Circle
        // to explicitly acknowledge and clear the fault, same as any other
        // e-stop trigger. Reconnecting the controller alone is not sufficient
        // acknowledgement that the fault condition has been assessed.
    } else {
        Serial.println("Controller connected, but slot already occupied, ignoring");
    }
}

void onDisconnectedController(ControllerPtr ctl) {
    if (myController == ctl) {
        Serial.println("Controller disconnected");
        myController = nullptr;
        triggerSafeStop("controller disconnected");
    }
}

// ============================================================================
// Safety
// ============================================================================

void driveMotor(int rpwmChannel, int lpwmChannel, int speed) {
    speed = clampToMax(speed, PWM_MAX);
    if (speed > 0) {
        ledcWrite(rpwmChannel, speed);
        ledcWrite(lpwmChannel, 0);
    } else if (speed < 0) {
        ledcWrite(rpwmChannel, 0);
        ledcWrite(lpwmChannel, -speed);
    } else {
        ledcWrite(rpwmChannel, 0);
        ledcWrite(lpwmChannel, 0);
    }
}

void stopAllMotors() {
    driveMotor(LEFT_RPWM_CHANNEL, LEFT_LPWM_CHANNEL, 0);
    driveMotor(RIGHT_RPWM_CHANNEL, RIGHT_LPWM_CHANNEL, 0);
    g_leftPWM = 0;
    g_rightPWM = 0;
    // Sweeper shutdown is handled separately by stopSweeperInstant(), see below,
    // it needs its own ramp-state reset, not just a PWM write, so it isn't
    // duplicated here, triggerSafeStop() calls both functions together
}

void stopSweeperInstant() {
    // Used by the e-stop ONLY, bypasses the ramp entirely, sweeper must cut
    // immediately on e-stop, a graceful 1.5s ramp-down defeats the point of
    // an emergency stop
    driveMotor(SWEEPER_RPWM_CHANNEL, SWEEPER_LPWM_CHANNEL, 0);
    g_sweeperPWM = 0;
    sweeperRampStartValue = 0;
    sweeperRampTargetValue = 0;
    sweeperRampStartMs = millis();
    sweeperRunning = false;
}

void setSweeperTarget(int target) {
    // Begins a new ramp toward "target" (either SWEEPER_TARGET_PWM or 0),
    // starting from whatever the sweeper's current PWM value happens to be
    // right now, this matters if Cross is pressed again mid-ramp, the ramp
    // restarts smoothly from the current value rather than jumping
    sweeperRampStartMs = millis();
    sweeperRampStartValue = g_sweeperPWM;
    sweeperRampTargetValue = target;
}

void updateSweeperRamp() {
    // Called every loop() iteration, advances g_sweeperPWM toward
    // sweeperRampTargetValue over SWEEPER_RAMP_MS, linear interpolation
    if (g_sweeperPWM == sweeperRampTargetValue) return;  // nothing to do

    unsigned long elapsed = millis() - sweeperRampStartMs;
    if (elapsed >= SWEEPER_RAMP_MS) {
        g_sweeperPWM = sweeperRampTargetValue;
    } else {
        // Linear interpolation: how far along the ramp are we (0 to 1,
        // expressed as elapsed/duration), applied to the start->target span
        g_sweeperPWM = sweeperRampStartValue +
            (long)(sweeperRampTargetValue - sweeperRampStartValue) * (long)elapsed / (long)SWEEPER_RAMP_MS;
    }
    driveMotor(SWEEPER_RPWM_CHANNEL, SWEEPER_LPWM_CHANNEL, g_sweeperPWM);
}

void triggerSafeStop(const char* reason) {
    if (!eStopActive) {
        Serial.printf("SAFE-STOP triggered: %s\n", reason);
    }
    eStopActive = true;
    stopAllMotors();
    stopSweeperInstant();
    digitalWrite(LED_RED_PIN,    HIGH);
    digitalWrite(LED_GREEN_PIN,  LOW);
    digitalWrite(LED_YELLOW_PIN, LOW);
}

void updateModeLEDs() {
    // Single source of truth for mode indicator LEDs. Call this anywhere
    // currentMode changes so LEDs always reflect actual state regardless
    // of what triggered the mode change (button press or joystick override).
    // Returns immediately if e-stop is active since triggerSafeStop()
    // already forces all LEDs to a known state independently.
    if (eStopActive) return;
    digitalWrite(LED_GREEN_PIN,  currentMode == MODE_MANUAL     ? HIGH : LOW);
    digitalWrite(LED_YELLOW_PIN, currentMode == MODE_AUTONOMOUS ? HIGH : LOW);
}

void clearSafeStop() {
    eStopActive = false;
    digitalWrite(LED_RED_PIN, LOW);
    updateModeLEDs();  // restore green or yellow depending on which mode we're in
    Serial.println("Safe-stop cleared");
}

// ============================================================================
// Input processing
// ============================================================================

int applyDeadzone(int value, int deadzone) {
    if (abs(value) < deadzone) return 0;
    return value;
}

int clampToMax(int value, int maxVal) {
    if (value > maxVal) return maxVal;
    if (value < -maxVal) return -maxVal;
    return value;
}

// Tank/skid-steer mixing. Inputs already deadzone-applied, range -JOYSTICK_MAX to +JOYSTICK_MAX.
// throttle: positive = forward. turn: positive = right (per your bench data, axisRX right = +512).
MotorOutput computeSkidSteer(int throttleRaw, int turnRaw) {
    // Clamp inputs first, in case anything upstream feeds in an out-of-range value
    int throttle = clampToMax(throttleRaw, JOYSTICK_MAX);
    int turn = clampToMax(turnRaw, JOYSTICK_MAX);

    // Classic differential mix
    int left = throttle + turn;
    int right = throttle - turn;

    // The sum can exceed +/-JOYSTICK_MAX (e.g. full forward + full turn), clamp before scaling,
    // not after, or you bias the output asymmetrically toward whichever side overflowed
    left = clampToMax(left, JOYSTICK_MAX);
    right = clampToMax(right, JOYSTICK_MAX);

    // Scale from joystick range to PWM range
    MotorOutput out;
    out.leftSpeed = map(left, -JOYSTICK_MAX, JOYSTICK_MAX, -PWM_MAX, PWM_MAX);
    out.rightSpeed = map(right, -JOYSTICK_MAX, JOYSTICK_MAX, -PWM_MAX, PWM_MAX);
    return out;
}

// ============================================================================
// UART Packet Parser (Pi -> ESP32)
// ============================================================================

bool parseUartPacket() {
    static uint8_t buf[UART_PACKET_SIZE];
    static int     bytesRead = 0;

    while (Serial2.available() > 0 && bytesRead < UART_PACKET_SIZE) {
        buf[bytesRead++] = Serial2.read();
    }

    if (bytesRead < UART_PACKET_SIZE) return false;
    bytesRead = 0;

    uint8_t checksum = 0;
    for (int i = 0; i < 7; i++) checksum ^= buf[i];
    if (checksum != buf[7]) {
        Serial.println("UART: checksum mismatch, packet discarded");
        return false;
    }

    g_bearingMrad = (int16_t)((buf[0] << 8) | buf[1]);

    uint32_t raw = ((uint32_t)buf[2] << 24) | ((uint32_t)buf[3] << 16)
                 | ((uint32_t)buf[4] << 8)  |  (uint32_t)buf[5];
    memcpy(&g_groundDistM, &raw, sizeof(float));

    g_modeFlag = buf[6];
    g_lastValidPacketMs = millis();
    if (g_modeFlag == 1) g_lastDetectionMs = millis();
    return true;
}

void runAutonomousSteering() {
    unsigned long now = millis();

    // Task complete check runs first, outside all zones. Fires when no ball
    // has been seen for SEARCH_TIMEOUT_MS regardless of dropout state.
    if (now - g_lastDetectionMs > SEARCH_TIMEOUT_MS && !g_captureDriving) {
        Serial.println("SEARCH timeout: no ball found, task complete, returning to MANUAL");
        currentMode      = MODE_MANUAL;
        g_captureDriving = false;
        stopAllMotors();
        updateModeLEDs();
        return;
    }

    // Zone 3: capture drive in progress, ignore all UART input
    if (g_captureDriving) {
        if (now - g_captureStartMs < CAPTURE_DRIVE_MS) {
            int capSpeed = constrain(CAPTURE_SPEED, 0, AUTO_MAX_PWM);
            driveMotor(LEFT_RPWM_CHANNEL,  LEFT_LPWM_CHANNEL,  capSpeed);
            driveMotor(RIGHT_RPWM_CHANNEL, RIGHT_LPWM_CHANNEL, capSpeed);
            g_leftPWM  = capSpeed;
            g_rightPWM = capSpeed;
        } else {
            g_captureDriving = false;
            stopAllMotors();
            Serial.println("CAPTURE complete, returning to SEARCH");
        }
        return;
    }

    // Zone 1: search sweep. Only enters if UART has gone silent OR the flag
    // has been 0 for longer than the dropout tolerance. Brief dropouts are
    // ignored and the robot holds its last approach trajectory.
    if (now - g_lastValidPacketMs > UART_TIMEOUT_MS ||
       (g_modeFlag == 0 && now - g_lastDetectionMs > DETECTION_DROPOUT_MS)) {
        g_inApproach = false;  // Reset approach flag so turn ramp restarts on next approach
        int searchSpeed = AUTO_MAX_PWM;  // 75% PWM pivot speed
        driveMotor(LEFT_RPWM_CHANNEL,  LEFT_LPWM_CHANNEL,   searchSpeed + LEFT_TRIM);
        driveMotor(RIGHT_RPWM_CHANNEL, RIGHT_LPWM_CHANNEL, -searchSpeed + RIGHT_TRIM);
        g_leftPWM  =  searchSpeed;
        g_rightPWM = -searchSpeed;
        return;
    }

    // Zone 2: ball detected and within valid range.
    // Note: g_lastDetectionMs is updated in parseUartPacket() when flag=1.
    // Do NOT update it here or the dropout timer never accumulates.

    if (g_groundDistM > 0.0f && g_groundDistM < 0.15f) {
        Serial.printf("CAPTURE triggered at %.3fm, driving forward %dms\n",
                      g_groundDistM, CAPTURE_DRIVE_MS);
        g_captureDriving = true;
        g_captureStartMs = now;
        return;
    }

    // Mark approach entry and start turn ramp from 50% to 75% over 1.5s.
    // Ramp resets each time the robot transitions from search into approach,
    // preventing jerky full-authority turns on initial target acquisition.
    if (!g_inApproach) {
        g_inApproach      = true;
        g_approachEntryMs = now;
    }
    unsigned long approachElapsed = now - g_approachEntryMs;
    int turnCap;
    if (approachElapsed >= TURN_RAMP_MS) {
        turnCap = AUTO_MAX_PWM;  // 75% full turn authority
    } else {
        turnCap = TURN_RAMP_START_PWM +
            (int)((long)(AUTO_MAX_PWM - TURN_RAMP_START_PWM)
                  * (long)approachElapsed / (long)TURN_RAMP_MS);
    }

    int turnOffset = (int)(AUTONOMOUS_KP * (float)g_bearingMrad / 10.0f);
    turnOffset = clampToMax(turnOffset, turnCap);

    int leftSpeed  = constrain(AUTONOMOUS_SPEED + turnOffset + LEFT_TRIM,  -AUTO_MAX_PWM, AUTO_MAX_PWM);
    int rightSpeed = constrain(AUTONOMOUS_SPEED - turnOffset + RIGHT_TRIM, -AUTO_MAX_PWM, AUTO_MAX_PWM);

    driveMotor(LEFT_RPWM_CHANNEL,  LEFT_LPWM_CHANNEL,  leftSpeed);
    driveMotor(RIGHT_RPWM_CHANNEL, RIGHT_LPWM_CHANNEL, rightSpeed);
    g_leftPWM  = leftSpeed;
    g_rightPWM = rightSpeed;
}

void processGamepad(ControllerPtr ctl) {
    lastControllerUpdateMs = millis();

    // --- R3 confirmed bitmask 0x0200 on this DualSense unit via Bluepad32 ---
    if (ctl->buttons() & 0x0200) {
        Serial.println("R3 pressed: ESP32 resetting...");
        delay(100);
        ESP.restart();
    }

    // --- Triangle: E-STOP, latching, works in both manual and autonomous mode ---
    // Triangle = y() (0x0008), per bench-confirmed mapping, NOT a named triangle()
    // accessor, Bluepad32 only exposes a()/b()/x()/y(), it does not have PlayStation-
    // labelled methods. This was my error in the previous version, corrected here.
    if (ctl->y()) {
        triggerSafeStop("Triangle pressed");
    }

    // --- Square: mode switch (manual <-> autonomous) ---
    // Mapping confirmed by bench test, 2 Jul: Cross=a (0x0001), Circle=b (0x0002),
    // Square=x (0x0004), Triangle=y (0x0008)
    static bool squarePrev = false;
    bool squareNow = ctl->x();
    if (squareNow && !squarePrev && !eStopActive) {
        currentMode = (currentMode == MODE_MANUAL) ? MODE_AUTONOMOUS : MODE_MANUAL;
        updateModeLEDs();
        if (currentMode == MODE_AUTONOMOUS) {
            // Auto-start sweeper when entering autonomous mode
            if (!sweeperRunning) {
                sweeperRunning = true;
                g_sweeperPWM = SWEEPER_MIN_PWM;        // snap to 10% immediately
                setSweeperTarget(SWEEPER_TARGET_PWM);  // ramp to 50% over 1.5s
                Serial.println("Autonomous: sweeper auto-started (10% -> 50% over 1.5s)");
            }
            g_lastDetectionMs = millis();  // reset search timeout on mode entry
        }
        Serial.printf("Mode switched to: %s\n", currentMode == MODE_MANUAL ? "MANUAL" : "AUTONOMOUS");
    }
    squarePrev = squareNow;

    // --- Cross: toggle sweeper motor on/off, ramps to/from 50% over 1.5s ---
    static bool crossPrev = false;
    bool crossNow = ctl->a();
    if (crossNow && !crossPrev && !eStopActive) {
        sweeperRunning = !sweeperRunning;
        Serial.printf("Sweeper: %s (ramping)\n", sweeperRunning ? "ON" : "OFF");
        if (sweeperRunning) {
            g_sweeperPWM = SWEEPER_MIN_PWM;        // snap to 10% immediately
            setSweeperTarget(SWEEPER_TARGET_PWM);  // ramp to 50% over 1.5s
        } else {
            setSweeperTarget(0);  // ramp down from current to 0 over 1.5s
        }
    }
    crossPrev = crossNow;

    // --- Circle: clear / reset E-STOP ---
    static bool circlePrev = false;
    bool circleNow = ctl->b();
    if (circleNow && !circlePrev && eStopActive) {
        clearSafeStop();
    }
    circlePrev = circleNow;

    // --- Manual mode driving: left stick forward/back, right stick skid-steer ---
    if (currentMode == MODE_MANUAL && !eStopActive) {
        int throttleRaw = applyDeadzone(-ctl->axisY(), JOYSTICK_DEADZONE);
        // axisY() confirmed negative-up on bench, 2 Jul: forward = -508, backward = +512
        int turnRaw = applyDeadzone(ctl->axisRX(), JOYSTICK_DEADZONE);

        MotorOutput out = computeSkidSteer(throttleRaw, turnRaw);
        driveMotor(LEFT_RPWM_CHANNEL,  LEFT_LPWM_CHANNEL,  out.leftSpeed  + LEFT_TRIM);
        driveMotor(RIGHT_RPWM_CHANNEL, RIGHT_LPWM_CHANNEL, out.rightSpeed + RIGHT_TRIM);

        // Store for telemetry printing only, has no effect on control logic
        g_axisYRaw = ctl->axisY();
        g_axisRXRaw = ctl->axisRX();
        g_throttle = throttleRaw;
        g_turn = turnRaw;
        g_leftPWM = out.leftSpeed;
        g_rightPWM = out.rightSpeed;
    }

    if (currentMode == MODE_AUTONOMOUS) {
        int throttleCheck = applyDeadzone(-ctl->axisY(), JOYSTICK_DEADZONE);
        int turnCheck     = applyDeadzone(ctl->axisRX(), JOYSTICK_DEADZONE);
        if (throttleCheck != 0 || turnCheck != 0) {
            currentMode = MODE_MANUAL;
            g_captureDriving = false;
            stopAllMotors();
            // Deliberately NOT stopping the sweeper here. The sweeper continues
            // spinning when joystick override returns to manual, consistent with
            // the design intent that only e-stop cuts the sweeper in auto context.
            // The Cross button toggle still works in manual to stop it manually.
            updateModeLEDs();
            Serial.println("Joystick override: forced back to MANUAL mode (sweeper continues)");
        }
        // Autonomous steering handled in loop() via runAutonomousSteering()
    }
}

void processControllers() {
    if (myController && myController->isConnected() && myController->hasData()) {
        if (myController->isGamepad()) {
            processGamepad(myController);
        } else {
            Serial.println("Connected device is not a gamepad, ignoring");
        }
    }
}

void printStatus() {
    Serial.printf(
        "mode=%s, estop=%s, sweeper=%s, sweeperPWM=%4d, "
        "bearing=%5dmrad, dist=%.3fm, flag=%d, "
        "axisY=%4d, axisRX=%4d, leftPWM=%4d, rightPWM=%4d\n",
        currentMode == MODE_MANUAL ? "MANUAL" : "AUTO",
        eStopActive ? "ACTIVE" : "clear",
        sweeperRunning ? "ON" : "off",
        g_sweeperPWM,
        g_bearingMrad, g_groundDistM, g_modeFlag,
        g_axisYRaw, g_axisRXRaw, g_leftPWM, g_rightPWM
    );
}

// ============================================================================
// Setup / Loop
// ============================================================================

void setup() {
    Serial.begin(115200);

    // LEDC PWM channels configured FIRST, before anything else, to claim all
    // motor output pins and drive them to a known zero as early as possible.
    // Previously these were configured after BP32.setup(), which left GPIO pins
    // floating during Bluetooth init and caused a brief transient on boot
    // (observed on scope as a ~0.5s pulse to 3.38V on GPIO14). Confirmed fix
    // by oscilloscope test, 2 Jul 2026.
    ledcSetup(LEFT_RPWM_CHANNEL,    PWM_FREQ, PWM_RESOLUTION_BITS);
    ledcAttachPin(LEFT_RPWM_PIN,    LEFT_RPWM_CHANNEL);
    ledcSetup(LEFT_LPWM_CHANNEL,    PWM_FREQ, PWM_RESOLUTION_BITS);
    ledcAttachPin(LEFT_LPWM_PIN,    LEFT_LPWM_CHANNEL);
    ledcSetup(RIGHT_RPWM_CHANNEL,   PWM_FREQ, PWM_RESOLUTION_BITS);
    ledcAttachPin(RIGHT_RPWM_PIN,   RIGHT_RPWM_CHANNEL);
    ledcSetup(RIGHT_LPWM_CHANNEL,   PWM_FREQ, PWM_RESOLUTION_BITS);
    ledcAttachPin(RIGHT_LPWM_PIN,   RIGHT_LPWM_CHANNEL);
    ledcSetup(SWEEPER_RPWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION_BITS);
    ledcAttachPin(SWEEPER_RPWM_PIN, SWEEPER_RPWM_CHANNEL);
    ledcSetup(SWEEPER_LPWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION_BITS);
    ledcAttachPin(SWEEPER_LPWM_PIN, SWEEPER_LPWM_CHANNEL);
    stopAllMotors();
    stopSweeperInstant();

    Serial.printf("Firmware: %s\n", BP32.firmwareVersion());
    BP32.setup(&onConnectedController, &onDisconnectedController);
    BP32.enableVirtualDevice(false);

    // UART2 link to Raspberry Pi (Pi GPIO14 -> ESP32 GPIO16, Pi GPIO15 <- ESP32 GPIO17)
    // Big-endian 8-byte packets at 10Hz from Pi vision pipeline
    Serial2.begin(115200, SERIAL_8N1, 16, 17);
    g_lastValidPacketMs = millis();
    g_lastDetectionMs   = millis();  // prevent immediate search timeout on boot
    Serial.println("UART2 to Pi initialised on GPIO16/17 at 115200 baud");

    // Status LED pin setup, pins confirmed clean of strapping/boot functions
    pinMode(LED_GREEN_PIN,  OUTPUT);
    pinMode(LED_YELLOW_PIN, OUTPUT);
    pinMode(LED_RED_PIN,    OUTPUT);
    digitalWrite(LED_GREEN_PIN,  LOW);
    digitalWrite(LED_YELLOW_PIN, LOW);
    digitalWrite(LED_RED_PIN,    LOW);

    // Boot into e-stop state. RED LED on immediately, requiring Circle
    // acknowledgement before any motion is permitted. Ensures robot never
    // moves after power-on or ESP32 reset without deliberate human input.
    eStopActive = true;
    digitalWrite(LED_RED_PIN, HIGH);
    Serial.println("Boot e-stop active. Press Circle to acknowledge and clear.");

    // Hardware watchdog NF-03: 3 second timeout. If loop() ever stops completing
    // within 3 seconds (frozen code, infinite loop, hung peripheral), the board
    // resets and all PWM outputs go to zero via the LEDC hardware reset state.
    // esp_task_wdt_add(NULL) registers the current task (Arduino loop task).
    esp_task_wdt_init(3, true);
    esp_task_wdt_add(NULL);
}

void loop() {
    bool dataUpdated = BP32.update();
    if (dataUpdated) {
        processControllers();
    }

    // --- NF-06: Bluetooth timeout safe-stop ---
    if (myController != nullptr && (millis() - lastControllerUpdateMs > CONTROLLER_TIMEOUT_MS)) {
        if (!eStopActive) {
            triggerSafeStop("controller update timeout");
        }
    }

    // --- Parse incoming UART packets from Pi (runs every iteration, non-blocking) ---
    parseUartPacket();

    // --- Autonomous steering (runs every iteration when in autonomous mode) ---
    if (currentMode == MODE_AUTONOMOUS && !eStopActive) {
        runAutonomousSteering();
    }

    // --- Sweeper ramp ---
    updateSweeperRamp();

    // --- Continuous telemetry, one line every PRINT_INTERVAL_MS, independent of
    // whether new controller data arrived this iteration ---
    if (millis() - lastPrintMs >= PRINT_INTERVAL_MS) {
        printStatus();
        lastPrintMs = millis();
    }

    vTaskDelay(1);  // yield to avoid tripping the RTOS watchdog
    esp_task_wdt_reset();  // kick hardware watchdog (NF-03), must fire within 3s
}