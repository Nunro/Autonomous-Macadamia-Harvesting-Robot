# Autonomous Macadamia Nut Harvesting Robot

**MCTR 474 Final Year Project**  
**Student:** Nunro Brenkman (37540505)  
**Institution:** North-West University, Potchefstroom  
**Year:** 2026

---

## Project Overview

A semi-autonomous robot designed to collect macadamia nuts (golf balls used as surrogates) using computer vision and differential drive skid-steer locomotion. The system uses two embedded controllers communicating over UART.

---

## Repository Structure

```
esp32/
  main_controller/
    main_controller.ino     # Complete ESP32 firmware (Arduino IDE)

raspberry_pi/
    harvest_vision.py       # Vision pipeline with MJPEG stream (Pi 5)

simulation/
    mctr474_simulation.m    # MATLAB R2025a simulation (7 figures, F5 to run)

docs/
    pinout_wiring_reference.txt
    simulation_figures_handover.txt
```

---

## Hardware

| Component | Details |
|---|---|
| Motion controller | ESP32 DevKit V1 (Bluepad32 firmware) |
| Vision controller | Raspberry Pi 5 (4GB) |
| Camera | Pi Camera Module 3 Wide |
| Drive motors | 2x RS PRO 12V geared DC (rear-wheel drive) |
| Motor drivers | 3x BTS7960 H-bridge |
| Battery | 2x Ingco 20V 4Ah Li-Ion in parallel |
| Pi power | Waveshare UPS HAT (B) + 2x Samsung 30Q 18650 |

---

## Branch Workflow (NF-02)

- **main** — stable, tested, working code only
- **development** — active development branch

All changes are made on `development` and merged to `main` only after verification.

---

## Quick Start

**ESP32:** Open `esp32/main_controller/main_controller.ino` in Arduino IDE with the `esp32_bluepad32 DOIT ESP32 DEVKIT V1` board selected. Flash to device.

**Raspberry Pi:** Copy `raspberry_pi/harvest_vision.py` to `/home/nunro/harvest/` on the Pi. The systemd service auto-starts it on boot. Stream available at `http://10.42.0.1:8080/stream` on the HarvestBot hotspot.

**Simulation:** Open `simulation/mctr474_simulation.m` in MATLAB R2025a and press F5. Seven figures generate automatically.
