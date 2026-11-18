# Wiring: Raspberry Pi + camera + ultrasonic wall-distance sensor

Power the Pi off before connecting anything.

## Parts

| Part | Notes |
| --- | --- |
| Raspberry Pi 4 or 5 (64-bit OS) | |
| Pi Camera Module (CSI ribbon) | or a USB webcam, which needs no wiring |
| Ultrasonic module with TRIG/ECHO pins | HC-SR04 for bench tests in air; JSN-SR04T has a sealed probe |
| 1 kΩ resistor | echo divider, upper leg |
| 2 kΩ resistor | echo divider, lower leg (two 1 kΩ in series also work) |
| Jumper wires, breadboard | |

## Why the divider is required

The sensor is powered from 5 V, so its ECHO pin outputs 5 V. Raspberry Pi GPIO pins
tolerate only 3.3 V, and 5 V on an input can permanently damage the Pi. The 1 kΩ + 2 kΩ
divider brings the echo signal to 5 V × 2 / (1 + 2) = 3.33 V. TRIG may be driven directly
from the Pi, because the sensor treats 3.3 V as a logic high.

## Pin connections

| Sensor pin | Connects to | Pi header pin | BCM GPIO |
| --- | --- | --- | --- |
| VCC | 5 V | 2 | |
| TRIG | GPIO23 | 16 | 23 |
| ECHO | divider output, then GPIO24 | 18 | 24 |
| GND | Ground | 6 | |

The software defaults are `--trig 23 --echo 24`. Use other pins by passing different BCM numbers.

## Circuit diagram

```
   RASPBERRY PI                                         ULTRASONIC MODULE
  ┌──────────────┐                                     ┌──────────────────┐
  │              │                                     │                  │
  │  5V   (pin 2)├─────────────────────────────────────┤ VCC              │
  │              │                                     │                  │
  │ GPIO23(pin16)├─────────────────────────────────────┤ TRIG             │
  │              │                                     │                  │
  │              │            ┌──────────┐             │                  │
  │ GPIO24(pin18)├────────────┤   node   ├──[ 1 kΩ ]───┤ ECHO             │
  │              │            └────┬─────┘             │                  │
  │              │                 │                   │                  │
  │              │              [ 2 kΩ ]               │                  │
  │              │                 │                   │                  │
  │  GND  (pin 6)├─────────────────┴───────────────────┤ GND              │
  │              │                                     │                  │
  └──────────────┘                                     └──────────────────┘

  The ECHO signal (5 V) passes through the 1 kΩ resistor to the node. The node
  feeds GPIO24 and is pulled to GND through the 2 kΩ resistor, giving 3.3 V at GPIO24.
```

The same circuit as a netlist:

```
5V (pin 2)        ── VCC
GND (pin 6)       ── GND (sensor) ── bottom of 2 kΩ
GPIO23 (pin 16)   ── TRIG
ECHO              ── 1 kΩ ── node ── GPIO24 (pin 18)
node              ── 2 kΩ ── GND
```

## Camera

Plug the ribbon cable into the CSI/CAMERA connector with the power off. Lift the connector's
latch, insert the cable with the contacts facing the correct side (towards the board's
contacts, as marked in the Pi documentation for your model), and press the latch down.
Raspberry Pi 5 uses a smaller 22-pin cable and a different connector from the Pi 4.
Check with:

```bash
rpicam-hello --list-cameras
```

## Checking the sensor on its own

```bash
python - <<'EOF'
import time
from gpiozero import DistanceSensor
s = DistanceSensor(echo=24, trigger=23, max_distance=2)
for _ in range(10):
    print(round(s.distance * 100, 1), "cm")
    time.sleep(0.5)
EOF
```

Point it at a wall about 50 cm away. If it always reads 200 cm (the maximum), check the
TRIG/ECHO wires and the VCC connection. If it reads 0, check the divider.

## Limits

- Air only: HC-SR04 and JSN-SR04T are designed for air. Their readings in water are not
  reliable (the transducer is not matched to water), so for the underwater prototype use an
  acoustic ranger designed for water, such as the Blue Robotics Ping sonar.
- Run with `--sound-speed 1480` for fresh water. Without it, distances measured in water
  will be wrong by a factor of about 4.3.
- Minimum range is about 2 cm for the HC-SR04 and about 20–25 cm for the JSN-SR04T.
- It reports distance to the wall, not crack depth inside the concrete.
- Millimetre sizes assume the camera is facing the wall squarely. Underwater, a flat port
  narrows the field of view by about 1.33× compared with air, so pass `--hfov` as the effective
  value in water.
