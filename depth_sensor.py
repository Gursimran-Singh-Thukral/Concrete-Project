"""Wall-distance sensing with a trigger/echo ultrasonic module (HC-SR04, JSN-SR04T, ...).

This measures the distance between the sensor and the surface in front of it.
It does not measure crack depth inside the concrete.
"""
from __future__ import annotations

import math

AIR_SPEED = 343.26     # m/s, the speed gpiozero assumes when it converts echo time to distance
WATER_SPEED = 1480.0   # m/s, fresh water at about 20 C


class UltrasonicSensor:
    """Distance in metres from a 5 V trigger/echo module wired to the Raspberry Pi GPIO.

    The echo pin must go through a voltage divider first (see docs/WIRING.md).
    gpiozero converts echo time to distance assuming sound travels at AIR_SPEED,
    so readings are rescaled when the sensor operates in a different medium.
    """

    def __init__(self, trigger_pin=23, echo_pin=24, sound_speed=AIR_SPEED, max_range=2.0):
        from gpiozero import DistanceSensor
        self.scale = sound_speed / AIR_SPEED
        self.max_range = max_range
        # gpiozero reports distance in air-metres, so its cap is the real cap divided by the scale.
        self.sensor = DistanceSensor(echo=echo_pin, trigger=trigger_pin,
                                     max_distance=max_range / self.scale, queue_len=5)

    def read(self):
        """Latest median distance in metres, or None when nothing was echoed within range."""
        distance = self.sensor.distance * self.scale
        return distance if distance < self.max_range * 0.999 else None

    def close(self):
        self.sensor.close()


def mm_per_pixel(distance_m, hfov_deg, frame_width):
    """Millimetres covered by one image pixel on a flat wall at the given distance."""
    return 2000 * distance_m * math.tan(math.radians(hfov_deg) / 2) / frame_width


def scale_boxes(metadata, distance_m, hfov_deg):
    """Copy of the detection metadata with distance, mm/pixel and box sizes in millimetres.

    Valid only for a camera facing the wall squarely; treat sizes as rough estimates.
    """
    if distance_m is None:
        return dict(metadata, distance_m=None)
    scale = mm_per_pixel(distance_m, hfov_deg, metadata["frame_width"])
    boxes = [dict(box, width_mm=box["width"] * scale, height_mm=box["height"] * scale)
             for box in metadata["boxes"]]
    return dict(metadata, distance_m=distance_m, mm_per_pixel=scale, boxes=boxes)
