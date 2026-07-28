"""DART station coordinates (latitude, longitude) for the live worker.

Supplies station coordinates to ``AnomalyAgent.process_station_data`` so the
Rayleigh-wave false-trigger check can run on live data (it compares the station
position to the seismic epicenter from the agent's recent seismic events).

Values are NDBC DART positions, kept in sync with the Mission Control station
map (``mission-control/frontend/src/data/stations.json``). Stations not listed
here get no coordinates, in which case the Rayleigh check is skipped for
them rather than evaluated against a fabricated position.
"""

from __future__ import annotations

# station_id -> (latitude_deg, longitude_deg)
DART_STATION_COORDS: dict[str, tuple[float, float]] = {
    '21401': (42.617, 152.583),
    '21413': (30.515, 152.117),
    '21414': (48.97, 178.165),
    '21415': (50.12, 171.867),
    '21416': (48.126, 163.355),
    '21418': (38.73, 148.8),
    '21419': (44.401, 155.653),
    '21420': (28.91, 135.0),
    '32067': (0.641, -81.262),
    '32401': (-20.442, -73.422),
    '32411': (4.979, -90.793),
    '32412': (-17.984, -86.374),
    '32413': (-7.407, -93.517),
    '43412': (16.012, -107.004),
    '43413': (10.927, -100.012),
    '46402': (50.913, -164.147),
    '46403': (52.647, -156.94),
    '46404': (45.87, -128.756),
    '46407': (42.704, -128.895),
    '46408': (49.677, -169.825),
    '46409': (55.338, -148.575),
    '46410': (57.633, -143.843),
    '46411': (39.337, -127.04),
    '46412': (32.4, -120.582),
    '46413': (48.042, -173.942),
    '46414': (53.764, -152.416),
    '46415': (52.975, -139.94),
    '46416': (49.939, -134.427),
    '46419': (48.815, -129.623),
    '51407': (19.53, -156.601),
    '51425': (-9.511, -176.258),
    '51426': (-23.11, -168.385),
    '52401': (19.285, 155.739),
    '52402': (11.928, 153.876),
    '52403': (4.018, 145.628),
    '52404': (20.627, 132.144),
    '52405': (13.034, 132.151),
    '52406': (-5.371, 164.985),
    '54401': (-33.109, -173.155),
}


def station_coordinates(station_id: str) -> tuple[float, float] | None:
    """Return ``(lat, lon)`` for a known DART station, or ``None`` if unknown."""
    return DART_STATION_COORDS.get(station_id)
