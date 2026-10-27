"""The One Euro Filter (Casiez, Roussel & Vogel, CHI 2012) - a low-pass
filter whose cutoff frequency adapts to signal speed: heavy smoothing while
a value is nearly still (killing jitter), light smoothing during a fast
real movement (keeping lag low). Standard reference implementation
(cristalhq/one-euro-filter and the authors' own site, hypertino.github.io/
OneEuroFilter) reproduced directly - no project-specific logic lives here,
only the gaze pipeline's choice of `min_cutoff`/`beta` does (see
docs/context/gaze.md for how those were picked).

Used here to smooth gaze yaw/pitch over time without lagging behind a real
saccade - a plain fixed-window moving average would do the opposite
(smooth real movement into mush, or barely filter jitter at all), which is
exactly the tradeoff this filter is designed to avoid.
"""

import math


class _LowPassFilter:
    def __init__(self, alpha: float, initial_value: float = 0.0):
        self._alpha = alpha
        self._value = initial_value
        self._initialized = False

    def filter(self, value: float, alpha: float | None = None) -> float:
        if alpha is not None:
            self._alpha = alpha
        if not self._initialized:
            self._value = value
            self._initialized = True
        else:
            self._value = self._alpha * value + (1.0 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value


def _smoothing_factor(dt_seconds: float, cutoff_hz: float) -> float:
    r = 2.0 * math.pi * cutoff_hz * dt_seconds
    return r / (r + 1.0)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_filter = _LowPassFilter(0.0)
        self._dx_filter = _LowPassFilter(0.0)
        self._last_value: float | None = None

    def filter(self, value: float, dt_seconds: float) -> float:
        dt_seconds = max(dt_seconds, 1e-6)  # guards the division in _smoothing_factor against a zero/negative dt
        dx = 0.0 if self._last_value is None else (value - self._last_value) / dt_seconds
        self._last_value = value

        edx = self._dx_filter.filter(dx, _smoothing_factor(dt_seconds, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x_filter.filter(value, _smoothing_factor(dt_seconds, cutoff))
