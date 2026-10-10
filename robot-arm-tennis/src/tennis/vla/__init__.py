"""Robot-arm tennis for Vision-Language-Action (VLA) policies.

worlds    "planet" worlds: only gravity changes, which rescales the game's time
          (geometry of every flight stays the same, it is just slower or faster)
tasks     instructed return drills (language instruction -> landing zone)
episode   episode runner with pluggable controllers (physics expert, action replay,
          VLA end-to-end, VLA + physics hierarchical) and simulated inference latency
render    camera images for the policy
"""
