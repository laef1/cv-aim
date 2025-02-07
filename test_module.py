import aimbot

aimbot.run_aim_alignment(
    aim_key="RMB", # No other binds besides RMB
    fov_radius=250,
    fov_enabled=True, # Doesn't display, still works | BROKEN
    ai_confidence=0.17,
    sensitivity=1.70,
    gpu=True,
    use_trt=False,
    x_offset=-25,
    y_offset=-40
)

