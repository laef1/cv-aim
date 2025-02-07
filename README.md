# cv-aim
Computer vision aim aligner made in Python &amp; Cython

# Prerequisites 
1. Download **requirements.txt, aim_alignment.pyx, and setup.py**
2. Put all 3 of them in the same folder!
3. Open CMD in the same directory as those 3 files
4. Type this into CMD ```pip install -r requirements.txt```

# How to Operate
1. Open CMD in the same directory as your previous folder you made in **Prerequisites**
2. To compile this .pyx file use this command ```python setup.py build_ext --inplace```
3. You should see a .pyd file. Now make sure the .pyd stays in the same directory as everything else
4. You will need an ONNX model to use, grab one and rename it to ```best.onnx``` and put it in the same directory as everything else
5. Make a new Python script (or copy test_module.py) and use these 2 lines
```Python
import aimbot

# You can obviously change these paramaters, You can only use RMB as aim_key for now
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
```
NOTE: To change the name, setupy.py -> Replace "Aimbot" with your perferred name.
