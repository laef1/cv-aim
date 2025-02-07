# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
YOLOv8 ONNX/TensorRT Inference & Aim Alignment Module
--------------------------------------------------------
NOTE: You have to use the name best.onnx in the same directory as this script.
      The ONNX model should have the following output shape: (1, 5, 8400).

  - Auto conversion of best.onnx to a TensorRT engine (.engine) with fp16,
    and loading that engine if available.
  - Fast screen capture using the dxcam module.
  - GPU mode with execution type chosen via a boolean parameter:
      True  = TensorRT,
      False = CUDA (ONNX Runtime with GPU).
      
Usage Example (in your main script):
    import aim_alignment
    # Parameters: aim_key, fov_radius, fov_enabled, ai_confidence, aim_strength, gpu, use_trt
    aim_alignment.run_aim_alignment("RMB", 300, True, 0.5, 0.75, True, True)
"""

from libc.math cimport sqrt

import sys, time, ctypes, os, subprocess, random, shutil
import numpy as np
cimport numpy as np
import onnxruntime as ort
import cv2
import dxcam
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cython
import onnx

from ctypes import c_long, c_uint


# Global dxcam camera instance
cdef object CAMERA = None

def init_camera():
    global CAMERA
    if CAMERA is None:
        CAMERA = dxcam.create(output_color="BGR", output_idx=0)


# Structs for Mouse Input
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", c_long),
        ("dy", c_long),
        ("mouseData", c_uint),
        ("dwFlags", c_uint),
        ("time", c_uint),
        ("dwExtraInfo", ctypes.c_ulonglong)
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", c_uint),
        ("mi", MOUSEINPUT)
    ]


# Inference Setup Functions
cpdef object load_onnx_session(str onnx_file_path="best.onnx", bint use_gpu=False):
    """
    Creates an ONNX Runtime InferenceSession.
    """
    cdef list providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_file_path, providers=providers)

def allocate_buffers(engine):
    """
    Allocates host and device buffers for a TensorRT engine built in explicit batch mode.

    Uses engine.num_io_tensors to iterate over all IO tensors.
    For each tensor, it retrieves the tensor name (via get_tensor_name(i)), its shape
    (via get_tensor_shape(name)) and its data type (via get_tensor_dtype(name)).
    It then allocates host memory with cuda.pagelocked_empty and device memory with cuda.mem_alloc.
    
    Returns:
      A tuple (inputs, outputs, bindings, stream)
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    nb_tensors = engine.num_io_tensors  # number of IO tensors
    for i in range(nb_tensors):
        binding_name = engine.get_tensor_name(i)
        dims = engine.get_tensor_shape(binding_name)
        size = trt.volume(dims)
        dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        # Determine if the tensor is input or output using get_tensor_mode
        if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name})
    return inputs, outputs, bindings, stream

def run_inference_trt(context, bindings, inputs, outputs, stream, input_data, engine):
    # Copy input data into the host buffer and then asynchronously to device memory.
    np.copyto(inputs[0]['host'], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    
    # For each I/O tensor, update its device address in the execution context.
    nb_io = engine.num_io_tensors 
    for i in range(nb_io):
        tensor_name = engine.get_tensor_name(i)
        addr = bindings[i]
        context.set_tensor_address(tensor_name, addr)
    
    # Execute inference asynchronously using the stream handle.
    context.execute_async_v3(stream.handle)
    
    # Copy the output from device to host.
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    
    # Reshape and return the output.
    out_np = np.array(outputs[0]['host'])
    out_np = out_np.reshape((1, 5, 8400))
    return out_np

def build_engine_from_onnx(onnx_file_path, engine_file_path, fp16=True):
    """
    Builds a TensorRT engine from an ONNX file using the TensorRT Python API.

    Parameters:
      onnx_file_path (str): Path to the ONNX model.
      engine_file_path (str): Path to save the serialized engine.
      fp16 (bool): Whether to build the engine with FP16 mode.

    Returns:
      engine: The built TensorRT engine.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    # Use explicit batch mode
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Load and parse the ONNX model.
    with open(onnx_file_path, 'rb') as model:
        model_data = model.read()
        if not parser.parse(model_data):
            print("Failed to parse ONNX file:")
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise RuntimeError("ONNX model parsing failed.")

    config = builder.create_builder_config()
    # Set the workspace size (1GB).
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: FP16 not supported on this platform. Building in FP32.")

    # Build the engine by first creating a serialized network.
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed.")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to deserialize the engine.")

    # Save the .engine to disk.
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Engine built and saved to:", engine_file_path)
    return engine

def setup_inference(gpu, use_trt):
    """
    Sets up the inference engine.

    Parameters:
      gpu (bool): True to use GPU; False for CPU.
      use_trt (bool): True to use TensorRT; False to use ONNX Runtime.

    For GPU inference with TensorRT, if a serialized engine file ('best.engine') does not
    exist in the current working directory, this function builds the engine from 'best.onnx'
    using TensorRT's Python API.

    Returns:
      For TensorRT: A tuple (engine, context, inputs, outputs, bindings, stream).
      For ONNX Runtime: An inference session.
    """
    if not gpu:
        return load_onnx_session("best.onnx", use_gpu=False)
    else:
        if use_trt:
            cwd = os.getcwd()
            onnx_path = os.path.join(cwd, "best.onnx")
            engine_path = os.path.join(cwd, "best.engine")
            if os.path.exists(engine_path):
                print("Loading existing TensorRT engine from:", engine_path)
                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(TRT_LOGGER)
                with open(engine_path, "rb") as f:
                    engine_data = f.read()
                engine = runtime.deserialize_cuda_engine(engine_data)
                if engine is None:
                    raise RuntimeError("Failed to deserialize the TensorRT engine from file.")
            else:
                print("Engine file not found. Building engine from ONNX...")
                engine = build_engine_from_onnx(onnx_path, engine_path, fp16=True)
            context = engine.create_execution_context()
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            return (engine, context, inputs, outputs, bindings, stream)
        else:
            return load_onnx_session("best.onnx", use_gpu=True)


# Inference Helper Functions
cpdef np.ndarray run_inference(object engine_obj, np.ndarray input_data):
    """
    Runs inference using either ONNX Runtime or TensorRT depending on engine_obj.
    """
    cdef np.ndarray out
    if isinstance(engine_obj, tuple):
        engine = engine_obj[0]
        context = engine_obj[1]
        inputs = engine_obj[2]
        outputs = engine_obj[3]
        bindings = engine_obj[4]
        stream = engine_obj[5]
        out = run_inference_trt(context, bindings, inputs, outputs, stream, input_data, engine)
    else:
        session = engine_obj
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: input_data})[0]
    out = np.squeeze(out, axis=0)
    out = np.transpose(out, (1, 0))
    return out

cpdef tuple parse_detections(np.ndarray out, double conf_thresh=0.15):
    """
    Given an output of shape (8400,5) [x_center, y_center, width, height, confidence],
    returns (x_center, y_center, confidence) for the highest-confidence detection
    above conf_thresh, or None if no detection qualifies.
    """
    valid = out[out[:, 4] >= conf_thresh]
    if valid.shape[0] == 0:
        return None
    i_best = int(np.argmax(valid[:, 4]))
    best = valid[i_best]
    return (best[0], best[1], best[4])

# Screen Capture & Preprocessing
cpdef np.ndarray capture_and_preprocess():
    """
    DxCam capturing, converts to BGR (3 channels),
    resizes to 640x640, normalizes pixel values to [0,1],
    and transposes the image to (1,3,640,640) for model input.
    """
    init_camera()
    img = CAMERA.grab()
    
    if img is None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
    else:
        # Ensure image has 3 channels (BGR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Proceed with preprocessing
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


# Mouse Movement using mouse_event
cpdef void send_mouse_relative(int dx, int dy):
    """
    Sends a relative mouse movement event with the given deltas (dx, dy) in pixels.
    """
    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)


# Key & Screen Helper Functions
cpdef bint is_key_pressed(str aim_key):
    """
    Checks if the specified key (e.g., "RMB") is pressed.
    """
    user32 = ctypes.windll.user32
    vk = 0x02 if aim_key.upper() == "RMB" else ord(aim_key.upper()[0])
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)

cpdef tuple get_screen_dimensions():
    """
    Returns the (width, height) of the primary screen.
    """
    user32 = ctypes.windll.user32
    return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))

# TODO: Actually work on this
cpdef void draw_circle(np.ndarray image, int cx, int cy, int radius, tuple color=(0, 255, 0), int thickness=2):
    """
    Draws a circle on the given image.
    
    Parameters:
      image: a NumPy ndarray (BGR image) on which the circle will be drawn.
      cx, cy: center coordinates of the circle.
      radius: the radius of the circle.
      color: a tuple for the circle color in BGR format (default is green).
      thickness: line thickness (default is 2).
    """
    cv2.circle(image, (cx, cy), radius, color, thickness)


# Main Aim Alignment Function

cpdef void run_aim_alignment(object aim_key,
                              int fov_radius,
                              bint fov_enabled,
                              double ai_confidence,
                              double sensitivity,
                              bint gpu,
                              bint use_trt,
                              double x_offset,
                              double y_offset):
    """
    Parameters:
      aim_key: The key to enable aim alignment, Hard coded "RMB".
      fov_radius: Field-of-view radius in pixels.
      fov_enabled: True if FOV constraint is enabled.
      ai_confidence: Confidence threshold for detections.
      sensitivity: Scaling factor for mouse movement.
      gpu: True to use GPU.
      use_trt: True to use TensorRT.
      x_offset, y_offset: Offsets for aim placement relative to the detected object's center.
    """
    cdef object engine_obj
    cdef int sw, sh, cx, cy
    cdef double scale_x, scale_y
    cdef np.ndarray input_data, out
    cdef int max_detections = 5
    cdef double INF = 1e10

    # Declare loop-scope variables
    cdef int closest_tx, closest_ty, detection_count, scaled_dx, scaled_dy
    cdef double min_distance, conf, x1, y1, x2, y2
    cdef int detection_center_x, detection_center_y
    cdef double dx, dy, distance, scaling_factor

    engine_obj = setup_inference(gpu, use_trt)
    sw, sh = get_screen_dimensions()
    cx, cy = sw // 2, sh // 2
    scale_x = float(sw) / 640.0
    scale_y = float(sh) / 640.0

    print(f"Screen: {sw}x{sh}, Center: ({cx},{cy})")
    print(f"FOV: {fov_enabled} (BROKEN) ({fov_radius}px)")
    print(f"Confidence: {ai_confidence:.2f}")
    print(f"Hold {aim_key} to activate...")

    while True:
        closest_tx = -1
        closest_ty = -1
        min_distance = INF
        detection_count = 0

        if is_key_pressed(aim_key):
            input_data = capture_and_preprocess()
            out = run_inference(engine_obj, input_data)

            for i in range(out.shape[0]):
                if detection_count >= max_detections:
                    break
                conf = out[i, 4]
                if conf < ai_confidence:
                    continue

                # Extract coordinates (with scaling)
                x1 = <double>out[i, 0] * scale_x
                y1 = <double>out[i, 1] * scale_y
                x2 = x1 + (<double>out[i, 2]) * scale_x
                y2 = y1 + (<double>out[i, 3]) * scale_y

                detection_center_x = <int>((x1 + x2) / 2)
                detection_center_y = <int>(y1 + (y2 - y1) * 0.5)

                dx = detection_center_x - cx
                dy = detection_center_y - cy
                distance = sqrt(dx*dx + dy*dy)

                if fov_enabled and distance > fov_radius:
                    continue

                if distance < min_distance:
                    min_distance = distance
                    closest_tx = detection_center_x
                    closest_ty = detection_center_y

                detection_count += 1

        if closest_tx != -1:
            # Get raw differences between target and screen center.
            dx = closest_tx - cx
            dy = closest_ty - cy

            # Apply X and Y offset differences
            dx += x_offset
            dy += y_offset

            scaling_factor = sensitivity
            scaled_dx = int(dx * scaling_factor)
            scaled_dy = int(dy * scaling_factor)

            send_mouse_relative(scaled_dx, scaled_dy)
            print(f"Adjusting aim by: ({scaled_dx}, {scaled_dy})")

        debug_img = np.zeros((sh, sw, 3), dtype=np.uint8)
        draw_circle(debug_img, cx, cy, fov_radius, (0, 0, 255), 2) # Not working
