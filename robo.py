import mujoco
import mujoco.viewer
import time

SIM_DURATION = 100  # seconds


def key_callback(keycode):
    if keycode == 256:  # Escape key in GLFW
        global ended
        ended = True


ended = False
# Load the model and data
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)  # Reset state and time.
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while data.time < SIM_DURATION:
        if not viewer.is_running() or ended:
            break
        step_start = time.time()
        mujoco.mj_step(model, data)

        # Synchronize the viewer
        viewer.sync()

        # Maintain real-time simulation
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
