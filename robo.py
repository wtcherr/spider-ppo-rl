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

print(f"# of generalized coordinates = dim(qpos):{model.nq}")
print(f"# of degrees of freedom = dim(qvel):{model.nv}")
print(f"# of actuators/controls = dim(ctrl):{model.nu}")
print(f"# of activation states = dim(act):{model.na}")
print(f"# of bodies:{model.nbody}")
print(f"# of joints:{model.njnt}")
print(f"# of joint:{model.njnt}")

mujoco.mj_resetData(model, data)  # Reset state and time.
print(data.qpos[: model.nq])
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while data.time < SIM_DURATION:
        if not viewer.is_running() or ended:
            break
        step_start = time.time()
        mujoco.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = int(data.time % 2)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1 - int(data.time % 2)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = int(data.time % 2)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(data.time % 2)

        # Synchronize the viewer
        viewer.sync()

        # Maintain real-time simulation
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
