import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
import numpy as np

# Load the model and data
model = mujoco.MjModel.from_xml_path("robo.xml")
data = mujoco.MjData(model)
NUM_REPEATS = 4
SIM_DURATION = 7  # seconds
PERTURBATION = 1e-7
n_steps = int(SIM_DURATION / model.opt.timestep)

ended = False


def key_callback(keycode):
    if keycode == 256:  # Escape key in GLFW
        global ended
        ended = True


dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

# Launch the passive viewer
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    # Set initial state to the 'spinning' keyframe if it exists
    for _ in range(NUM_REPEATS):
        if not viewer.is_running() or ended:
            break

        timevals = np.zeros(n_steps)
        angular_velocity = np.zeros((n_steps, 3))
        stem_height = np.zeros(n_steps)

        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            # # perturb initial velocities
            data.qvel[:] += PERTURBATION * np.random.randn(model.nv)
        else:
            mujoco.mj_resetData(model, data)

        for i in range(n_steps):
            if not viewer.is_running() or ended:
                break
            step_start = time.time()
            # Step the simulation
            mujoco.mj_step(model, data)
            timevals[i] = data.time
            angular_velocity[i] = data.qvel[3:6].copy()
            stem_height[i] = data.geom_xpos[2, 2]

            # Synchronize the viewer
            viewer.sync()

            # Maintain real-time simulation
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        ax[0].plot(timevals, angular_velocity)
        ax[1].plot(timevals, stem_height)


ax[0].set_title("angular velocity")
ax[0].set_ylabel("radians / second")

ax[1].set_xlabel("time (seconds)")
ax[1].set_ylabel("meters")
_ = ax[1].set_title("stem height")
plt.show()
