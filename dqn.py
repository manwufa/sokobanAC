from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os
import glfw
import numpy as np
from OpenGL.GLU import *
from OpenGL.GL import *

class Viewer():

    def __init__(self, sim):
        self.sim = sim

        if not glfw.init():
            return
        # Create a windowed mode window and its OpenGL context
        self._window = glfw.create_window(640, 480, "Hello World", None, None)
        if not self._window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self._window)

    def render(self):
        # Make the window's context current
        glfw.make_context_current(self._window)
        if glfw.window_should_close(self._window):
            glfw.terminate()

        self.image = self.sim.render(640,480).astype(np.float) / 255.0

        # Render here, e.g. using pyOpenGL
        self.texture = glGenTextures(1)
        glEnable(GL_TEXTURE_2D)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glTexImage2D(GL_TEXTURE_2D, 0,
                     GL_RGB,
                     self.image.shape[1], self.image.shape[0], 0,
                     GL_RGB,
                     GL_FLOAT,
                     self.image)
        glDisable(GL_TEXTURE_2D)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0., 0., 0., 0.)
        glClearDepth(1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glEnable(GL_TEXTURE_2D)
        # draw a textured quad, shrink it a bit so the edge is clear
        glBegin(GL_QUADS)
        glTexCoord2f(0., 0.)
        glVertex3f(-0.9, -0.9, 0.)
        glTexCoord2f(1., 0.)
        glVertex3f(0.9, -0.9, 0.)
        glTexCoord2f(1., 1.)
        glVertex3f(0.9, 0.9, 0.)
        glTexCoord2f(0., 1.)
        glVertex3f(-0.9, 0.9, 0.)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # Swap front and back buffers
        glfw.swap_buffers(self._window)

        # Poll for and process events
        glfw.poll_events()
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

model = load_model_from_path("/home/wf/Downloads/mujoco-py-master/xmls/tosser.xml")
sim = MjSim(model)

viewer = Viewer(sim)


sim_state = sim.get_state()

while True:
    sim.set_state(sim_state)

    for i in range(1000):
        if i < 150:
            sim.data.ctrl[:] = 0.0
        else:
            sim.data.ctrl[:] = -1.0
        sim.step()
        viewer.render()

    if os.getenv('TESTING') is not None:
        break