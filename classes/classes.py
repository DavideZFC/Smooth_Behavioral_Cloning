import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense


class irregular_policy:
    '''
        An hand-crafted policy for Muontain Car designed not to satisfy the Lipschitzness assumption
    '''
    def __init__(self):
        self.hex = 0

    def predict(self, state):
        try:
            if (state[0]<0 and state[1]<0):
                return np.array([-1.0])
            return np.array([1.0]).reshape(-1,1)
        except:
            N = state.shape[0]
            ret = np.zeros((N,1))
            for i in range(N):
                ret[i,0] = self.predict(state[i,:])
            return ret



class Actor_Model:
    def __init__(self, input_shape, action_space):
        X_input = Input(input_shape)
        self.action_space = action_space
        
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="tanh")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        #self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))

    def predict(self, state):
        return self.Actor.predict(state)


class pytorchActor:
    def __init__(self, actor):
        self.model = PerceptronNet(inp = 24, out = 4, layers=[512,256,64], final_tanh=True)
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
    
        h = 0
        with torch.no_grad():
            for param in self.model.parameters():
                param.copy_(torch.from_numpy(actor.Actor.weights[h].numpy().T))
                h += 1

    def save(self, name):
        name = name+'.pt'
        torch.save(self.model, name)

    def predict(self, state):
        # x_tensor = torch.FloatTensor(x).unsqueeze(1).to(self.device)
        x_tensor = torch.FloatTensor(state).to(self.device)
        input = Variable(x_tensor).to(self.device).requires_grad_()
        pred = self.model(input).to('cpu').detach().numpy()

        return pred


class PerceptronNet(nn.Module):
    def __init__(self, inp=3, out=1, layers=[50, 50], final_tanh = False):
        super(PerceptronNet, self).__init__()
        
        layers_list = [nn.Linear(inp, layers[0]),
        nn.ReLU()]
        for i in range(1,len(layers)):
            layers_list.append(nn.Linear(layers[i-1], layers[i]))
            layers_list.append(nn.ReLU())

        layers_list.append(nn.Linear(layers[-1], out))
        if final_tanh:
            layers_list.append(nn.Tanh())
        self.layer = Sequential(*layers_list)

    def forward(self, x):
        return self.layer(x)




class Regression():
    def __init__(self, inp, out, layers=[50,50]):

        self.iter = 0
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.inp = inp
        self.out = out
        self.layers = layers

    def train(self, X, y, epochs=10, lr=0.01, bs=100, weight_decay=0):
        X_train=torch.FloatTensor(X)
        y_train=torch.FloatTensor(y)

        self.model = PerceptronNet(self.inp, self.out, layers=self.layers).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        train = torch.utils.data.TensorDataset(X_train,y_train)

        train_loader = torch.utils.data.DataLoader(train, batch_size = bs, shuffle = True)

        for e in range(epochs):

            running_loss = 0.0

            for inputs, labels in train_loader:

                X = Variable(inputs).to(self.device)
                #Y = Variable(labels).unsqueeze(1).to(self.device)
                Y = Variable(labels).to(self.device)


                output = self.model(X)
                
                loss = criterion(output, Y).requires_grad_()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() 
            
            #print(running_loss)

        self.final_loss = running_loss
                            
    def prediction(self, x):
        # x_tensor = torch.FloatTensor(x).unsqueeze(1).to(self.device)
        x_tensor = torch.FloatTensor(x).to(self.device)
        input = Variable(x_tensor).to(self.device).requires_grad_()
        pred = self.model(input).to('cpu').detach().numpy()

        return pred

    def save(self, name):
        name = name+'.pt'
        torch.save(self.model, name)




import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, intrinsic_noise = 0.0, gravity=9.8):
        self.gravity = gravity #20
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.lim_steps = 200
        self.steps = 0

        self.intrinsic_noise = intrinsic_noise

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        action = action + np.random.normal(loc=0, scale=self.intrinsic_noise)
        action = np.clip(action, -1.0, 1.0)
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians \
            or self.steps > self.lim_steps
        done = bool(done)
        self.steps += 1

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.steps = 0
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()
