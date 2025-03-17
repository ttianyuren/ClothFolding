import numpy as np
import time
import math
import genesis as gs
from typing import Tuple

gs.init()
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(1280, 720),
        max_FPS=60,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

cloth= scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=2.0,
        pos=(0, 0, 0.5),
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0),
        vis_mode="visual",
    ),
)
scene.build()
def curr_time():
    """Return the time tick in milliseconds."""
    return time.monotonic() * 1000
fixed_points = [
    (-1, 1, 1.0),
    (-1, -1, 1.0)
]
x_ref = np.array([
    [-1, 1, 1.0],  # Corner 3 moves to Corner 1
    [-1, -1, 1.0],    # Corner 4 moves to Corner 2
])
for point in fixed_points:
    particle = cloth.find_closest_particle(point)
    cloth.fix_particle(particle)


class MPPI():
    def __init__(self,u_prev,horizon_step_T,param_exploration=0.01):
        self.u_prev = u_prev
        self.flag_done = False
        self.K = 10
        self.T = horizon_step_T
        sigma_value = 0.1  # Variance of each control input (adjustable)
        self.dim_u = 6
        self.Sigma = np.diag([sigma_value] *self.dim_u )
        self.param_exploration = param_exploration

        
    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """Search the closest waypoint to the vehicle on the reference path."""
        SEARCH_IDX_LEN = 200 # [points] forward search range
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx
        # get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_yaw = self.ref_path[nearest_idx,2]
        ref_v = self.ref_path[nearest_idx,3]
        # update nearest waypoint index if necessary
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx 

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v
    
    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """Sample epsilon_(t) in Step 1 for each of the control dimension."""
        # check if sigma row size == sigma col size == size_dim_u and size_dim_u > 0
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma / covariance matrix must be a square matrix with the size of size_dim_u.")
            raise ValueError
        # sample epsilon
        mu = np.zeros((size_dim_u)) # set average as a zero vector
        # For each of the control 
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step)) 
        return epsilon

    def calc_control_input(self, observed_x: np.ndarray):
        """
        Compute the optimal control input using Model Predictive Path Integral (MPPI).
        """
        # Load previous control input sequence
        u = self.u_prev
        x0 = observed_x  # Set initial x value from observation

        # Check if cloth is close enough to reference configuration
        cloth_error = np.linalg.norm(x0 - x_ref)
        completion_threshold = 0.05  

        if cloth_error < completion_threshold:
            self.flag_done = True
            return (), [], [], [], self.flag_done

        # Buffer for cost evaluation
        S = np.zeros(self.K) 

        # Step 1: Sample noise epsilon
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        # Step 2: Predict future states and evaluate cost
        v = np.zeros((self.K, self.T, self.dim_u))  

        for k in range(self.K):         
            x = x0  
            for t in range(1, self.T+1):
                if k < (1.0 - self.param_exploration) * self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1]  # Exploitation
                else:
                    v[k, t-1] = epsilon[k, t-1]  # Exploration
                x = self._F(x, v[k, t-1])
                S[k] += self.compute_cloth_cost(x, v[k, t-1], u[t-1]) 
            S[k] += self.compute_cloth_cost(x, 0, 0)  # Terminal cost
            self.reset(x0,x) 

        # Step 3: Compute information-theoretic weights
        w = self._compute_weights(S)  
        w_epsilon = np.zeros((self.T, self.dim_u))  

        for t in range(self.T):
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        # Apply moving average filter
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)
        u += w_epsilon

        # Update previous control input sequence (shift left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        return u[0], u, self.flag_done
    
    def reset(self,x0,x):
        for step in range(5):
            for i in range(2):
                particle = cloth.find_closest_particle(tuple(x[i]))
                cloth.set_particle_position(particle,x0[i])
        scene.step()
    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """
        Predicts the next state of the cloth using Genesis' force application function.

        Args:
            x_t (np.ndarray): Current state [positions, velocities].
            v_t (np.ndarray): Control input (forces applied to grip points).
            dt (float): Time step for integration.

        Returns:
            np.ndarray: Predicted next state [positions, velocities].
        """
        
        x_t_plus_1 = x_t.copy()  # Initialize next state
        for step in range(5):
            for i in range(2):  # Iterate for selected particles
                particle_position = tuple(x_t[i])  
                particle = cloth.find_closest_particle(particle_position)   
                velocity = v_t[i * 3:(i + 1) * 3]  # Extract corresponding velocity
                cloth.set_particle_velocity(particle, velocity)

            scene.step()  # Advance simulation

        for i in range(2):
            particle_position = tuple(x_t[i])  
            particle = cloth.find_closest_particle(particle_position)   
            velocity = np.array([0,0,0])  # Extract corresponding velocity
            cloth.set_particle_velocity(particle, velocity)
        for i in range(2):
            particle_pos = tuple(x_t[i])
            particle = cloth.find_closest_particle(particle_pos)
            pos = cloth.get_particles()
            x_t_plus_1[i] = pos[particle]
        return x_t_plus_1
        


    def _cost_jit(self, x_t: np.ndarray) -> float:
        """Calculate state cost using JIT-compiled function."""
        pass
    
    def compute_cloth_cost(self,x_t: np.ndarray, v_t: np.ndarray, prev_v_t: np.ndarray, 
                       w_tracking=1.0, w_control=0.1, w_smoothness=0.05) -> float:
        """
        Computes the cost function for cloth folding in MPPI.

        Args:
            x_t (np.ndarray): Current state [positions of key points].
            v_t (np.ndarray): Current control input (applied velocities).
            x_ref (np.ndarray): Reference positions for key points.
            prev_v_t (np.ndarray): Previous control input (to penalize abrupt changes).
            w_tracking (float): Weight for tracking error.
            w_control (float): Weight for minimizing control effort.
            w_smoothness (float): Weight for minimizing jerk.

        Returns:
            float: Cost value for this trajectory sample.
        """
        
        # Tracking cost: Sum of squared distances from the reference points
        J_tracking = np.sum(np.linalg.norm(x_t - x_ref, axis=1) ** 2)

        # Control effort cost: Sum of squared velocity inputs (penalizing excessive velocity)
        J_control = np.sum(v_t ** 2)

        # Smoothness cost: Sum of squared differences between successive velocity commands
        J_smoothness = np.sum((v_t - prev_v_t) ** 2)

        # Weighted sum of all cost terms
        total_cost = (w_tracking * J_tracking) + (w_control * J_control) + (w_smoothness * J_smoothness)

        return total_cost


    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """Compute weights for each sample."""
        # prepare buffer
        w = np.zeros((self.K))
        # calculate rho
        rho = S.min()
        # calculate eta
        eta = 0.0
        for k in range(self.K):
            eta += np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        # calculate weight
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        return w
    
    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply moving average filter for smoothing input sequence.

        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean
    
    def _g(self, v: np.ndarray) -> float:
        """Clamp input signals. This is a hard-constraint."""
        if len(v) > 2:
            raise ValueError("Error: More than two inputs are not supported yet.")  # noqa: EM101
        # limit control inputs to the left and right wheel velocities
        v[0] = np.clip(v[0], -self.abs_vb_val, self.abs_vb_val) 
        v[1] = np.clip(v[1], -self.abs_omega_val, self.abs_omega_val) 
        #v[1] = np.clip(v[1], 0, self.abs_omega_val) 
        return v
    def _phi(self, x_T: np.ndarray) -> float:
        """Calculate terminal cost."""
        # parse x_T
        x, y, yaw = x_T # In diffdrive, only three states
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]
        v = self.vehicle_model.curr_body_vel
        # calculate terminal cost
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                        self.terminal_cost_weight[2]*(yaw-ref_yaw)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2
        
        # add penalty for collision with obstacles
        terminal_cost += self._is_collided(x_T) * 1.0e10
        return terminal_cost
    def _is_collided(self,  x_t: np.ndarray) -> bool:
        """Check if the vehicle is collided with obstacles."""
        # vehicle shape parameters
        vw, vl = self.vehicle_w, self.vehicle_l
        safety_margin_rate = self.collision_safety_margin_rate
        vw, vl = vw*safety_margin_rate, vl*safety_margin_rate
        # get current states
        x, y, yaw = x_t
        # key points for collision check
        vehicle_shape_x = [-0.5*vl, -0.5*vl, 0.0, +0.5*vl, +0.5*vl, +0.5*vl, 0.0, -0.5*vl, -0.5*vl]
        vehicle_shape_y = [0.0, +0.5*vw, +0.5*vw, +0.5*vw, 0.0, -0.5*vw, -0.5*vw, -0.5*vw, 0.0]
        rotated_vehicle_shape_x, rotated_vehicle_shape_y = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [x, y]) # make the vehicle be at the center of the figure
        # check if the key points are inside the obstacles
        for obs in self.obstacle_circles: # for each circular obstacles
            obs_x, obs_y, obs_r = obs # [m] x, y, radius
            for p in range(len(rotated_vehicle_shape_x)):
                if (rotated_vehicle_shape_x[p]-obs_x)**2 + (rotated_vehicle_shape_y[p]-obs_y)**2 < obs_r**2:
                    return 1.0 # collided
        return 0.0 # not collided
    def _affine_transform(self, xlist: list, ylist: list, angle: float, translation: list=[0.0, 0.0]) -> Tuple[list, list]:
        """Rotate shape and return location on the x-y plane."""
        transformed_x = []
        transformed_y = []
        if len(xlist) != len(ylist):
            print("[ERROR] xlist and ylist must have the same size.")
            raise AttributeError
        for i, xval in enumerate(xlist):
            transformed_x.append((xlist[i])*np.cos(angle)-(ylist[i])*np.sin(angle)+translation[0])
            transformed_y.append((xlist[i])*np.sin(angle)+(ylist[i])*np.cos(angle)+translation[1])
        transformed_x.append(transformed_x[0])
        transformed_y.append(transformed_y[0])
        return transformed_x, transformed_y
    


# Initialize MPPI controller
horizon_step_T = 20  # Prediction horizon
u_prev = np.zeros((horizon_step_T, 6))  # Initialize previous control inputs

mppi = MPPI(u_prev, horizon_step_T)

# Initial state of the cloth
observed_x =np.array([
    [1, 1, 1.0],
    [1, -1, 1.0],
])  # Replace with actual cloth state initialization

# Run MPC for 1000 steps
for step in range(1000):
    optimal_u, _, flag_done = mppi.calc_control_input(observed_x)

    if flag_done:
        print(f"Cloth reached target configuration at step {step}")
        break

    # Apply the first control input from MPPI
    observed_x = mppi._F(observed_x, mppi._g(optimal_u), mppi.delta_t)  # Simulate next state

    print(f"Step {step}, Applied Control: {optimal_u}")

while True:
    scene.step()
