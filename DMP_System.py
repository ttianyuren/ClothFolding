import numpy as np
import matplotlib.pyplot as plt


class SystemOptions:
    def __init__(self):
        self.time_zoom = 1  # Time scaling factor
        self.dt = 0.05  # Time step

        self.start = 0  # Start position
        self.goal = 1  # Goal position
        self.scale_zoom = self.goal - self.start  # Scale factor for nonlinear terms
        self.originSys_alpha_y = 25  # Alpha parameter for the original system
        self.originSys_beta_y = 6  # Beta parameter for the original system

        self.nonlinearSys_n_bfs = 10  # Number of basis functions for the nonlinear system
        self.nonlinearSys_alpha_x = 8  # Alpha parameter for the nonlinear system


class OriginSystem:
    """
    Defines the original system dynamics:
    ydd = alpha_y * (beta_y * (goal - y) - yd)
    yd += tau * ydd * dt
    y += tau * yd * dt
    tau > 1 indicates accelerated simulation, tau < 1 indicates decelerated simulation.
    """

    def __init__(self, sys_option: SystemOptions):
        """
        Initializes the original system with parameters.
        """
        self.tau = sys_option.time_zoom
        self.dt = sys_option.dt
        self.alpha_y = sys_option.originSys_alpha_y
        self.beta_y = sys_option.originSys_beta_y

        self.ydd = 0  # Acceleration
        self.yd = 0  # Velocity
        self.y = sys_option.start  # Position
        self.goal = sys_option.goal  # Goal position

    def prepare_step(self, nonlinear_term):
        """
        Computes the acceleration.
        """
        self.ydd = self.alpha_y * (self.beta_y * (self.goal - self.y) - self.yd) + nonlinear_term

    def run_step(self):
        """
        Integrates velocity and position using acceleration. tau is always used with dt.
        """
        self.y += self.yd * self.tau * self.dt
        self.yd += self.ydd * self.tau * self.dt

    def test(self):
        """
        Tests the system under different time scales. Convergence behavior should be similar due to scaling.
        """
        t = np.linspace(self.dt, 3, 30)  # Simulate from 0 to 3 seconds
        self.tau = 1 / 3  # Adjust tau for the changed time scale
        y_log = []
        for i in range(30):
            y_log.append(self.y)
            self.prepare_step(0)
            self.run_step()
        plt.plot(t, y_log)
        plt.show()


class NonlinearSystem:
    """
    Defines the nonlinear system dynamics:
    phi_i(x) = exp(-0.5 * (x - c_i) ** 2 * D)
    xd = -alpha_x * x (canonical dynamical system)
    x += tau * xd * dt
    """

    def __init__(self, sys_option: SystemOptions):
        self.tau = sys_option.time_zoom
        self.dt = sys_option.dt
        self.n_bfs = sys_option.nonlinearSys_n_bfs
        self.alpha_x = sys_option.nonlinearSys_alpha_x
        self.scale_zoom = sys_option.scale_zoom

        self.basic_fun_mean_time = np.linspace(0, 1, self.n_bfs + 2)[1:-1]  # Basis function means in time
        self.basic_fun_mean_canonical = np.exp(-self.alpha_x * self.basic_fun_mean_time)  # Means in canonical space
        self.basic_fun_weight = np.random.random(self.n_bfs)  # Random weights for basis functions
        self.sx2 = np.ones(self.n_bfs)  # Used for imitation learning
        self.sxtd = np.ones(self.n_bfs)
        self.basic_fun_var = (np.diff(self.basic_fun_mean_canonical) * 0.55) ** 2
        self.basic_fun_var = 1 / np.hstack((self.basic_fun_var, self.basic_fun_var[-1]))  # Variance of basis functions

        self.x = 1  # Initial position
        self.xd = 0  # Initial velocity

    def set_basic_fun_weight(self, weight):
        self.basic_fun_weight = weight

    def get_basic_fun_weight(self):
        return self.basic_fun_weight

    def prepare_step(self):
        self.xd = -self.alpha_x * self.x

    def run_step(self):
        self.x += self.xd * self.tau * self.dt

    def get_psi_now(self):
        psi = np.exp(-0.5 * (self.x - self.basic_fun_mean_canonical) ** 2 * self.basic_fun_var)
        return psi

    def calc_nonlinear_term(self):
        f = self.basic_fun_weight.dot(self.calc_g())
        return f

    def calc_g(self):
        psi = self.get_psi_now()
        g = psi / np.sum(psi + 1e-10) * self.x * self.scale_zoom
        return g

    def test(self):
        """
        Plots the distribution of basis functions over time.
        """
        start_time = 0
        end_time = 10
        self.tau = 1 / 10  # Adjust tau for the changed time scale
        t = np.linspace(start_time, end_time, int((end_time - start_time) // self.dt))
        y = []
        for i in range(int((end_time - start_time) // self.dt)):
            y.append(self.get_psi_now())
            self.prepare_step()
            self.run_step()
        plt.plot(t, y)
        plt.show()


class DMPOptions:
    def __init__(self):
        self.start_sys_time = 0.0  # Start time for the system
        self.end_sys_time = 2.0  # End time for the system
        self.start_dmp_time = 0.0  # Start time for the DMP
        self.end_dmp_time = 1.0  # End time for the DMP


class DMPSystem:
    """
    DMP system dynamics:
    Within dmp_time range, f_term equals calc_f_term.
    Outside this range, f_term equals 0.
    """

    def __init__(self, sys_option: SystemOptions, dmp_option: DMPOptions):
        self.dt = sys_option.dt
        self.sys_option = sys_option
        self.dmp_option = dmp_option
        sys_option.tau = 1 / (dmp_option.end_dmp_time - dmp_option.start_dmp_time)  # Adjust tau based on DMP time range
        self.n_step = int((dmp_option.end_sys_time - dmp_option.start_sys_time) // sys_option.dt)  # Number of steps
        self.origin_sys = OriginSystem(sys_option)
        self.nonlinear_sys = NonlinearSystem(sys_option)

        self.t = 0  # Time
        self.y = 0  # Position
        self.yd = 0  # Velocity
        self.ydd = 0  # Acceleration
        self.x = 1  # Canonical system variable
        self.xd = 0  # Canonical system velocity
        self.psi = np.zeros(sys_option.nonlinearSys_n_bfs)  # Basis function values
        self.basic_fun_weight = np.zeros(sys_option.nonlinearSys_n_bfs)  # Weights for the basis functions
        self.nonlinear_term = 0  # Nonlinear term value

    def run_step(self, has_dmp):
        if has_dmp:
            self.nonlinear_term = self.nonlinear_sys.calc_nonlinear_term()
            self.nonlinear_sys.prepare_step()
            self.nonlinear_sys.run_step()
            self.origin_sys.prepare_step(self.nonlinear_term)
            self.origin_sys.run_step()
        else:
            self.nonlinear_term = 0.0
            self.origin_sys.prepare_step(self.nonlinear_term)
            self.origin_sys.run_step()
        self.y = self.origin_sys.y
        self.yd = self.origin_sys.yd
        self.ydd = self.origin_sys.ydd
        self.x = self.nonlinear_sys.x
        self.xd = self.nonlinear_sys.xd
        self.psi = self.nonlinear_sys.get_psi_now()
        self.basic_fun_weight = self.nonlinear_sys.get_basic_fun_weight()
        self.t += self.dt

    def run_trajectory(self):
        y = np.zeros(self.n_step)
        t = np.zeros(self.n_step)
        for i in range(self.n_step):
            if self.dmp_option.start_dmp_time <= self.t <= self.dmp_option.end_dmp_time:
                self.run_step(has_dmp=True)
            else:
                self.run_step(has_dmp=False)
            y[i] = self.y
            t[i] = self.t
        return y, t

    def run_fit_trajectory(self, target: np.ndarray, target_d: np.ndarray, target_dd: np.ndarray):
        """
        Fits a trajectory to the given target position, velocity, and acceleration.
        """
        y0 = target[0]
        g = target[-1]

        X = np.zeros(target.shape)
        G = np.zeros(target.shape)
        x = 1

        for i in range(len(target)):
            X[i] = x
            G[i] = g
            xd = -self.nonlinear_sys.alpha_x * x
            x += xd * self.sys_option.time_zoom * self.dt

        self.origin_sys.scale_zoom = g - y0
        F_target = (target_dd / (self.sys_option.time_zoom ** 2) - self.origin_sys.alpha_y * (
                self.origin_sys.beta_y * (G - target) - target_d / self.sys_option.time_zoom))
        PSI = np.exp(
            -0.5 * ((X.reshape((-1, 1)).repeat(self.nonlinear_sys.n_bfs,
                                               axis=1) - self.nonlinear_sys.basic_fun_mean_canonical.reshape(1, -1)
                     .repeat(target.shape, axis=0)) ** 2) * (
                self.nonlinear_sys.basic_fun_var.reshape(1, -1).repeat(target.shape, axis=0)))
        X *= self.origin_sys.scale_zoom
        self.nonlinear_sys.sx2 = ((X * X).reshape((-1, 1)).repeat(self.nonlinear_sys.n_bfs, axis=1) * PSI).sum(axis=0)
        self.nonlinear_sys.sxtd = ((X * F_target).reshape((-1, 1)).repeat(self.nonlinear_sys.n_bfs, axis=1) * PSI).sum(axis=0)
        self.nonlinear_sys.basic_fun_weight = self.nonlinear_sys.sxtd / (self.nonlinear_sys.sx2 + 1e-10)
        self.set_weight(self.nonlinear_sys.basic_fun_weight)

    def calc_g(self):
        return self.nonlinear_sys.calc_g()

    def set_weight(self, weight):
        self.basic_fun_weight = weight
        self.nonlinear_sys.set_basic_fun_weight(weight)

    def test1(self):
        """
        Tests the DMP system running a trajectory.
        """
        self.set_weight(1000 * np.random.random(self.basic_fun_weight.shape))
        y, t = self.run_trajectory()
        plt.plot(t, y)
        plt.show()

    def test2(self):
        """
        Tests the DMP system imitating a given trajectory.
        """
        target_trajectory = np.linspace(0, 1, self.n_step)
        target_trajectory_d = np.linspace(0.2, 0.2, self.n_step)  # Constant velocity
        target_trajectory_dd = np.linspace(0, 0, self.n_step)  # Zero acceleration
        self.run_fit_trajectory(target_trajectory, target_trajectory_d, target_trajectory_dd)
        y, t = self.run_trajectory()
        plt.plot(t, y)
        plt.plot(t, target_trajectory, '-')
        plt.legend(['DMPs', 'desired'])
        plt.show()


if __name__ == '__main__':
    system_option = SystemOptions()
    dmp_option = DMPOptions()

    # Uncomment to test different components of the system
    # origin_term = OriginSystem(system_option)
    # origin_term.test()

    # nonlinear_term = NonlinearSystem(system_option)
    # nonlinear_term.test()

    # dmps = DMPSystem(system_option, dmp_option)
    # dmps.test1()

    dmps = DMPSystem(system_option, dmp_option)
    dmps.test2()
