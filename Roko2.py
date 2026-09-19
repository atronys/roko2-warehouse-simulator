import math, random, Parameters, numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

class Roko2:
    # Robot constant parameters — warehouse configuration
    _mass = 8.0          # robot mass, kg
    _rw = 0.08           # wheels radius, m
    _lx = 0.8            # length of the robot, m
    _ly = 0.5            # width of the robot, m
    _lz = 0.2            # height of the robot, m
    _wheelbase = 0.6     # distance between front and rear axles, m
    _max_steering_angle = 0.8  # max steering angle, rad (~46 deg, R_min=0.58m)
    _mw = 0.2            # wheel mass, kg
    # Moments of inertia (simplified)
    _Jy = 0.25 * _mass * _lx**2 + 1/12 * _mass * _lz**2

    # Robot motion parameters
    _x = 0
    _y = 0
    _heading = 0
    _velocity = 0
    _angular_rate = 0
    _steering_angle = 0   # actual front-wheel steering angle

    # Robot control values
    _drive_force = 0      # longitudinal force, N
    _steering_rate = 0    # steering angular velocity, rad/s

    # Robot PD control variables
    _velocity_error = 0
    _steering_error = 0

    # System variables
    _dt = 0.1
    _random_seed = 0

    # Arrays to store motion parameters at every simulation step
    _X = []
    _Y = []
    _Heading = []
    _Velocity = []
    _Angular_rate = []
    _Time = []


    def update(self):
        # Steering dynamics (first-order lag)
        self._steering_angle += self._steering_rate * self._dt
        self._steering_angle = max(-self._max_steering_angle,
                                   min(self._max_steering_angle, self._steering_angle))

        # Linear acceleration (F = ma)
        linear_acceleration = self._drive_force / self._mass

        # Angular rate from bicycle/ackermann model: omega = v * tan(delta) / L
        self._angular_rate = self._velocity * math.tan(self._steering_angle) / self._wheelbase

        # Update pose
        self._heading += self._angular_rate * self._dt
        self._x += self._velocity * math.cos(self._heading) * self._dt
        self._y += self._velocity * math.sin(self._heading) * self._dt

        # Update velocity
        self._velocity += linear_acceleration * self._dt
        self._velocity = max(-0.8, min(0.8, self._velocity))

        # Limit heading angle to [-pi, pi] interval
        if abs(self._heading) > math.pi:
            self._heading -= numpy.sign(self._heading) * 2 * math.pi

        self._random_seed += 1

        # Collect logs for further visualization
        self._X.append(self._x)
        self._Y.append(self._y)
        self._Heading.append(self._heading * 180.0 / math.pi)
        self._Velocity.append(self._velocity)
        self._Angular_rate.append(self._angular_rate)
        if len(self._Time) == 0:
            self._Time.append(self._dt)
        else:
            self._Time.append(self._Time[-1] + self._dt)


    def __init__(self, x, y, heading, velocity, angular_rate):
        self._x = x
        self._y = y
        self._heading = heading
        self._velocity = velocity
        self._angular_rate = angular_rate
        self._steering_angle = 0.0

        self._X = []
        self._Y = []
        self._Heading = []
        self._Velocity = []
        self._Angular_rate = []
        self._Time = []


    def set_motion(self, velocity, angular_rate):
        '''Set desired linear velocity and angular rate.
        Converts angular rate to ackermann steering angle:
            desired_steering = atan2(angular_rate * wheelbase, velocity)
        Then applies PD control on velocity and steering angle.'''
        # Velocity PD controller — use actual velocity (not noisy measurement)
        vel_error = velocity - self._velocity
        drive_force = 4.0 * vel_error + 0.5 * (vel_error - self._velocity_error)
        drive_force = max(-1.0, min(1.0, drive_force))
        self._velocity_error = vel_error

        # Convert desired angular_rate to desired steering angle
        if abs(velocity) > 0.01:
            desired_steering = math.atan2(angular_rate * self._wheelbase, velocity)
        else:
            desired_steering = 0.0
        desired_steering = max(-self._max_steering_angle,
                               min(self._max_steering_angle, desired_steering))

        # Steering PD controller — use actual steering angle (not noisy measurement)
        steer_error = desired_steering - self._steering_angle
        steering_rate = 3.0 * steer_error + 0.5 * (steer_error - self._steering_error)
        steering_rate = max(-1.0, min(1.0, steering_rate))
        self._steering_error = steer_error

        self._drive_force = drive_force
        self._steering_rate = steering_rate


    def get_measurements(self):
        '''Get measured values for all motion parameters with simulated sensor errors.'''
        gyro_noise = 1.0 * math.pi/180
        gnss_shift = 0.002
        gnss_noise = 0.002
        odo_scale = 0.08
        odo_noise = 0.03

        x = self._x + gnss_noise * (random.random() - 0.5)
        y = self._y + gnss_noise * (random.random() - 0.5)
        heading = self._heading + gyro_noise * (random.random() - 0.5)
        velocity = (self._velocity + odo_noise * (random.random() - 0.5)) * (1 + 2 * odo_scale * (random.random() - 0.5))
        angular_rate = (self._angular_rate + 0.15 * gyro_noise * (random.random() - 0.5)) * (1 + 2 * odo_scale * (random.random() - 0.5))
        steering_angle = (self._steering_angle + 0.05 * gyro_noise * (random.random() - 0.5)) * (1 + 2 * odo_scale * (random.random() - 0.5))

        return Parameters.Parameters(x, y, heading, velocity, angular_rate, steering_angle)


    def plot_results(self):
        gridsize = (3,2)
        fig = plt.figure(figsize=(13,9.3))
        ax1 = plt.subplot2grid(gridsize,(0,0), rowspan=2)
        ax2 = plt.subplot2grid(gridsize,(2,0))
        ax3 = plt.subplot2grid(gridsize,(0,1))
        ax4 = plt.subplot2grid(gridsize,(1,1))
        ax5 = plt.subplot2grid(gridsize,(2,1))
        # Plot trajectory
        ax1.set_title('Robot trajectory')
        ax1.plot(self._X, self._Y, 'b', linewidth=2)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        # Plot X and Y graphs
        ax2.plot(self._Time, self._X, 'b', linewidth=2, label='X')
        ax2.plot(self._Time, self._Y, 'r', linewidth=2, label='Y')
        ax2.grid()
        ax2.set_xlabel('Time, s')
        ax2.set_ylabel('Coordinates, m')
        ax2.legend()
        # Plot Heading graph
        ax3.plot(self._Time, self._Heading, 'g', linewidth=2)
        ax3.grid()
        ax3.set_ylabel('Heading angle, deg')
        ax3.yaxis.set_major_locator(MultipleLocator(90))
        # Plot Angular rate graph
        ax4.plot(self._Time, self._Angular_rate, 'b', linewidth=2)
        ax4.grid()
        ax4.set_ylabel('Angular rate, rad/s')
        # Plot Velocity graph
        ax5.plot(self._Time, self._Velocity, 'r', linewidth=2)
        ax5.grid()
        ax5.set_xlabel('Time, s')
        ax5.set_ylabel('Velocity, m/s')
        # Display all plots

        return (plt, ax1)
