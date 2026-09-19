import math
import matplotlib.pyplot as plt
import importlib
from maze_generator import generate_maze, maze_to_walls, sense_wall, get_shelves, add_custom_walls
import exercise
import Roko2

calculate_control = exercise.calculate_control
reset_navigation = exercise.reset_navigation
start_navigation = exercise.start_navigation
get_navigation_status = exercise.get_navigation_status
is_navigation_complete = exercise.is_navigation_complete


def reload_sim():
    global calculate_control, reset_navigation, start_navigation, get_navigation_status, is_navigation_complete
    importlib.reload(exercise)
    calculate_control = exercise.calculate_control
    reset_navigation = exercise.reset_navigation
    start_navigation = exercise.start_navigation
    get_navigation_status = exercise.get_navigation_status
    is_navigation_complete = exercise.is_navigation_complete


def select_shelf_interactive():
    shelves = get_shelves()
    print("\n" + "=" * 50)
    print("AVAILABLE SHELVES")
    print("=" * 50)
    for num, x, y in shelves[:24]:
        print(f"  Shelf {num:2d}: coordinates ({x:4.1f}, {y:4.1f})")
    print("=" * 50)
    while True:
        try:
            target_shelf = int(input("\nEnter shelf number (1-24): "))
            shelf = next((s for s in shelves if s[0] == target_shelf), None)
            if shelf:
                return target_shelf
            else:
                print(f"Shelf #{target_shelf} not found")
        except ValueError:
            print("Enter a number")


class Simulator:
    def __init__(self, sim_time, trajectory=None, width=5, height=5,
                 target_shelf=None, interactive=True, custom_walls=None):
        maze = generate_maze(width, height)
        self.walls = maze_to_walls(maze)
        self.maze_walls = self.walls[:]
        self.walls = add_custom_walls(self.walls, custom_walls)
        self.maze = maze
        self.sim_time = sim_time
        self.time = 0
        
        self.time_delta = 0.2
        
        start_x, start_y = 2.5, 2.5
        self.start_x = start_x
        self.start_y = start_y

        self.robot = Roko2.Roko2(start_x, start_y, 0, 0, 0)
        self.params = self.robot.get_measurements()
        self.distances = (0, 0, 0)

        self.velocity_control = 0.0
        self.heading_control = 0.0

        self.X_array = []
        self.Y_array = []
        self.time_array = []
        self.speed_array = []
        self.heading_array = []
        self.front_laser_array = []
        self.left_laser_array = []
        self.right_laser_array = []

        shelves = get_shelves()
        if interactive and target_shelf is None:
            target_shelf = select_shelf_interactive()
        elif target_shelf is None:
            target_shelf = 1

        shelf = next((s for s in shelves if s[0] == target_shelf), shelves[0])
        self.target_shelf_num = target_shelf
        self.final_target = [shelf[1], shelf[2]]

        print(f"\n{'='*50}")
        print(f"SELECTED SHELF #{target_shelf}")
        print(f"Target coordinates: ({self.final_target[0]:.1f}, {self.final_target[1]:.1f})")
        print(f"Start position: ({start_x:.1f}, {start_y:.1f})")
        print(f"Simulation step: {self.time_delta}s (faster simulation)")
        print(f"{'='*50}\n")

        reset_navigation()
        start_navigation(target_shelf, self.final_target[0], self.final_target[1],
                         start_x, start_y, 0.0)

    def _get_wheel_odometry(self):
        delta_distance = self.params.velocity * self.time_delta
        delta_angle = self.params.angular_rate * self.time_delta
        return delta_distance, delta_angle

    def measure(self):
        self.params = self.robot.get_measurements()
        h = self.robot._heading
        pos = (self.robot._x, self.robot._y)
        self.distances = (
            sense_wall(self.walls, pos, h + math.pi / 2),
            sense_wall(self.walls, pos, h),
            sense_wall(self.walls, pos, h - math.pi / 2),
        )
        self.expected_distances = (
            sense_wall(self.maze_walls, pos, h + math.pi / 2),
            sense_wall(self.maze_walls, pos, h),
            sense_wall(self.maze_walls, pos, h - math.pi / 2),
        )
        self.unknown_wall_front = False
        actual = self.distances[1]
        expected = self.expected_distances[1]
        if actual is not None and expected is not None:
            if actual < 1.9 and actual < expected - 0.5:
                self.unknown_wall_front = True

    def calculate_control(self):
        delta_distance, delta_angle = self._get_wheel_odometry()
        
        self.velocity_control, self.heading_control = calculate_control(
            self.params.heading,
            self.distances,
            self.robot._x,
            self.robot._y,
            delta_distance,
            delta_angle,
            self.unknown_wall_front
        )

    def check_target(self):
        return is_navigation_complete()
    
    def get_navigation_status(self):
        return get_navigation_status()

    def get_robot_corners(self):
        x, y = self.robot._x, self.robot._y
        heading = self.robot._heading
        
        length = self.robot._lx
        width = self.robot._ly
        
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        
        corners = []
        for dx, dy in [(length/2, width/2), (length/2, -width/2), 
                       (-length/2, -width/2), (-length/2, width/2)]:
            world_x = x + dx * cos_h - dy * sin_h
            world_y = y + dx * sin_h + dy * cos_h
            corners.append((world_x, world_y))
        
        return corners

    def predict_collision(self, forward_distance=0.1):
        from maze_generator import segment_intersect_walls
        
        corners = self.get_robot_corners()
        heading = self.robot._heading
        
        dx = math.cos(heading) * forward_distance
        dy = math.sin(heading) * forward_distance
        
        projected_corners = []
        for cx, cy in corners:
            projected_corners.append((cx + dx, cy + dy))
        
        for i in range(4):
            x1, y1 = projected_corners[i]
            x2, y2 = projected_corners[(i+1) % 4]
            
            if segment_intersect_walls(self.walls, x1, y1, x2, y2):
                return True
        
        return False

    def check_collision(self):
        from maze_generator import segment_intersect_walls
        
        corners = self.get_robot_corners()
        
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i+1) % 4]
            
            if segment_intersect_walls(self.walls, x1, y1, x2, y2):
                return True
        
        return False

    def move(self):
        robot_dt = self.robot._dt
        n_substeps = max(1, round(self.time_delta / robot_dt))
        
        self.robot.set_motion(self.velocity_control, self.heading_control)
        for _ in range(n_substeps):
            self.robot.update()
            self.time += robot_dt
        
        if self.check_collision():
            print(f"\n[WARN] COLLISION DETECTED!")
            return False
        
        return True

    def accumulate_data(self):
        self.X_array.append(self.robot._x)
        self.Y_array.append(self.robot._y)
        self.time_array.append(self.time)
        self.speed_array.append(self.params.velocity)
        self.heading_array.append(self.params.heading)
        self.front_laser_array.append(self.distances[1])
        self.left_laser_array.append(self.distances[0])
        self.right_laser_array.append(self.distances[2])

    def check_simulation_done(self):
        return self.time >= self.sim_time or is_navigation_complete()

    def plot_main_data(self):
        plt_obj, ax = self.robot.plot_results()
        for wall in self.walls:
            ax.plot(wall[0], wall[1], 'k-', lw=2)
        ax.plot(self.final_target[0], self.final_target[1],
                'r*', markersize=18, label=f'Shelf #{self.target_shelf_num}')
        ax.plot(self.start_x, self.start_y, 'go', markersize=10, label='Start')
        
        corners = self.get_robot_corners()
        rect_x = [c[0] for c in corners] + [corners[0][0]]
        rect_y = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(rect_x, rect_y, 'r-', linewidth=2, alpha=0.5)
        
        ax.set_title(f'Robot Trajectory to Shelf #{self.target_shelf_num}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 29)
        ax.set_ylim(0, 29)
        plt_obj.legend(loc='upper right')
        plt_obj.tight_layout()

        fig2, ax6 = plt.subplots(figsize=(13, 3))
        dist_plot = [d if d is not None else float('nan') for d in self.front_laser_array]
        ax6.plot(self.time_array, dist_plot, 'm-', linewidth=2)
        ax6.set_xlabel('Time, s')
        ax6.set_ylabel('Distance, m')
        ax6.set_title('Front Laser Rangefinder')
        ax6.grid(True, alpha=0.3)
        fig2.tight_layout()

        fig3, ax7 = plt.subplots(figsize=(13, 3))
        left_dist = [d if d is not None else float('nan') for d in self.left_laser_array]
        ax7.plot(self.time_array, left_dist, 'c-', linewidth=2)
        ax7.set_xlabel('Time, s')
        ax7.set_ylabel('Distance, m')
        ax7.set_title('Left Laser Rangefinder')
        ax7.grid(True, alpha=0.3)
        fig3.tight_layout()

        fig4, ax8 = plt.subplots(figsize=(13, 3))
        right_dist = [d if d is not None else float('nan') for d in self.right_laser_array]
        ax8.plot(self.time_array, right_dist, 'y-', linewidth=2)
        ax8.set_xlabel('Time, s')
        ax8.set_ylabel('Distance, m')
        ax8.set_title('Right Laser Rangefinder')
        ax8.grid(True, alpha=0.3)
        fig4.tight_layout()

        plt_obj.show()


def check_navigation_complete():
    return is_navigation_complete()