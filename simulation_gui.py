import dearpygui.dearpygui as dpg
import threading
import time
import math
import os
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import sys

import Simulator
Sim = Simulator.Simulator
reload_sim = Simulator.reload_sim

# Global variables
path_series = None
robot_rect_series = None
heading_series = None
speed_series = None
front_laser_series = None
left_laser_series = None
right_laser_series = None
collision_warning = None
status_text = None
stop_simulation = False


def reload():
    reload_sim()
    print("[GUI] Control module reloaded")


def get_robot_corners(robot_x, robot_y, robot_heading):
    length = 0.8
    width = 0.5
    
    cos_h = math.cos(robot_heading)
    sin_h = math.sin(robot_heading)
    
    corners = []
    for dx, dy in [(length/2, width/2), (length/2, -width/2), 
                   (-length/2, -width/2), (-length/2, width/2)]:
        world_x = robot_x + dx * cos_h - dy * sin_h
        world_y = robot_y + dx * sin_h + dy * cos_h
        corners.append((world_x, world_y))
    
    rect_x = [c[0] for c in corners] + [corners[0][0]]
    rect_y = [c[1] for c in corners] + [corners[0][1]]
    
    return rect_x, rect_y


def parse_custom_walls(text):
    walls = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) >= 4:
            try:
                x1 = float(parts[0])
                y1 = float(parts[1])
                x2 = float(parts[2])
                y2 = float(parts[3])
                walls.append(((x1, x2), (y1, y2)))
            except ValueError:
                pass
    return walls


def run_simulation(sender, data):
    global stop_simulation, path_series, robot_rect_series, heading_series, speed_series, front_laser_series, left_laser_series, right_laser_series, collision_warning, status_text
    
    stop_simulation = False
    
    sim_time = dpg.get_value("sim_time_input")
    sim_dt = dpg.get_value("sim_time_dt")
    sim_dt = max(0.05, min(0.2, sim_dt))
    
    shelf_str = dpg.get_value("shelf_input")
    shelf_num = int(shelf_str.split(":")[0])
    
    print(f"\n{'='*60}")
    print(f"[GUI] Starting simulation to shelf #{shelf_num}")
    print(f"{'='*60}\n")
    
    custom_str = dpg.get_value("custom_walls_input")
    custom_walls = parse_custom_walls(custom_str) if custom_str.strip() else None
    if custom_walls:
        print(f"[GUI] Custom walls: {custom_walls}")
    
    simulator = Sim(sim_time=sim_time, trajectory=None, target_shelf=shelf_num, interactive=False, custom_walls=custom_walls)
    
    dpg.set_value(path_series, [[], []])
    dpg.set_value(robot_rect_series, [[], []])
    dpg.set_value(heading_series, [[], []])
    dpg.set_value(speed_series, [[], []])
    dpg.set_value(front_laser_series, [[], []])
    dpg.set_value(left_laser_series, [[], []])
    dpg.set_value(right_laser_series, [[], []])
    
    if dpg.does_item_exist("target_scatter"):
        dpg.set_value("target_scatter", [[simulator.final_target[0]], [simulator.final_target[1]]])
    
    for i, wall in enumerate(simulator.walls):
        wall_tag = f"wall_{i}"
        if not dpg.does_item_exist(wall_tag):
            x_coords = [wall[0][0], wall[0][1]]
            y_coords = [wall[1][0], wall[1][1]]
            dpg.add_line_series(x_coords, y_coords, parent="y_axis", tag=wall_tag)
    
    dpg.set_value(status_text, f"Simulation started... Shelf #{shelf_num}")
    
    sim_step = 0
    collision_detected = False
    last_log_time = 0
    
    skip_collision_steps = 10
    
    while not stop_simulation and not simulator.check_simulation_done():
        try:
            simulator.measure()
            simulator.calculate_control()
            
            if sim_step > skip_collision_steps:
                if simulator.check_collision():
                    collision_detected = True
                    print(f"\n⚠️ COLLISION at step {sim_step}!")
                    print(f"   Robot position: ({simulator.robot._x:.2f}, {simulator.robot._y:.2f})")
                    dpg.set_value(collision_warning, True)
                    dpg.configure_item("collision_text", show=True)
                    dpg.set_value(status_text, f"⚠️ COLLISION! Step {sim_step}")
                    break
            
            if not simulator.move():
                collision_detected = True
                break
            
            simulator.accumulate_data()
            
            # Обновляем графики реже (каждые 5 шагов)
            if sim_step % 5 == 0:
                dpg.set_value(path_series, [simulator.X_array, simulator.Y_array])
                
                rect_x, rect_y = get_robot_corners(
                    simulator.robot._x, 
                    simulator.robot._y, 
                    simulator.robot._heading
                )
                dpg.set_value(robot_rect_series, [rect_x, rect_y])
                
                if len(simulator.time_array) > 0:
                    dpg.set_value(heading_series, [simulator.time_array, simulator.heading_array])
                    dpg.set_value(speed_series, [simulator.time_array, simulator.speed_array])
                    valid = [(simulator.time_array[i], simulator.front_laser_array[i])
                             for i in range(len(simulator.front_laser_array))
                             if simulator.front_laser_array[i] is not None]
                    if valid:
                        ft, fd = zip(*valid)
                        dpg.set_value(front_laser_series, [list(ft), list(fd)])

                    valid_left = [(simulator.time_array[i], simulator.left_laser_array[i])
                                  for i in range(len(simulator.left_laser_array))
                                  if simulator.left_laser_array[i] is not None]
                    if valid_left:
                        lt, ld = zip(*valid_left)
                        dpg.set_value(left_laser_series, [list(lt), list(ld)])

                    valid_right = [(simulator.time_array[i], simulator.right_laser_array[i])
                                   for i in range(len(simulator.right_laser_array))
                                   if simulator.right_laser_array[i] is not None]
                    if valid_right:
                        rt, rd = zip(*valid_right)
                        dpg.set_value(right_laser_series, [list(rt), list(rd)])
            
            # Логирование реже (каждые 10 секунд симуляции)
            if simulator.time - last_log_time > 10:
                nav_status = simulator.get_navigation_status()
                print(f"[GUI] Time: {simulator.time:.1f}s, Stage: {nav_status['stage']}")
                last_log_time = simulator.time
                dpg.set_value(status_text, f"Simulation: {simulator.time:.1f}s")
            
            if simulator.check_target():
                print(f"\n✅ Shelf #{shelf_num} REACHED!")
                dpg.set_value(status_text, f"✅ Shelf #{shelf_num} reached in {simulator.time:.1f}s")
                break
            
            sim_step += 1
            
            # ===== ГЛАВНОЕ: УБРАТЬ ЗАДЕРЖКУ =====
            # time.sleep(sim_dt)  # ПОЛНОСТЬЮ УБРАТЬ!
            # или очень маленькая задержка:
            # time.sleep(0.0001)
            
        except Exception as e:
            print(f"[ERROR] Simulation error: {e}")
            import traceback
            traceback.print_exc()
            dpg.set_value(status_text, f"Error: {e}")
            break
    
    if not collision_detected and simulator.check_target():
        print(f"\n✅ SIMULATION COMPLETED SUCCESSFULLY! Time: {simulator.time:.1f}s\n")
    elif collision_detected:
        print(f"\n⚠️ SIMULATION STOPPED DUE TO COLLISION\n")
    else:
        print(f"\n⚠️ SIMULATION TIME EXPIRED\n")
    
    dpg.set_value(status_text, "Simulation finished")

    if len(simulator.front_laser_array) > 0:
        import json
        data_path = os.path.join(tempfile.gettempdir(), f"_front_laser_{os.getpid()}.json")
        with open(data_path, 'w') as f:
            json.dump([
                list(simulator.time_array),
                [d if d is not None else None for d in simulator.front_laser_array]
            ], f)
        script = (
            'import sys, json, matplotlib.pyplot as plt\n'
            'with open(sys.argv[1], "r") as f: t, d = json.load(f)\n'
            'd = [x if x is not None else float("nan") for x in d]\n'
            'plt.figure(figsize=(12, 5))\n'
            'plt.plot(t, d, "m-", linewidth=2)\n'
            'plt.xlabel("Time, s")\n'
            'plt.ylabel("Distance, m")\n'
            'plt.title("Front Laser Rangefinder")\n'
            'plt.grid(True, alpha=0.3)\n'
            'plt.tight_layout()\n'
            'plt.show()\n'
        )
        subprocess.Popen([sys.executable, '-c', script, data_path])

    if len(simulator.left_laser_array) > 0:
        import json
        data_path = os.path.join(tempfile.gettempdir(), f"_left_laser_{os.getpid()}.json")
        with open(data_path, 'w') as f:
            json.dump([
                list(simulator.time_array),
                [d if d is not None else None for d in simulator.left_laser_array]
            ], f)
        script = (
            'import sys, json, matplotlib.pyplot as plt\n'
            'with open(sys.argv[1], "r") as f: t, d = json.load(f)\n'
            'd = [x if x is not None else float("nan") for x in d]\n'
            'plt.figure(figsize=(12, 5))\n'
            'plt.plot(t, d, "c-", linewidth=2)\n'
            'plt.xlabel("Time, s")\n'
            'plt.ylabel("Distance, m")\n'
            'plt.title("Left Laser Rangefinder")\n'
            'plt.grid(True, alpha=0.3)\n'
            'plt.tight_layout()\n'
            'plt.show()\n'
        )
        subprocess.Popen([sys.executable, '-c', script, data_path])

    if len(simulator.right_laser_array) > 0:
        import json
        data_path = os.path.join(tempfile.gettempdir(), f"_right_laser_{os.getpid()}.json")
        with open(data_path, 'w') as f:
            json.dump([
                list(simulator.time_array),
                [d if d is not None else None for d in simulator.right_laser_array]
            ], f)
        script = (
            'import sys, json, matplotlib.pyplot as plt\n'
            'with open(sys.argv[1], "r") as f: t, d = json.load(f)\n'
            'd = [x if x is not None else float("nan") for x in d]\n'
            'plt.figure(figsize=(12, 5))\n'
            'plt.plot(t, d, "y-", linewidth=2)\n'
            'plt.xlabel("Time, s")\n'
            'plt.ylabel("Distance, m")\n'
            'plt.title("Right Laser Rangefinder")\n'
            'plt.grid(True, alpha=0.3)\n'
            'plt.tight_layout()\n'
            'plt.show()\n'
        )
        subprocess.Popen([sys.executable, '-c', script, data_path])


def start_simulation(sender, data):
    global stop_simulation
    stop_simulation = False
    
    dpg.set_value(collision_warning, False)
    dpg.configure_item("collision_text", show=False)
    
    simulation_thread = threading.Thread(target=run_simulation, args=(sender, data))
    simulation_thread.daemon = True
    simulation_thread.start()


def stop_simulation_callback(sender, data):
    global stop_simulation
    stop_simulation = True
    print("[GUI] Simulation stopped by user")
    dpg.set_value(status_text, "Simulation stopped")


def create_main_window():
    global path_series, robot_rect_series, heading_series, speed_series, front_laser_series, left_laser_series, right_laser_series
    
    with dpg.window(label="Map and Robot Trajectory", pos=(250, 0), width=600, height=600):
        with dpg.plot(label="Robot Trajectory", height=550, width=550):
            dpg.add_plot_legend()
            
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="X (meters)")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y (meters)", tag="y_axis")
            
            dpg.set_axis_limits(x_axis, 0, 30)
            dpg.set_axis_limits(y_axis, 0, 30)
            
            path_series = dpg.add_line_series([], [], label="Robot trajectory", parent=y_axis)
            robot_rect_series = dpg.add_line_series([], [], label="Robot", parent=y_axis)
            dpg.add_scatter_series([0], [0], label="Target", parent=y_axis, tag="target_scatter")
    
    with dpg.window(label="Heading Control", pos=(850, 0), width=400, height=300):
        with dpg.plot(label="Robot Heading", height=250, width=350):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time (seconds)")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Heading (radians)")
            heading_series = dpg.add_line_series([], [], label="Actual heading", parent=y_axis)
            dpg.set_axis_limits(x_axis, 0, 100)
            dpg.set_axis_limits(y_axis, -3.2, 3.2)
    
    with dpg.window(label="Velocity Control", pos=(850, 310), width=400, height=250):
        with dpg.plot(label="Linear Velocity", height=200, width=350):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time (seconds)")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Velocity (m/s)")
            speed_series = dpg.add_line_series([], [], label="Actual velocity", parent=y_axis)
            dpg.set_axis_limits(x_axis, 0, 100)
            dpg.set_axis_limits(y_axis, 0, 0.6)

    with dpg.window(label="Front Laser", pos=(0, 455), width=250, height=195):
        with dpg.plot(label="Front Laser Distance", height=145, width=220):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Distance (m)")
            front_laser_series = dpg.add_line_series([], [], label="Front laser", parent=y_axis)
            dpg.set_axis_limits(x_axis, 0, 100)
            dpg.set_axis_limits(y_axis, 0, 10)

    with dpg.window(label="Left Laser", pos=(250, 455), width=250, height=195):
        with dpg.plot(label="Left Laser Distance", height=145, width=220):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Distance (m)")
            left_laser_series = dpg.add_line_series([], [], label="Left laser", parent=y_axis)
            dpg.set_axis_limits(x_axis, 0, 100)
            dpg.set_axis_limits(y_axis, 0, 10)

    with dpg.window(label="Right Laser", pos=(500, 455), width=250, height=195):
        with dpg.plot(label="Right Laser Distance", height=145, width=220):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Distance (m)")
            right_laser_series = dpg.add_line_series([], [], label="Right laser", parent=y_axis)
            dpg.set_axis_limits(x_axis, 0, 100)
            dpg.set_axis_limits(y_axis, 0, 10)


def create_control_panel():
    global collision_warning, status_text
    
    with dpg.window(label="Simulation Control", width=250, height=450, pos=(0, 0)):
        dpg.add_text("ROKO2 Parameters")
        dpg.add_separator()
        
        dpg.add_input_float(tag="sim_time_input", label="Max time (s)", 
                           default_value=10000.0, min_value=10, max_value=50000)
        dpg.add_input_float(tag="sim_time_dt", label="Time step (s)", 
                           default_value=0.05, min_value=0.01, max_value=0.1)
        
        dpg.add_separator()
        dpg.add_text("Select shelf:")
        
        shelves = [
            "1: (5.4, 6.0) - row 1",
            "2: (9.4, 6.0) - row 1",
            "3: (13.4, 6.0) - row 1",
            "4: (17.4, 6.0) - row 1",
            "5: (21.4, 6.0) - row 1",
            "6: (25.4, 6.0) - row 1",
            "7: (5.4, 12.0) - row 2",
            "8: (9.4, 12.0) - row 2",
            "9: (13.4, 12.0) - row 2",
            "10: (17.4, 12.0) - row 2",
            "11: (21.4, 12.0) - row 2",
            "12: (25.4, 12.0) - row 2",
            "13: (5.4, 18.0) - row 3",
            "14: (9.4, 18.0) - row 3",
            "15: (13.4, 18.0) - row 3",
            "16: (17.4, 18.0) - row 3",
            "17: (21.4, 18.0) - row 3",
            "18: (25.4, 18.0) - row 3",
            "19: (5.4, 24.0) - row 4",
            "20: (9.4, 24.0) - row 4",
            "21: (13.4, 24.0) - row 4",
            "22: (17.4, 24.0) - row 4",
            "23: (21.4, 24.0) - row 4",
            "24: (25.4, 24.0) - row 4",
        ]
        
        dpg.add_combo(tag="shelf_input", items=shelves, 
                     label="Shelf", default_value=shelves[0], width=200)
        
        dpg.add_separator()
        dpg.add_button(label="▶ Start", callback=start_simulation, width=200, height=30)
        dpg.add_button(label="⏹ Stop", callback=stop_simulation_callback, width=200, height=30)
        dpg.add_button(label="⟳ Reload Control", callback=reload, width=200, height=30)
        
        dpg.add_separator()
        
        with dpg.group(horizontal=True):
            collision_warning = dpg.add_checkbox(label="Collision!", default_value=False, enabled=False)
            dpg.add_text("(wall collision detected)", color=[255, 0, 0, 255], tag="collision_text", show=False)
        
        dpg.add_separator()
        status_text = dpg.add_text("Ready to start", color=[100, 200, 100, 255])
        
        dpg.add_separator()
        dpg.add_text("Custom walls (x1,y1,x2,y2):", bullet=True)
        dpg.add_input_text(tag="custom_walls_input", label="one segment per line", default_value="", width=200, height=60, multiline=True)
        
        dpg.add_separator()
        dpg.add_text("Navigation mode:", bullet=True)
        dpg.add_text("• Right/Left hand rule", bullet=True)
        dpg.add_text("• Ackermann steering", bullet=True)
        dpg.add_text("• Dead reckoning (no GPS)", bullet=True)


def main():
    dpg.create_context()
    
    create_main_window()
    create_control_panel()
    
    dpg.create_viewport(title='ROKO2 Robot Simulator - FAST MODE', 
                        width=1300, height=650)
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()