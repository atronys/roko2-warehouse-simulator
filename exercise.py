import math

WALL_FOLLOW_OFFSET = 0.5

ROBOT_LENGTH = 0.8
ROBOT_WIDTH = 0.5
HALF_LENGTH = ROBOT_LENGTH / 2
HALF_WIDTH = ROBOT_WIDTH / 2

DESIRED_WALL_DISTANCE = 1.3
FRONT_DANGER_ZONE = 1.3
WALL_DANGER_ZONE = 1.3

MIN_SIDE_DISTANCE = 0.7
STUCK_COUNTER_LIMIT = 60
WALL_CENTER_DIST = 1.0
DEAD_BAND = 0.4
STUCK_PROGRESS_THRESHOLD = 0.02

# Reverse-turn constants
REVERSE_VEL = 0.08
REVERSE_PHASE_TIMEOUT = 80
REVERSE_MIN_FORWARD_DIST = 0.70
REVERSE_TURN_VEL = 0.10


class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0.0
        self.previous_error = 0.0
        self.dt = 0.1
        self.prev_deriv = 0.0
        self.integral_limit = 0.3
        self.error_limit = 0.3

    def update(self, error, dt=0.1):
        self.dt = dt
        error = max(-self.error_limit, min(self.error_limit, error))
        proportional = self.kp * error
        self.integral += error * self.dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        integral = self.ki * self.integral
        derivative_raw = (error - self.previous_error) / self.dt
        derivative = 0.7 * derivative_raw + 0.3 * self.prev_deriv
        self.prev_deriv = derivative
        derivative = self.kd * derivative
        self.previous_error = error
        output = proportional + integral + derivative
        return max(self.output_limits[0], min(self.output_limits[1], output))

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.prev_deriv = 0.0


def get_passage_y(target_y):
    if target_y <= 8:
        return 9.0
    elif target_y <= 14:
        return 15.0
    elif target_y <= 20:
        return 21.0
    else:
        return 27.0


class NavigationState:
    def __init__(self):
        self.stage = 0
        self.attempt = 0
        self.target_x = 0.0
        self.target_y = 0.0
        self.passage_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.progress = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.heading_at_start = 0.0
        self.turned = False
        self.shelf_num = 0
        self.turn_phase = 0
        self.stuck_counter = 0
        self.max_progress = 0.0
        self.last_front_blocked = False
        self.facing_back = False

        # Reverse-turn state
        self.reverse_forward_dist = 0.0
        self.reverse_odometry = 0.0
        self.reverse_phase_ticks = 0
        self.return_turn_done = False
        self.between_shelves = False


_nav = NavigationState()

_pid_heading = PID(kp=1.5, ki=0.01, kd=0.1, output_limits=(-0.5, 0.5))
_pid_wall = PID(kp=0.1, ki=0.002, kd=0.25, output_limits=(-0.25, 0.25))

_last_vel = 0.0
_last_ang = 0.0

MAX_VEL = 0.3
MAX_ANG = 0.35
SMOOTH = 0.45


def reset_navigation():
    global _nav, _pid_heading, _pid_wall, _last_vel, _last_ang
    _nav = NavigationState()
    _pid_heading.reset()
    _pid_wall.reset()
    _last_vel = 0.0
    _last_ang = 0.0


def start_navigation(shelf_num, shelf_x, shelf_y, start_x, start_y, heading):
    global _nav
    _nav.shelf_num = shelf_num
    _nav.target_x = shelf_x
    _nav.target_y = shelf_y
    _nav.passage_y = get_passage_y(shelf_y)
    _nav.start_x = start_x
    _nav.start_y = start_y
    _nav.stage = 0
    _nav.attempt = 0
    _nav.progress = 0.0
    _nav.last_x = start_x
    _nav.last_y = start_y
    _nav.heading_at_start = heading
    _nav.turned = False
    _nav.turn_phase = 0
    _nav.stuck_counter = 0
    _nav.max_progress = 0.0
    _nav.last_front_blocked = False
    _nav.facing_back = False
    _nav.return_turn_done = False
    print(f"[NAV] Shelf {shelf_num} at ({shelf_x:.1f}, {shelf_y:.1f}) | Attempt 0: EAST then NORTH")


def get_navigation_status():
    return {
        'stage': _nav.stage,
        'attempt': _nav.attempt,
        'progress': _nav.progress,
        'target_x': _nav.target_x,
        'target_y': _nav.target_y,
        'shelf_num': _nav.shelf_num,
        'passage_y': _nav.passage_y,
    }


def is_navigation_complete():
    if _nav.attempt == 0:
        return _nav.stage == 3
    else:
        return _nav.stage >= 5


def calc_turn_control(current_heading, target_heading, target_vel=0.12, target_ang=0.35):
    angle_error = target_heading - current_heading
    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
    if abs(angle_error) < 0.15:
        return True, 0.0, 0.0
    return False, target_vel, target_ang


def calculate_control(current_heading, distances, robot_x=0, robot_y=0, delta_distance=0.0, delta_angle=0.0, unknown_wall_front=False):
    global _last_vel, _last_ang

    if robot_x != 0 or robot_y != 0:
        dx = robot_x - _nav.last_x
        dy = robot_y - _nav.last_y
        _nav.last_x = robot_x
        _nav.last_y = robot_y

        if _nav.stage == 0:
            if _nav.attempt == 0:
                if abs(dx) > 0.001:
                    _nav.progress += abs(dx) if dx > 0 else 0
            else:
                if abs(dy) > 0.001:
                    _nav.progress += abs(dy) if dy > 0 else 0
        elif _nav.stage == 2:
            if _nav.attempt == 0:
                if abs(dy) > 0.001:
                    _nav.progress += abs(dy) if dy > 0 else 0
            else:
                if abs(dx) > 0.001:
                    _nav.progress += abs(dx) if dx > 0 else 0
        elif _nav.stage == 4:
            go_north = _nav.target_y > _nav.passage_y
            if abs(dy) > 0.001:
                _nav.progress += abs(dy) if (dy > 0) == go_north else 0
        elif _nav.stage == 11:
            _nav.progress = abs(robot_y - _nav.start_y)
        elif _nav.stage == 12:
            _nav.progress = robot_x - _nav.start_x

    dist_left, dist_front, dist_right = distances
    dist_left = dist_left if dist_left is not None else 99.0
    dist_right = dist_right if dist_right is not None else 99.0
    dist_front = dist_front if dist_front is not None else 99.0

    target_vel = 0.0
    target_ang = 0.0

    front_blocked = dist_front < FRONT_DANGER_ZONE

    # ============= Unknown wall detection (stage 0, attempt 0) =============
    if _nav.stage == 0 and _nav.attempt == 0:
        if unknown_wall_front:
            print(f"\n[UNKNOWN WALL] Unexpected wall detected in front! Returning to base")
            print(f"[UNKNOWN WALL] Will try attempt 1: NORTH then EAST via passage y={_nav.passage_y}")
            _nav.stage = 10
            _nav.progress = 0.0
            _nav.turned = False
            _nav.turn_phase = 0
            _nav.stuck_counter = 0
            _nav.facing_back = False
            _nav.return_turn_done = False
            _nav.reverse_odometry = 0.0
            _nav.reverse_phase_ticks = 0
            _nav.heading_at_start = current_heading
            _pid_wall.reset()
            _pid_heading.reset()
            target_vel = 0.0
            target_ang = 0.0

    # ============= Unknown wall detection (stage 2, attempt 0) =============
    if _nav.stage == 2 and _nav.attempt == 0:
        if unknown_wall_front:
            _nav.between_shelves = (dist_left < 2.5 and dist_right < 2.5) or _nav.progress > 5.0
            if _nav.between_shelves:
                print(f"\n[UNKNOWN WALL] Wall ahead + shelves on BOTH sides (L={dist_left:.1f} R={dist_right:.1f})")
                print(f"[UNKNOWN WALL] Reverse-turn maneuver")
            else:
                print(f"\n[UNKNOWN WALL] Wall ahead, but open sides (L={dist_left:.1f} R={dist_right:.1f})")
                print(f"[UNKNOWN WALL] Simple turn 180")
            _nav.stage = 10
            _nav.progress = 0.0
            _nav.turned = False
            _nav.turn_phase = 0
            _nav.stuck_counter = 0
            _nav.facing_back = False
            _nav.return_turn_done = False
            _nav.reverse_odometry = 0.0
            _nav.reverse_phase_ticks = 0
            _nav.heading_at_start = current_heading
            _pid_wall.reset()
            _pid_heading.reset()
            target_vel = 0.0
            target_ang = 0.0

    # ===================== STAGE 0: Go along =====================
    if _nav.stage == 0:
        if _nav.attempt == 0:
            if front_blocked:
                if _last_vel > 0:
                    _last_vel = 0.0
                target_vel = -0.06
                target_ang = 0.2
                _pid_wall.reset()
            else:
                wall_dist = dist_right
                error = DESIRED_WALL_DISTANCE - wall_dist
                error = max(-0.15, min(0.15, error))
                target_ang = _pid_wall.update(error, 0.1)
                target_vel = MAX_VEL
                if abs(error) > 0.1:
                    target_vel = MAX_VEL * 0.7
                target_ang = max(-0.2, min(0.2, target_ang))

            target_needed = abs(_nav.target_x + WALL_FOLLOW_OFFSET - _nav.start_x)
            if int(_nav.progress * 20) % 20 == 0 and _nav.progress > 0:
                print(f"[EAST] X: {_nav.progress:.2f}/{target_needed:.2f}m")

            if _nav.progress >= target_needed - 0.3:
                print(f"\n[X+] X target reached, turning NORTH")
                _nav.stage = 1
                _nav.progress = 0.0
                _nav.heading_at_start = current_heading
                _nav.turned = False
                _nav.turn_phase = 0
                _pid_wall.reset()
                target_vel = 0.0
                target_ang = 0.0
        else:
            target_heading = math.pi / 2
            heading_error = target_heading - current_heading
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

            if abs(heading_error) > 0.15:
                target_ang = max(-0.2, min(0.2, heading_error * 0.8))
                target_vel = 0.1
                _pid_wall.reset()
            else:
                if front_blocked:
                    if _last_vel > 0:
                        _last_vel = 0.0
                    target_vel = -0.06
                    target_ang = -0.2
                    _pid_wall.reset()
                else:
                    wall_dist = dist_left
                    error = wall_dist - DESIRED_WALL_DISTANCE
                    error = max(-0.15, min(0.15, error))
                    target_ang = _pid_wall.update(error, 0.1) + heading_error * 0.3
                    target_vel = MAX_VEL
                    if abs(error) > 0.1:
                        target_vel = MAX_VEL * 0.7
                    target_ang = max(-0.2, min(0.2, target_ang))

            target_needed = abs(_nav.passage_y - _nav.start_y)
            if int(_nav.progress * 10) % 10 == 0 and _nav.progress > 0:
                print(f"[NORTH-alt] Y: {_nav.progress:.2f}/{target_needed:.2f}m")

            if _nav.progress >= target_needed - 0.3:
                print(f"\n[Y+] Passage y={_nav.passage_y:.0f} reached, turning EAST")
                _nav.stage = 1
                _nav.progress = 0.0
                _nav.heading_at_start = current_heading
                _nav.turned = False
                _nav.turn_phase = 0
                _pid_wall.reset()
                _pid_heading.reset()
                target_vel = 0.0
                target_ang = 0.0

    # ===================== STAGE 1: Turn =====================
    elif _nav.stage == 1:
        if not _nav.turned:
            if _nav.turn_phase == 0:
                if _nav.attempt == 0:
                    print(f"[TURN] LEFT to NORTH")
                    turn_target = math.pi / 2
                else:
                    print(f"[TURN] RIGHT to EAST")
                    turn_target = 0.0
                _nav.turn_phase = 1

            if _nav.attempt == 0:
                target_vel = 0.12
                target_ang = +0.35
                turn_target = math.pi / 2
            else:
                target_vel = 0.12
                target_ang = -0.35
                turn_target = 0.0
            angle_error = turn_target - current_heading
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

            if abs(angle_error) < 0.15:
                dir_name = "NORTH" if _nav.attempt == 0 else "EAST"
                print(f"[TURN] Now facing {dir_name}")
                _nav.turned = True
                _nav.heading_at_start = current_heading
                _nav.stage = 2
                _nav.progress = 0.0
                _nav.turn_phase = 0
                _pid_heading.reset()
                _pid_wall.reset()
                target_vel = 0.0
                target_ang = 0.0

    # ===================== STAGE 2: Go to target =====================
    elif _nav.stage == 2:
        target_ang = 0.0
        target_heading = math.pi / 2 if _nav.attempt == 0 else 0.0
        heading_error = target_heading - current_heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        if dist_left < WALL_CENTER_DIST + DEAD_BAND:
            if _nav.attempt == 1:
                error = dist_left - 2.0
            else:
                error = dist_left - WALL_CENTER_DIST
            target_ang = _pid_wall.update(error, 0.1) + heading_error * 0.3
        elif dist_right < WALL_CENTER_DIST + DEAD_BAND:
            if _nav.attempt == 1:
                error = 1.0 - dist_right
            else:
                error = WALL_CENTER_DIST - dist_right
            target_ang = _pid_wall.update(error, 0.1) + heading_error * 0.3
        else:
            target_ang = _pid_heading.update(heading_error, 0.1)
        target_ang = max(-0.12, min(0.12, target_ang))

        if _nav.attempt == 0:
            target_needed = abs(_nav.target_y - _nav.start_y)
        else:
            target_needed = abs(_nav.target_x + WALL_FOLLOW_OFFSET - _nav.start_x)

        remaining = target_needed - _nav.progress
        if remaining > 0.8:
            target_vel = MAX_VEL
        elif remaining > 0.3:
            target_vel = 0.18
        else:
            target_vel = 0.1

        axis = "Y" if _nav.attempt == 0 else "X"
        if int(_nav.progress * 10) % 10 == 0 and _nav.progress > 0:
            print(f"[{axis}] {_nav.progress:.2f}/{target_needed:.2f}m, rem={remaining:.2f}")

        if remaining < 0.2:
            if _nav.attempt == 0:
                print(f"\n[SHELF #{_nav.shelf_num}] REACHED!")
                _nav.stage = 3
            else:
                print(f"\n[X+] X reached at passage, adjusting Y to target")
                _nav.stage = 4
                _nav.progress = 0.0
                _nav.heading_at_start = current_heading
                _pid_heading.reset()
                _pid_wall.reset()
            target_vel = 0.0
            target_ang = 0.0

    # ===================== STAGE 4: Y adjustment (attempt 1) =====================
    elif _nav.stage == 4:
        dx = _nav.target_x - robot_x
        dy = _nav.target_y - robot_y
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - current_heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        if abs(heading_error) > 0.15:
            target_ang = max(-0.2, min(0.2, heading_error * 0.8))
            target_vel = 0.2
        else:
            target_ang = _pid_heading.update(heading_error, 0.1)
            target_ang = max(-0.08, min(0.08, target_ang))
            target_vel = MAX_VEL

        y_remaining = abs(robot_y - _nav.target_y)
        if int(_nav.progress * 10) % 10 == 0 and _nav.progress > 0:
            print(f"[Y-adj] {_nav.progress:.2f} / {y_remaining:.2f}m left")

        if y_remaining < 0.5:
            print(f"\n[SHELF #{_nav.shelf_num}] REACHED via alt path!")
            _nav.stage = 5
            target_vel = 0.0
            target_ang = 0.0

    # ===================== STAGE 5: Done (attempt 1) =====================
    elif _nav.stage == 5:
        target_vel = 0.0
        target_ang = 0.0

    # ===================== STAGE 10: Turn 180 for return =====================
    elif _nav.stage == 10:
        if _nav.between_shelves:
            # === REVERSE-TURN: between shelves, cannot spin in place ===
            if _nav.turn_phase == 0:
                # Phase 0: go forward, measure distance by odometry.
                if _nav.reverse_phase_ticks == 0:
                    _nav.reverse_odometry = 0.0
                    print("[REVERSE] Phase 0: going forward...")
                _nav.reverse_phase_ticks += 1
                _nav.reverse_odometry += abs(delta_distance)

                if _nav.reverse_odometry >= REVERSE_MIN_FORWARD_DIST or _nav.reverse_phase_ticks > REVERSE_PHASE_TIMEOUT:
                    _nav.turn_phase = 1
                    _nav.reverse_forward_dist = _nav.reverse_odometry
                    _nav.reverse_odometry = 0.0
                    _nav.reverse_phase_ticks = 0
                    print(f"[REVERSE] Phase 1: backward. Forward dist = {_nav.reverse_forward_dist:.3f}m")
                    target_vel = 0.0
                    target_ang = 0.0
                else:
                    target_vel = REVERSE_VEL
                    target_ang = 0.0

            elif _nav.turn_phase == 1:
                # Phase 1: backward by the same odometry distance.
                _nav.reverse_phase_ticks += 1
                if delta_distance < 0:
                    _nav.reverse_odometry += abs(delta_distance)

                if _nav.reverse_odometry >= _nav.reverse_forward_dist or _nav.reverse_phase_ticks > REVERSE_PHASE_TIMEOUT:
                    print(f"[REVERSE] Phase 1 done. Reverse dist = {_nav.reverse_odometry:.3f}m")
                    print(f"[REVERSE] Going to base via stage 12.")
                    _nav.stage = 12
                    _nav.progress = 0.0
                    _nav.turn_phase = 0
                    _nav.facing_back = True
                    _pid_wall.reset()
                    _pid_heading.reset()
                    target_vel = 0.0
                    target_ang = 0.0
                else:
                    target_vel = -REVERSE_VEL
                    target_ang = 0.0
                    if dist_left < 0.4:
                        target_ang = 0.12
                    elif dist_right < 0.4:
                        target_ang = -0.12

        else:
            # === SIMPLE TURN: no shelves on sides, just spin 180° ===
            if not _nav.turned:
                if _nav.turn_phase == 0:
                    print(f"[RETURN] Turning 180 to go back")
                    _nav.turn_phase = 1
                target_vel = 0.1
                target_ang = +0.35
                heading_diff = current_heading - _nav.heading_at_start
                heading_diff = abs(math.atan2(math.sin(heading_diff), math.cos(heading_diff)))
                if heading_diff > math.pi - 0.2:
                    print(f"[RETURN] Now facing back, returning to base")
                    _nav.turned = True
                    _nav.heading_at_start = current_heading
                    _nav.stage = 12
                    _nav.progress = 0.0
                    _nav.turn_phase = 0
                    _nav.facing_back = True
                    _pid_wall.reset()
                    _pid_heading.reset()
                    target_vel = 0.0
                    target_ang = 0.0

    # ===================== STAGE 12: Return to base (go south if needed, then west) =====================
    elif _nav.stage == 12:
        if robot_y > _nav.start_y + 0.3:
            target_heading = -math.pi / 1.8
            heading_error = target_heading - current_heading
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            if abs(heading_error) > 0.15:
                target_ang = max(-0.2, min(0.2, heading_error * 0.8))
                target_vel = 0.2
                _pid_wall.reset()
            else:
                wall_dist = dist_left
                error = wall_dist - DESIRED_WALL_DISTANCE
                error = max(-0.15, min(0.15, error))
                target_ang = _pid_wall.update(error, 0.1)
                target_vel = MAX_VEL
                if abs(error) > 0.1:
                    target_vel = MAX_VEL * 0.7
                target_ang = max(-0.2, min(0.2, target_ang))

            dy_left = robot_y - _nav.start_y
            if int(dy_left * 10) % 10 == 0 and dy_left > 0:
                print(f"[RETURN] Y left to go south: {dy_left:.2f}m")
        else:
            target_heading = math.pi
            heading_error = target_heading - current_heading
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            if abs(heading_error) > 0.15:
                target_ang = max(-0.2, min(0.2, heading_error * 0.8))
                target_vel = 0.2
                _pid_wall.reset()
            else:
                wall_dist = dist_left
                error = wall_dist - DESIRED_WALL_DISTANCE
                error = max(-0.15, min(0.15, error))
                target_ang = _pid_wall.update(error, 0.1)
                target_vel = MAX_VEL
                if abs(error) > 0.1:
                    target_vel = MAX_VEL * 0.7
                target_ang = max(-0.2, min(0.2, target_ang))

            dx_left = robot_x - _nav.start_x
            if int(dx_left * 10) % 10 == 0 and dx_left > 0:
                print(f"[RETURN] X left: {dx_left:.2f}m")

            if dx_left <= 0.3:
                print(f"\n[RETURN] Back at base! pos=({robot_x:.2f},{robot_y:.2f})")
                print(f"[RETURN] Switching to alternative path")
                _nav.attempt = 1
                _nav.start_x = robot_x
                _nav.start_y = robot_y
                _nav.stage = 0
                _nav.progress = 0.0
                _nav.heading_at_start = current_heading
                _nav.turned = False
                _nav.turn_phase = 0
                _nav.facing_back = False
                _nav.return_turn_done = False
                _pid_heading.reset()
                _pid_wall.reset()
                target_vel = 0.0
                target_ang = 0.0
                print(f"[NAV] Attempt 1: NORTH to {_nav.passage_y}, then EAST to shelf")

    # ===================== STAGE 3/5: Done =====================
    elif _nav.stage == 3 or _nav.stage == 5:
        target_vel = 0.0
        target_ang = 0.0

    velocity = _last_vel + SMOOTH * (target_vel - _last_vel)
    angular_rate = _last_ang + SMOOTH * (target_ang - _last_ang)
    _last_vel = velocity
    _last_ang = angular_rate

    velocity = max(-0.08, min(MAX_VEL, velocity))
    angular_rate = max(-MAX_ANG, min(MAX_ANG, angular_rate))

    return velocity, angular_rate
