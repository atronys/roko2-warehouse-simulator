
class Parameters:

    # Robot motion parameters
    x = 0
    y = 0
    heading = 0   #yaw
    velocity = 0    
    angular_rate = 0
    steering_angle = 0


    def __init__(self, x, y, heading, velocity, angular_rate, steering_angle=0):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = velocity
        self.angular_rate = angular_rate
        self.steering_angle = steering_angle
