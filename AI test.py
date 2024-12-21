import pygame
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

pygame.init()
screen = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE | pygame.DOUBLEBUF)
screen_width = screen.get_width()
screen_height = screen.get_height()
rand = random.Random()
fps = 30
fps_interval = 1000 / fps
last_tick = pygame.time.get_ticks()
last_frame = pygame.time.get_ticks()
clock = pygame.time.Clock()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = []



def get_biased_random_float(start, end):
    # Generate a random float between 0 and 1, then square it to bias towards lower values
    biased_random = random.random() ** 2
    result = round(start + (end - start) * biased_random, 1)
    return result


def get_distance(pos_x_1, pos_y_1, pos_x_2, pos_y_2):
    distance = math.sqrt((pos_x_2 - pos_x_1) ** 2 + (pos_y_2 - pos_y_1) ** 2)
    return distance


def get_rotated_vertices(center_x, center_y, size, angle):
    """Returns the vertices of a rotated square."""
    half_size = size / 2
    angle = math.radians(angle)

    # Define corners relative to the center
    corners = [
        (-half_size, -half_size),
        (half_size, -half_size),
        (half_size, half_size),
        (-half_size, half_size)
    ]

    # Rotate and translate the corners
    vertices = []
    for corner_x, corner_y in corners:
        rotated_x = math.cos(angle) * corner_x - math.sin(angle) * corner_y + center_x
        rotated_y = math.sin(angle) * corner_x + math.cos(angle) * corner_y + center_y
        vertices.append((rotated_x, rotated_y))
    return vertices


def is_circle_in_rotated_square(circle_x, circle_y, radius, vertices):
    """Check if a circle intersects with a rotated square using SAT."""
    for i in range(len(vertices)):
        # Get the current and next vertex
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]

        # Find the edge vector
        edge_x = x2 - x1
        edge_y = y2 - y1

        # Find the perpendicular (normal)
        normal_x = -edge_y
        normal_y = edge_x

        # Normalize the normal vector
        length = math.sqrt(normal_x ** 2 + normal_y ** 2)
        normal_x /= length
        normal_y /= length

        # Project all vertices onto the normal
        min_square, max_square = float('inf'), float('-inf')
        for vx, vy in vertices:
            projection = (vx * normal_x + vy * normal_y)
            min_square = min(min_square, projection)
            max_square = max(max_square, projection)

        # Project the circle onto the normal (extend the circle by its radius)
        circle_projection = (circle_x * normal_x + circle_y * normal_y)
        min_circle = circle_projection - radius
        max_circle = circle_projection + radius

        # Check for overlap
        if max_square < min_circle or max_circle < min_square:
            return False  # No overlap on this axis, so no collision
    return True  # Overlaps all axes, so there is a collision


def is_point_in_rotated_square(point_x, point_y, vertices):
    """Check if a point is inside a rotated square using SAT."""
    for i in range(len(vertices)):
        # Get the current and next vertex
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]

        # Find the edge vector
        edge_x = x2 - x1
        edge_y = y2 - y1

        # Find the perpendicular (normal)
        normal_x = -edge_y
        normal_y = edge_x

        # Project all vertices onto the normal
        min_square, max_square = float('inf'), float('-inf')
        for vx, vy in vertices:
            projection = (vx * normal_x + vy * normal_y)
            min_square = min(min_square, projection)
            max_square = max(max_square, projection)

        # Project the ball center onto the normal
        ball_projection = (point_x * normal_x + point_y * normal_y)

        # Check for overlap
        if not (min_square <= ball_projection <= max_square):
            return False  # No overlap on this axis, so no collision
    return True  # Overlaps all axes, so the point is inside the polygon


def draw_debug_info(ball, square_vertices):
    print((int(ball.x), int(ball.y)))
    pygame.draw.circle(screen, (255, 0, 0), (int(ball.x), int(ball.y)), ball.radius, 1)
    # Draw the square's collision boundaries
    pygame.draw.polygon(screen, (0, 255, 0), square_vertices, 1)


def closest_point_on_line(px, py, x1, y1, x2, y2):
    """Find the closest point on a line segment to a given point (px, py)."""
    line_dx = x2 - x1
    line_dy = y2 - y1
    length_squared = line_dx ** 2 + line_dy ** 2

    # Project point onto the line, clamped to the segment
    t = max(0, min(1, ((px - x1) * line_dx + (py - y1) * line_dy) / length_squared))
    closest_x = x1 + t * line_dx
    closest_y = y1 + t * line_dy

    return closest_x, closest_y


def draw_mtv(ball, mtv_x, mtv_y):
    pygame.draw.line(
        screen,
        (255, 255, 0),
        (ball.x, ball.y),
        (ball.x + mtv_x * ball.radius, ball.y + mtv_y * ball.radius),
        2
    )
    #pygame.draw.circle(screen, (255, 0, 0), (ball.x, ball.y), 4)


class Ball:
    def __init__(self, x, y, radius, velocity, color, ai=False):
        self.x = x
        self.y = y
        self.radius = radius
        self.vel_x, self.vel_y = velocity
        self.color = color
        self.counter = 0
        self.ai = ai
        self.model = None
        self.optimizer = None

    def move(self):
        if not self.ai:
            # Update ball position
            self.x += self.vel_x
            self.y += self.vel_y
            # Bounce off screen edges
            if self.x - self.radius < 0 or self.x + self.radius > screen_width:
                self.vel_x = -self.vel_x
                self.x = max(self.radius, min(screen_width - self.radius, self.x))
            if self.y - self.radius < 0 or self.y + self.radius > screen_height:
                self.vel_y = -self.vel_y
                self.y = max(self.radius, min(screen_height - self.radius, self.y))

            # Check collisions with rotating enemies
            for enemy in enemies.red_nemesis_list:
                vertices = enemy["vertices"]
                if is_circle_in_rotated_square(self.x, self.y, self.radius, vertices):
                    prev_x, prev_y = self.x, self.y
                    self.resolve_collision_with_square(enemy)
                    if is_point_in_rotated_square(self.x, self.y, vertices):
                        abs_x = self.x - prev_x
                        abs_y = self.y - prev_y
                        self.x += abs_x
                        self.y += abs_y
        else:
            """"Ball AI (Neural network taught by the reference ball, the "balls" ball, preferably pytorch and with my RTX 4060) code goes here, Parameters for input of the AI include: Position of the ball and the distance 
            between the ball and each of the 3 closest enemies and their rotation and the distance to the 4 walls"""
            inputs1 = self.collect_ai_inputs()
            reward = self.calculate_reward()

            # Make the forward pass to compute log probabilities
            outputs = self.model(inputs1)  # Forward pass
            # Now, log_probabilities will be populated

            # Apply predictions to velocity
            self.vel_x, self.vel_y = outputs[0].detach().cpu().numpy()

            self.vel_x *= 20  # Scale back to screen dimensions
            self.vel_y *= 20

            self.x += self.vel_x
            self.y += self.vel_y

            if self.x < -100 or self.x > screen_width + 100:  # Extreme bounds
                self.x = screen_width / 2  # Reset to center
                self.vel_x = 0  # Reset velocity
            if self.y < -100 or self.y > screen_height + 100:  # Extreme bounds
                self.y = screen_height / 2  # Reset to center
                self.vel_y = 0  # Reset velocity

            # Now update the model with the reward
            self.update_model(reward)

    def calculate_reward(self):
        global inputs
        reward = 0

        # Distance to the closest enemy
        closest_enemy = min(
            enemies.red_nemesis_list,
            key=lambda e: math.sqrt((ai_ball.x - e["pos_x"]) ** 2 + (ai_ball.y - e["pos_y"]) ** 2)
        )
        dist_to_enemy = math.sqrt((ai_ball.x - closest_enemy["pos_x"]) ** 2 + (ai_ball.y - closest_enemy["pos_y"]) ** 2)

        # # Check if the ball is out of bounds and penalize
        # if self.x < 0 or self.x > screen_width:
        #     self.vel_x = -self.vel_x * 0.8  # Reduce speed when bouncing
        #     reward -= 5  # Stronger penalty for going out of bounds
        # elif self.y < 0 or self.y > screen_height:
        #     self.vel_y = -self.vel_y * 0.8  # Reduce speed when bouncing
        #     reward -= 5  # Stronger penalty for going out of bounds

            # Reward for staying away from enemies (closer = higher reward)
        reward -= max(0, 1 - dist_to_enemy / screen_width)  # Normalize distance to a scale based on screen width

        # Optionally, add a small reward for staying close to the center of the screen (if desired)
        # This will make the AI prefer the center area and avoid the edges.
        center_dist = math.sqrt((ai_ball.x - screen_width / 2) ** 2 + (ai_ball.y - screen_height / 2) ** 2)
        reward -= center_dist / screen_width * 4 # Penalty for moving away from the center, optional

        # Staying away from walls
        dist_to_left = ai_ball.x
        dist_to_right = screen_width - ai_ball.x
        dist_to_top = ai_ball.y
        dist_to_bottom = screen_height - ai_ball.y

        reward += max(0, dist_to_left / screen_width)
        reward += max(0, dist_to_right / screen_width)
        reward += max(0, dist_to_top / screen_height)
        reward += max(0, dist_to_bottom / screen_height)

        # Encourage motion (optional)
        reward += math.sqrt(ai_ball.vel_x ** 2 + ai_ball.vel_y ** 2) / screen_width
        if reward < 0: reward = 0
        return reward

    def collect_ai_inputs(self):
        global inputs

        # Ball's position
        inputs.extend([self.x / screen_width, self.y / screen_height])  # Normalize positions

        # Closest enemies
        closest_enemies = sorted(
            enemies.red_nemesis_list,
            key=lambda e: math.sqrt((self.x - e["pos_x"]) ** 2 + (self.y - e["pos_y"]) ** 2)
        )[:3]  # Get the 3 closest enemies

        for enemy in closest_enemies:
            distance = math.sqrt((self.x - enemy["pos_x"]) ** 2 + (self.y - enemy["pos_y"]) ** 2) / max(screen_width,
                                                                                                        screen_height)
            inputs.append(distance)
            inputs.append(enemy["rotation"] / 360)  # Normalize rotation

        # Distance to walls
        inputs.append(self.x / screen_width)  # Left wall
        inputs.append((screen_width - self.x) / screen_width)  # Right wall
        inputs.append(self.y / screen_height)  # Top wall
        inputs.append((screen_height - self.y) / screen_height)  # Bottom wall

        return torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)  # Batch dimension

    def update_model(self, reward):
        self.optimizer.zero_grad()

        # Convert the reward to a tensor
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)

        # Make sure log_probabilities is not None
        if self.model.log_probabilities is not None:
            # Calculate loss
            print(reward_tensor, reward)
            loss = -reward_tensor * self.model.log_probabilities # Maximize reward
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
        else:
            print("log_probabilities is None. Ensure the forward pass is called first.")

    def resolve_collision_with_square(self, enemy):
        self.counter += 1
        if not self.ai:
            if self.counter == 1:
                self.counter = 0
                vertices = enemy["vertices"]

                # Check if the ball is inside the square
                if is_circle_in_rotated_square(self.x, self.y, self.radius, vertices):
                    # Initialize variables for the closest MTV
                    closest_mtv_x, closest_mtv_y = 0, 0
                    min_overlap = float('inf')
                    ball_inside = True

                    for i in range(len(vertices)):
                        x1, y1 = vertices[i]
                        x2, y2 = vertices[(i + 1) % len(vertices)]

                        # Find the closest point on the edge to the ball's center
                        closest_x, closest_y = closest_point_on_line(self.x, self.y, x1, y1, x2, y2)

                        # Calculate the distance and MTV for this edge
                        dx = self.x - closest_x
                        dy = self.y - closest_y
                        distance = math.sqrt(dx ** 2 + dy ** 2)

                        if distance == 0:
                            continue  # Skip this edge if the ball center is exactly on it

                        overlap = self.radius - distance
                        if overlap > 0:  # Check if there's an overlap
                            ball_inside = False  # Ball is not fully inside; it's overlapping an edge
                            if overlap < min_overlap:
                                min_overlap = overlap
                                closest_mtv_x, closest_mtv_y = dx / distance, dy / distance  # Normalized MTV

                    # If the ball is fully inside, push it out to the closest edge
                    if ball_inside:
                        center_dx = self.x - enemy["pos_x"]
                        center_dy = self.y - enemy["pos_y"]
                        center_distance = math.sqrt(center_dx ** 2 + center_dy ** 2)

                        if center_distance > 0:
                            closest_mtv_x = center_dx / center_distance
                            closest_mtv_y = center_dy / center_distance
                            min_overlap = self.radius

                    # Apply MTV if a valid overlap was found
                    if min_overlap > 0 and min_overlap != float('inf'):
                        self.x += closest_mtv_x * min_overlap
                        self.y += closest_mtv_y * min_overlap
                        draw_mtv(balls, closest_mtv_x * min_overlap, closest_mtv_y * min_overlap)

                        # Reflect the velocity using the MTV
                        normal_x, normal_y = closest_mtv_x, closest_mtv_y
                        dot = self.vel_x * normal_x + self.vel_y * normal_y
                        self.vel_x -= 2 * dot * normal_x
                        self.vel_y -= 2 * dot * normal_y


    @staticmethod
    def draw():
        for enemy in enemies.red_nemesis_list:
            vertices = enemy["vertices"]
            if vertices:
                draw_debug_info(balls, vertices)
                draw_debug_info(ai_ball, vertices)
        #pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


class BallAI(nn.Module):
    def __init__(self):
        super(BallAI, self).__init__()
        self.log_probabilities = None
        self.fc1 = nn.Linear(12, 64)  # 9 inputs (position + distances)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # 2 outputs (vel_x, vel_y)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for outputs
        # Log probabilities (example, you can adjust this to match your model)
        log_probs = F.log_softmax(x, dim=-1)

        # Store the log probabilities for later use
        self.log_probabilities = log_probs
        #print("ai is done"+str(x))
        return x

class Enemy:
    def __init__(self):
        self.red_nemesis_list = []

    def move(self):
        for i in range(len(self.red_nemesis_list) - 1, -1, -1):  # Iterate in reverse
            red_nemesis = self.red_nemesis_list[i]
            speed = red_nemesis["speed"]
            red_nemesis["pos_x"] += red_nemesis["vel_x"]
            red_nemesis["pos_y"] += red_nemesis["vel_y"]
            if red_nemesis["rotation"] <= 0:
                red_nemesis["rotation"] -= speed / 3
            else:
                red_nemesis["rotation"] += speed / 3
            if red_nemesis["rotation"] >= 360:
                repeat_angle = red_nemesis["rotation"] - 360
                red_nemesis["rotation"] = repeat_angle
            if red_nemesis["rotation"] <= -360:
                repeat_angle = red_nemesis["rotation"] + 360
                red_nemesis["rotation"] = repeat_angle
            if abs(red_nemesis["pos_x"]) > screen_width * 2 or abs(red_nemesis["pos_y"]) > screen_height * 2:
                self.red_nemesis_list.pop(i)
                print("Enemy 'optimized', Enemy amount: " + str(len(self.red_nemesis_list)))
            if len(self.red_nemesis_list) < 4:
                self.wave()

    def choose_random_props(self):
        width = screen.get_width()
        height = screen.get_height()
        margin = 100
        x, y = 0, 0
        for red_nemesis in self.red_nemesis_list:
            if not red_nemesis["random_props_chosen"]:
                #if red_nemesis["label"] == "random_guy":
                if red_nemesis["speed"] is None:
                    red_nemesis["speed"] = get_biased_random_float(0.5, 3)
                if red_nemesis["size"] is None:
                    red_nemesis["size"] = rand.randint(50, 80)
                if red_nemesis["rotation"] is None:
                    red_nemesis["rotation"] = rand.randint(-1, 2)

                edge = random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top':
                    x = random.randint(-margin, width + margin)
                    y = -margin
                elif edge == 'bottom':
                    x = random.randint(-margin, width + margin)
                    y = height + margin
                elif edge == 'left':
                    x = -margin
                    y = random.randint(-margin, height + margin)
                elif edge == 'right':
                    x = width + margin
                    y = random.randint(-margin, height + margin)
                if red_nemesis["pos_x"] is None:
                    red_nemesis["pos_x"] = x
                if red_nemesis["pos_y"] is None:
                    red_nemesis["pos_y"] = y

                red_nemesis["surface"] = pygame.Surface((red_nemesis["size"], red_nemesis["size"]), pygame.SRCALPHA)

                if red_nemesis["direction"] is None:
                    if red_nemesis["pos_x"] <= screen_width / 2 and red_nemesis["pos_y"] <= screen_height / 2:
                        red_nemesis["direction"] = "se"
                        red_nemesis["vel_x"] += red_nemesis["speed"]
                        red_nemesis["vel_y"] += red_nemesis["speed"]
                    if red_nemesis["pos_x"] >= screen_width / 2 and red_nemesis["pos_y"] <= screen_height / 2:
                        red_nemesis["direction"] = "sw"
                        red_nemesis["vel_x"] -= red_nemesis["speed"]
                        red_nemesis["vel_y"] += red_nemesis["speed"]
                    if red_nemesis["pos_x"] <= screen_width / 2 and red_nemesis["pos_y"] >= screen_height / 2:
                        red_nemesis["direction"] = "ne"
                        red_nemesis["vel_x"] += red_nemesis["speed"]
                        red_nemesis["vel_y"] -= red_nemesis["speed"]
                    if red_nemesis["pos_x"] >= screen_width / 2 and red_nemesis["pos_y"] >= screen_height / 2:
                        red_nemesis["direction"] = "nw"
                        red_nemesis["vel_x"] -= red_nemesis["speed"]
                        red_nemesis["vel_y"] -= red_nemesis["speed"]
                red_nemesis["random_props_chosen"] = True

    def wave(self):
        #if self.red_nemesis_list: self.red_nemesis_list.clear()
        enemy_amount = rand.randint(2, 8)
        for _ in range(enemy_amount):
            self.create_enemy(None, None, None, None, (255, 0, 0), "random_guy",
                              30, 100, 100, None, None, False, None)
        #next_round =

    def create_enemy(self, pos_x, pos_y, size, speed, color, label, age, hp, max_hp, rotation, direction, is_attacked,
                     weirdness):
        enemy_properties = {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "vel_x": 0,
            "vel_y": 0,
            "size": size,
            "speed": speed,
            "color": color,
            "label": label,
            "age": age,
            "hp": hp,
            "max_hp": max_hp,
            "rotation": rotation,
            "vertices": [],
            "last_rotation": 0,
            "direction": direction,
            "is_attacked": is_attacked,
            "weirdness": weirdness,
            "random_props_chosen": False,
            "surface": None,
            "image": None,
            "rotated_surface": pygame.Surface((40, 40), pygame.SRCALPHA)
        }
        self.red_nemesis_list.append(enemy_properties)
        self.choose_random_props()

    def draw(self):
        for red_nemesis in self.red_nemesis_list:
            pos_x = red_nemesis["pos_x"]
            pos_y = red_nemesis["pos_y"]
            size = red_nemesis["size"]
            color = red_nemesis["color"]
            hp = red_nemesis["hp"]
            max_hp = red_nemesis["max_hp"]
            # Calculate vertices based on position, size, and rotation
            red_nemesis["vertices"] = get_rotated_vertices(
                red_nemesis["pos_x"],
                red_nemesis["pos_y"],
                red_nemesis["size"],
                red_nemesis["rotation"]
            )
            vertices = red_nemesis["vertices"]
            if red_nemesis["is_attacked"]:
                hp_factor = hp / max_hp
                pygame.draw.rect(screen, (255, 0, 0), (pos_x - size / 2, pos_y + size, size, size / 10))
                pygame.draw.rect(screen, (0, 255, 0), (pos_x - size / 2, pos_y + size, size * hp_factor + 1, size / 10))
            pygame.draw.polygon(screen, color, vertices)
            pygame.draw.polygon(screen, (255, 255, 0), vertices, 3)

ball_velocity = (rand.randint(1,5), rand.randint(1,5))
ball_position = (rand.randint(100,1000), rand.randint(100,1000))
balls = Ball(500, 500, 20, ball_velocity, (255, 255, 0))
ai_ball = Ball(500, 500, 20, ball_velocity, (255, 255, 0), True)
ai_ball.model = BallAI().to(device)  # Your PyTorch model
ai_ball.optimizer = torch.optim.Adam(ai_ball.model.parameters(), lr=0.01)
print(ai_ball.model, ai_ball.optimizer)


enemies = Enemy()
enemies.wave()

running = True
while running:
    current_time = pygame.time.get_ticks()

    # Calculate the time since the last logical tick and the last frame
    time_since_last_tick = current_time - last_tick
    time_since_last_frame = current_time - last_frame
    for event in pygame.event.get():
        pass
    if time_since_last_frame >= (1000 // fps):  # Run FPS updates
        last_frame = current_time
        screen.fill((0, 0, 0))
        balls.move()
        ai_ball.move()
        enemies.move()
        balls.draw()
        enemies.draw()
        inputs = []
        pygame.display.flip()

#    clock.tick(max(fps, 30))
