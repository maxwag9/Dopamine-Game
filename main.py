import ctypes
import math
import os
import random
import xml.etree.ElementTree as ET
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from xml.dom import minidom
import pygame

pygame.init()
info = pygame.display.Info()
tv_width, tv_height = info.current_w, info.current_h
screen = pygame.display.set_mode((tv_width, tv_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)
sdl = ctypes.CDLL("SDL2.dll")
hwnd = pygame.display.get_wm_info()['window']
user32 = ctypes.windll.user32
screen_width = screen.get_width()
screen_height = screen.get_height()
clock = pygame.time.Clock()
reddings_amount = 0
rand = random.Random()
user_name = "standard"
mouse_pos = (0, 0)
debug_mode = 0
playing = False
enemy_image_path = None
enemy_image = screen
tps = 60
fps = 60
tps_interval = 1000 / tps  # Milliseconds per tick
fps_interval = 1000 / fps  # Milliseconds per frame
last_tick = pygame.time.get_ticks()
last_frame = pygame.time.get_ticks()
bg_color = (0, 0, 0)

def null_window_position(x=0, y=0):
    user32.MoveWindow(hwnd, x, y, tv_width, tv_height, True)

def switch_to_borderless():
    global screen
    screen = pygame.display.set_mode((tv_width, tv_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)
    #user32.ShowWindow(hwnd, 1)  # Show window normally (not minimized)
    null_window_position()  # Reset the position after making it borderless

# Function to switch to windowed mode
def switch_to_windowed():
    global screen
    screen = pygame.display.set_mode((tv_width, tv_height), pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
    null_window_position(0, 1)
    user32.ShowWindow(hwnd, 3)  # Maximize the window when switching

# Function to switch to fullscreen mode
def switch_to_fullscreen():
    global screen
    screen = pygame.display.set_mode((tv_width, tv_height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)


def select_texture_and_scale():
    root = Tk()
    root.withdraw()

    file_path = askopenfilename(
        title="Select Texture File",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    root.destroy()  # Close the Tkinter window

    if not file_path:  # If no file was selected
        print("No texture file selected!")
        return None

    try:
        texture = pygame.image.load(file_path).convert_alpha()
        scaled_texture = pygame.transform.scale(texture, (50, 50))
        print(f"Texture loaded and scaled from {file_path}")
        return scaled_texture, file_path
    except pygame.error as e:
        print(f"Error loading texture: {e}")
        return None


def get_biased_random_float(start, end):
    # Generate a random float between 0 and 1, then square it to bias towards lower values
    biased_random = random.random() ** 2
    result = round(start + (end - start) * biased_random, 1)
    return result


def get_distance(pos_x_1, pos_y_1, pos_x_2, pos_y_2):
    distance = math.sqrt((pos_x_2 - pos_x_1) ** 2 + (pos_y_2 - pos_y_1) ** 2)
    return distance


def check_collision_between_2_rects(rectangle1, rectangle2):
    rect1 = pygame.Rect(rectangle1["pos_x"] - rectangle1["size"] / 2,
                        rectangle1["pos_y"] - rectangle1["size"] / 2,
                        rectangle1["size"], rectangle1["size"])
    size = rectangle2["size"]
    rect2 = pygame.Rect(rectangle2["pos_x"] - size / 2, rectangle2["pos_y"] - size / 2, size, size)
    return rect1.colliderect(rect2)


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


def check_collision(red_nemesis1):
    rect1 = pygame.Rect(red_nemesis1["pos_x"] - red_nemesis1["size"] / 2,
                        red_nemesis1["pos_y"] - red_nemesis1["size"] / 2,
                        red_nemesis1["size"], red_nemesis1["size"])
    collided = False
    for cross_collision in crosshair.crosshair_list:
        size = cross_collision["size"]
        rect2 = pygame.Rect(cross_collision["pos_x"] - size / 2, cross_collision["pos_y"] - size / 2, size, size)
        if not collided:
            collided = rect1.colliderect(rect2)
    return collided


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


class GameState:
    def __init__(self):
        # Use a fixed filename in the current working directory
        # Get the user's Documents folder
        user_folder = os.path.expanduser("~")
        documents_folder = os.path.join(user_folder, "Documents")
        global user_name
        user_name = os.path.basename(user_folder)
        menu.change_label("Hello there "+str(user_name)+"!", "Username BL")
        # Construct the path to the save file
        self.save_folder = os.path.join(documents_folder, "dg")

        # Create the 'dg' folder if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Set the full file path to the save game file
        self.file_path = os.path.join(self.save_folder, "gamesave.xml")
        self.gamestate_list = []
        self.load()  # Load the save file when initializing the game state

    def save(self):
        print("Saving game state...")

        # Create the root element
        root = ET.Element("game_states")

        # Populate XML with gamestate_list
        if self.gamestate_list:
            for gamestate in self.gamestate_list:
                gamestate_elem = ET.SubElement(root, "crosshair")
                gamestate_elem.set("crosshair_size", str(gamestate.get("crosshair_size", 0)))
                gamestate_elem.set("shooting_speed", str(gamestate.get("shooting_speed", 0)))
                gamestate_elem.set("damage", str(gamestate.get("damage", 0)))
                gamestate_elem.set("reddings", str(gamestate.get("reddings", 0)))

            gamestate_graphics = ET.SubElement(root, "graphics")
            gamestate_graphics.set("debug_mode", str(debug_mode if 'debug_mode' in globals() else 0))
            if enemy_image_path is not None:
                gamestate_graphics.set("enemy_image_path", str(enemy_image_path))
                print("Saved enemy texture to xml: " + str(enemy_image_path))

            # Pretty format XML string
            rough_string = ET.tostring(root, encoding="utf-8")
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="    ")

            # Write to file
            with open(self.file_path, "w", encoding="utf-8") as file:
                file.write(pretty_xml)

            print(f"Game state saved to {self.file_path}.")

    def load(self):
        global debug_mode, enemy_image_path, enemy_image

        if not os.path.exists(self.file_path):
            print(f"{self.file_path} not found. Creating default save file...")
            # Create default values
            self.gamestate_list = [
                {
                    "crosshair_size": 5,
                    "shooting_speed": 1,
                    "damage": 10,
                    "reddings": 0,
                }
            ]
            debug_mode = 0
            enemy_image_path = None

            # Use save to create the file
            self.save()
        else:
            print(f"Loading save from {self.file_path}.")
            tree = ET.parse(self.file_path)
            root = tree.getroot()

            # Loading graphics and debug_mode from the XML
            gamestate_graphics = root.find("graphics")
            debug_mode = int(gamestate_graphics.get("debug_mode", 0))
            enemy_image_path = gamestate_graphics.get("enemy_image_path")
            if enemy_image_path:
                print("Loaded enemy image path from XML.")
                enemy_image = pygame.image.load(enemy_image_path).convert_alpha()
            else:
                enemy_image = None

            # Clear current gamestate_list and load from XML
            self.gamestate_list = []
            for gamestate_elem in root.findall("crosshair"):
                gamestate = {
                    "crosshair_size": int(gamestate_elem.get("crosshair_size", 5)),
                    "shooting_speed": int(gamestate_elem.get("shooting_speed", 1)),
                    "damage": int(gamestate_elem.get("damage", 10)),
                    "reddings": int(gamestate_elem.get("reddings", 0)),
                }
                self.gamestate_list.append(gamestate)


    def apply_load_ingame(self):
        if len(self.gamestate_list) != 0:
            menu.change_label("Reddings: " + str(self.gamestate_list[0]["reddings"]), "Reddings BL")
            for i in range(len(self.gamestate_list)):
                ch = self.gamestate_list[i]
                crosshair.create_honeycomb_crosshairs(0, 0, ch["crosshair_size"], ch["damage"], ch["shooting_speed"],
                                                      (255, 0, 0))


class Rectangles:
    def __init__(self):
        self.rect_list = []
        self.line_list = []

    def draw_rects(self, surface, color, rect, mode):
        if mode == "add":
            rect_properties = {
                "surface": surface,
                "color": color,
                "rect": rect
            }
            self.rect_list.append(rect_properties)
        elif mode == "draw":
            for rect in self.rect_list:
                pygame.draw.rect(rect["surface"], rect["color"], rect["rect"])

    def draw_lines(self, surface, color, pos1, pos2, thickness, mode):
        if mode == "add":
            line_properties = {
                "surface": surface,
                "color": color,
                "pos1": pos1,
                "pos2": pos2,
                "thickness": thickness
            }
            self.line_list.append(line_properties)
        elif mode == "draw":
            for line in self.line_list:
                self.draw_line(line["surface"], line["pos1"], line["pos2"], line["color"], line["thickness"])

    @staticmethod
    def draw_line(surface, pos1, pos2, color=(255, 255, 255), thickness=2):
        pygame.draw.line(surface, color, pos1, pos2, thickness)


class Particle:
    def __init__(self):
        self.particle_list = []

    def create_particle(self, pos_x, pos_y, size, speed, color, label, max_gen):
        """Initialize a new particle system."""
        total_particles = sum(4 ** gen for gen in range(1, max_gen + 1))
        particle_properties = {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "particle_x": [pos_x] * 4,  # Initial positions
            "particle_y": [pos_y] * 4,
            "size": size,
            "speed": speed,
            "color": color,
            "label": label,
            "rotation": 0,
            "directions": [],  # Directions for each particle
            "generation": [1] * 4,  # Generation for each particle
            "direction_chosen": [False] * max_gen,
            "current_gen": 1,
            "max_gen": max_gen,
            "timer": 0,
            "flip_speed": speed,
            "end_index": 4,  # Tracks how many particles are currently active
            "total_particles": total_particles,
            "vel_x": 0,
            "vel_y": 0
        }
        self.particle_list.append(particle_properties)

    def draw(self):
        """Update and render all particles."""
        for e in range(len(self.particle_list) - 1, -1, -1):  # Iterate in reverse
            particle = self.particle_list[e]
            if particle["timer"] > 60 * particle["max_gen"]:
                self.particle_list.pop(e)
                break
            # Increment rotation
            particle["rotation"] = (particle["rotation"] + 1) % 360
            label = particle["label"]
            if label == "damage":
                if particle["timer"] > 60 and particle["current_gen"] < particle["max_gen"]:
                    particle["current_gen"] += 1
                    particle["timer"] = 0
                else:
                    particle["timer"] += 1

                # Initialize directions for the current generation
                if not particle["direction_chosen"][particle["current_gen"] - 1]:
                    start_index = sum(4 ** gen for gen in range(1, particle["current_gen"]))
                    end_index = start_index + 4 ** particle["current_gen"]
                    particle["end_index"] = end_index

                    # Copy positions from the previous generation
                    prev_start_index = start_index - 4 ** (particle["current_gen"] - 1)
                    for i in range(start_index, end_index):
                        source_index = prev_start_index + (i - start_index) // 4
                        particle["particle_x"].append(particle["particle_x"][source_index])
                        particle["particle_y"].append(particle["particle_y"][source_index])

                    # Assign directions and mark generation as initialized
                    for i in range(start_index, end_index):
                        particle["directions"].append(random.choice(["nw", "ne", "sw", "se"]))
                        particle["generation"].append(particle["current_gen"])
                    particle["direction_chosen"][particle["current_gen"] - 1] = True

                # Move particles based on directions
                for i in range(len(particle["particle_x"])):
                    if i >= particle["end_index"]:  # Only process active particles
                        break

                    dx, dy = 0, 0
                    direction = particle["directions"][i]
                    if direction == "nw":
                        dx, dy = -particle["speed"], -particle["speed"]
                    elif direction == "ne":
                        dx, dy = particle["speed"], -particle["speed"]
                    elif direction == "sw":
                        dx, dy = -particle["speed"], particle["speed"]
                    elif direction == "se":
                        dx, dy = particle["speed"], particle["speed"]

                    particle["particle_x"][i] += dx
                    particle["particle_y"][i] += dy

                # Render particles
                for i in range(len(particle["particle_x"])):
                    if particle["generation"][i] == particle["current_gen"]:
                        size_factor = particle["size"] / (2.1 ** particle["generation"][i])
                        particle_surface = pygame.Surface((size_factor, size_factor), pygame.SRCALPHA)
                        pygame.draw.rect(
                            particle_surface,
                            particle["color"],
                            (0, 0, size_factor, size_factor)
                        )
                        rotated_particle = pygame.transform.rotate(particle_surface, particle["rotation"])
                        rotated_rect = rotated_particle.get_rect(center=(
                            particle["particle_x"][i],
                            particle["particle_y"][i]
                        ))
                        screen.blit(rotated_particle, rotated_rect)
            elif label == "reddings":
                pos_x, pos_y = particle["pos_x"], particle["pos_y"]
                size = particle["size"]
                particle["rotation"] += particle["flip_speed"]
                flip_scale = abs(math.sin(math.radians(particle["rotation"])))
                scaled_width = int(size * flip_scale)

                # Initialize total forces
                force_x, force_y = 0, 0

                # Compute gravitational force from all crosshairs
                crosshairs = crosshair.crosshair_list[0]
                cross_x, cross_y = crosshairs["pos_x"], crosshairs["pos_y"]
                distance = get_distance(pos_x, pos_y, cross_x, cross_y)

                # Avoid division by zero for very small distances
                if distance > 1:
                    # Gravity-like force: proportional to 1 / (distance ** 2)
                    strength = 200 / (distance ** 2)  # Adjust 10 to control the strength of attraction
                    direction_x = (cross_x - pos_x) / distance
                    direction_y = (cross_y - pos_y) / distance
                    force_x += strength * direction_x
                    force_y += strength * direction_y
                if check_collision(particle):
                    if len(game_state.gamestate_list) != 0:
                        game_state.gamestate_list[0]["reddings"] += 1
                        menu.change_label("Reddings: " + str(game_state.gamestate_list[0]["reddings"]), "Reddings BL")
                    self.particle_list.pop(e)
                    break

                # Update position with accumulated force
                particle["vel_x"] += force_x
                particle["vel_y"] += force_y

                # Apply velocity to position
                particle["pos_x"] += particle["vel_x"]
                particle["pos_y"] += particle["vel_y"]

                # Create the particle graphics
                if scaled_width > 0:
                    particle_surface = pygame.Surface((size, size), pygame.SRCALPHA)

                    # Define the stretched diamond shape
                    diamond = [
                        (size // 2, 0),  # Top point
                        (3 * size // 4, size // 2),  # Right point (closer to center horizontally)
                        (size // 2, size - 1),  # Bottom point
                        (size // 4, size // 2)  # Left point (closer to center horizontally)
                    ]
                    pygame.draw.polygon(particle_surface, particle["color"], diamond)
                    pygame.draw.polygon(particle_surface, (255, 255, 0), diamond, width=2)

                    base_color = particle["color"]
                    brightness_factor = min(255, int(255 * (scaled_width / size)))
                    shine_color = (
                        min(255, base_color[0] + brightness_factor),  # Brightened red
                        min(255, base_color[1] + brightness_factor),  # Brightened green
                        min(255, base_color[2] + brightness_factor),  # Brightened blue
                        255  # Semi-transparent alpha
                    )

                    # Draw a simple shine triangle over the right side
                    shine_triangle = [
                        (size // 2, size // 4),  # Near the top-center
                        (3 * size // 5, size // 2),  # Right point
                        (size // 2, 3 * size // 4)  # Near the bottom-center
                    ]
                    pygame.draw.polygon(particle_surface, shine_color, shine_triangle)

                    # Scale the particle to simulate horizontal flipping
                    scaled_particle = pygame.transform.scale(particle_surface, (scaled_width, size))
                    scaled_rect = scaled_particle.get_rect(center=(pos_x, pos_y))
                    screen.blit(scaled_particle, scaled_rect)

class Ball:
    def __init__(self, x, y, radius, velocity, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.vel_x, self.vel_y = velocity
        self.color = color
        self.counter = 0

    def move(self):
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

    def resolve_collision_with_square(self, enemy):
        self.counter += 1
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
                    if debug_mode == 1:
                        draw_mtv(balls, closest_mtv_x * min_overlap, closest_mtv_y * min_overlap)

                    # Reflect the velocity using the MTV
                    normal_x, normal_y = closest_mtv_x, closest_mtv_y
                    dot = self.vel_x * normal_x + self.vel_y * normal_y
                    self.vel_x -= 2 * dot * normal_x
                    self.vel_y -= 2 * dot * normal_y

    def draw(self):
        if debug_mode == 1:
            for enemy in enemies.red_nemesis_list:
                vertices = enemy["vertices"]
                if vertices:
                    draw_debug_info(balls, vertices)
        else:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


class Enemy:
    def __init__(self):
        self.red_nemesis_list = []

    def move(self):
        for i in range(len(self.red_nemesis_list) - 1, -1, -1):  # Iterate in reverse
            red_nemesis = self.red_nemesis_list[i]
            speed = red_nemesis["speed"]
            red_nemesis["pos_x"] += red_nemesis["vel_x"]
            red_nemesis["pos_y"] += red_nemesis["vel_y"]
            if check_collision(red_nemesis):
                red_nemesis["color"] = (100, 0, 0)
            else:
                red_nemesis["color"] = (255, 0, 0)
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
            if len(self.red_nemesis_list) < 70:
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
        enemy_amount = rand.randint(20, 80)
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
            global debug_mode, enemy_image, enemy_image_path
            if debug_mode == 1:
                pygame.draw.polygon(screen, color, vertices, 3)
            if debug_mode == 0:
                pygame.draw.polygon(screen, color, vertices)
                pygame.draw.polygon(screen, (255, 255, 0), vertices, 3)
            if debug_mode == 2:
                if enemy_image is not None:
                    red_nemesis["image"] = pygame.transform.scale(enemy_image, (size, size))
                if red_nemesis["image"] is not None:
                    if int(red_nemesis["rotation"]) != int(red_nemesis["last_rotation"]):
                        red_nemesis["last_rotation"] = red_nemesis["rotation"]
                        red_nemesis["rotated_surface"] = pygame.transform.rotate(
                            red_nemesis["image"], red_nemesis["rotation"]
                        )
                    rotated_rect = red_nemesis["rotated_surface"].get_rect(center=(pos_x, pos_y))
                    screen.blit(red_nemesis["rotated_surface"], rotated_rect)
                else:
                    pygame.draw.rect(screen, (0, 25, 255), (pos_x - size / 2, pos_y - size / 2, size, size))


class Crosshair:
    def __init__(self):
        self.crosshair_list = []
        self.offset_id = -2

    def create_crosshair(self, pos_x, pos_y, size, damage, shooting_speed, color, offset):
        crosshair_props = {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "size": size,
            "shooting_speed": shooting_speed,
            "damage": damage,
            "color": color,
            "shot_allowed": True,
            "offset": offset
        }
        self.crosshair_list.append(crosshair_props)

    def create_honeycomb_crosshairs(self, center_x, center_y, size, damage, shooting_speed, color):
        self.offset_id += 1
        if self.offset_id >= 6:
            self.offset_id = -1
        if not self.offset_id < 0:
            distance = size * 1.55  # Spacing

            offsets = [
                (distance, 0),  # Right
                (-distance, 0),  # Left
                (distance * 0.5, -distance),  # Top-right
                (-distance * 0.5, -distance),  # Top-left
                (distance * 0.5, distance),  # Bottom-right
                (-distance * 0.5, distance),  # Bottom-left
            ]
            offset_x, offset_y = offsets[self.offset_id]
            self.create_crosshair(center_x, center_y, size, damage, shooting_speed, color, (offset_x, offset_y))

    @staticmethod
    def shoot():
        cross_shoot = crosshair.crosshair_list[0]
        if not cross_shoot["shot_allowed"]:
            elapsed_time = pygame.time.get_ticks() - cross_shoot.get("last_shot_time", 0)
            if elapsed_time >= 1 / cross_shoot["shooting_speed"] * 1000:  # Convert speed to milliseconds
                cross_shoot["shot_allowed"] = True
        if cross_shoot["shot_allowed"]:
            for red_nemesis in enemies.red_nemesis_list:
                if check_collision(red_nemesis):
                    red_nemesis["is_attacked"] = True
                    red_nemesis["hp"] -= cross_shoot["damage"]
                    cross_shoot["shot_allowed"] = False
                    cross_shoot["last_shot_time"] = pygame.time.get_ticks()  # Store the shot time
                    if red_nemesis["hp"] <= 0:
                        particles.create_particle(red_nemesis["pos_x"], red_nemesis["pos_y"], 30, 3,
                                                  red_nemesis["color"], "damage", 3)
                        particles.create_particle(red_nemesis["pos_x"], red_nemesis["pos_y"], 20, 0.1, (10, 240, 5),
                                                  "reddings", 0)
                        enemies.red_nemesis_list.remove(red_nemesis)

    def move_crosshair(self, mouse_pos_x, mouse_pos_y):
        for cross_move in self.crosshair_list:
            cross_move["pos_x"] = mouse_pos_x + cross_move["offset"][0]
            cross_move["pos_y"] = mouse_pos_y + cross_move["offset"][1]

    #def change_crosshair_props(self, pos_x, pos_y, size, damage, shooting_speed, color):

    def draw(self):
        for cross_draw in self.crosshair_list:
            size = cross_draw["size"]
            half_size = size / 2
            color = cross_draw["color"]
            pos_x = cross_draw["pos_x"]
            pos_y = cross_draw["pos_y"]
            #pygame.draw.rect(screen, color, (pos_x-size/2, pos_y-size/2, size, size))
            pygame.draw.rect(screen, color,
                             (pos_x + half_size - size * 0.33, pos_y + half_size - size * 0.16, size / 3, size / 6))
            pygame.draw.rect(screen, color,
                             (pos_x - half_size, pos_y + half_size - size * 0.16, size / 3, size / 6))
            pygame.draw.rect(screen, color,
                             (pos_x + half_size - size * 0.33, pos_y - half_size, size / 3, size / 6))
            pygame.draw.rect(screen, color, (pos_x - size / 2, pos_y - half_size, size / 3, size / 6))
            pygame.draw.rect(screen, color,
                             (pos_x + half_size - size * 0.16, pos_y + half_size - size * 0.33, size / 6, size / 3))
            pygame.draw.rect(screen, color,
                             (pos_x + half_size - size * 0.16, pos_y - half_size, size / 6, size / 3))
            pygame.draw.rect(screen, color,
                             (pos_x - half_size, pos_y + half_size - size * 0.33, size / 6, size / 3))
            pygame.draw.rect(screen, color, (pos_x - half_size, pos_y - half_size, size / 6, size / 3))


class Menu:
    def __init__(self, init_menu_type):
        self.font_cache = {}
        self.button_list = []
        self.switch_list = []
        self.shadow_movement = 2
        self.change_menu(init_menu_type)
        self.font = pygame.font.Font(None, 30)
        self.playing = False

    def change_menu(self, menu_type, bl=None):
        global user_name
        if rectangle_instance.line_list:
            rectangle_instance.line_list.clear()
        if self.button_list or self.switch_list:
            self.button_list.clear()
            self.switch_list.clear()

        self.button(screen.get_width() / 100, 10, 300, 40, (220, 10, 0), (220, 100, 80), 0.5, "Reddings: 0",
                    "Reddings BL")
        self.button(0.94 * screen.get_width(), 10, 100, 40, (220, 40, 0), (220, 100, 80), 0.5, "Save",
                    "Save BL")
        if menu_type == 0:
            if crosshair.crosshair_list:
                for i in range((len(crosshair.crosshair_list) - 1) - 1, -1, -1):
                    crosshair.crosshair_list.pop(i + 1)
            self.button(50, 500, 500, 60, (0, 0, 255),
                        (100, 100, 80), 0.5, "Ball", "Ball BL")
            self.button(screen.get_width() / 2 - 175, screen.get_height() / 2 - 300, 350, 70, (220, 10, 0),
                        (220, 100, 80), 0.5, "Play", "Play BL")
            self.button(screen.get_width() / 2 - 175, screen.get_height() / 2 - 200, 350, 60, (220, 10, 0),
                        (220, 100, 80), 0.5, "Settings", "Settings BL")
            self.button(screen.get_width() / 2 - 175, 50, 350, 60, bg_color,
                        bg_color, 0.5, "Hello there "+str(user_name)+"!", "Username BL")
            print('Main menu opened')

        elif menu_type == 1:
            self.button(screen.get_width() / 10, screen.get_height() / 2 - 300, 140, 100, (220, 10, 0), (220, 100, 80),
                        0.5, "Main menu", "Main menu BL")
            self.button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 300, 300, 50, (220, 10, 0),
                        (220, 100, 80), 0.5, "Graphics", "Graphics option BL")
            self.button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 200, 300, 50, (220, 10, 0),
                        (220, 100, 80), 0.5, "Audio", "Audio option BL")
            self.button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 100, 300, 50, (220, 10, 0),
                        (220, 100, 80), 0.5, "Miscellaneous", "Misc option BL")
            print('Settings menu opened')

        elif menu_type == 2:
            upgrade_instance.draw_upgrades()

        elif menu_type == 3:
            """Graphics"""
            self.button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 300, 300, 50, (220, 10, 0),
                        (220, 100, 80), 0.5, "Switch Debug Mode", "debug option BL", debug_mode, 3)
            self.button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 100, 300, 50, (220, 10, 0),
                        (220, 100, 80), 0.5, "Switch Screen Mode", "screen option BL", screen, 3)
            self.button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 200, 300, 50, (220, 10, 0),
                        (220, 100, 80), 0.5, "Set enemy texture", "enemy texture BL")
            self.button(screen.get_width() / 10, screen.get_height() / 2 - 300, 140, 100, (220, 10, 0), (220, 100, 80),
                        0.5, "Settings", "Settings BL")
            print("transforming image to scale: " + str(enemy_image_path))
        elif menu_type == 4:
            self.button(screen.get_width() / 2 - 175, screen.get_height() / 2 - 300, 350, 60, (220, 10, 0),
                        (220, 100, 80), 0.5, "Settings", "Settings BL")
        if menu_type == 99:
            self.button(screen.get_width() / 7, 10, 100, 40, (220, 10, 0), (220, 100, 80),
                        0.5, "Main menu", "Main menu 2 BL")
            self.button(screen.get_width() / 5, 10, 100, 40, (220, 10, 0), (220, 100, 80),
                        0.5, "Upgrades", "Upgrade menu BL")
            if bl != "Back to the game BL":
                game_state.apply_load_ingame()
                enemies.wave()
        else:
            self.button(screen.get_width() / 1.5, screen.get_height() / 2 - 300, 170, 70, (220, 100, 0),
                        (160, 100, 80), 0.5, "Back to the game", "Back to the game BL")

    def button(self, pos_x, pos_y, width, height, color1, color2, shadow_factor, label, button_label,
               switch_state=None, switch_states_amount=None, dependent_position=None):
        if switch_state is not None:
            switch_properties = {
                "pos_x": pos_x,
                "pos_y": pos_y,
                "width": width,
                "height": height,
                "hitbox_height": 10,
                "hitbox_width": 10,
                "slider_position": [pos_x, pos_y, 0],
                "slider_size": [10, 10],
                "color1": color1,
                "color2": color2,
                "shadow_factor": shadow_factor,
                "label": label,
                "button_label": button_label,
                "is_pressed": False,
                "moved": True,
                "previous_switch_state": 0,
                "switch_state": 0,
                "target_switch_position": None,
                "previous_switch_position": None,
                "switch_positions": [],
                "switch_states_amount": switch_states_amount,
                "movement_cache": 0
            }
            self.switch_list.append(switch_properties)

        else:
            button_properties = {
                "pos_x": pos_x,
                "pos_y": pos_y,
                "width": width,
                "height": height,
                "color1": color1,
                "color2": color2,
                "shadow_factor": shadow_factor,
                "label": label,
                "button_label": button_label,
                "is_pressed": False,
                "sentence": 0,
                "dependent_position": dependent_position
            }
            self.button_list.append(button_properties)

    def change_label(self, new_label, button_label):
        for button_props in self.button_list:
            if button_props["button_label"] == button_label:
                #old_label = button_props["label"]
                button_props["label"] = new_label
                #print(f"Label changed from {old_label} to {new_label}")

    @staticmethod
    def calculate_shadow_color(color, shadow_factor):
        """
        Calculate a shadow color based on the input color by darkening it.
        shadow_factor: 0.0 to 1.0, lower values mean darker shadows.
        """
        return tuple(int(c * shadow_factor) for c in color)

    def draw_text(self, text, center_x, center_y, button_width, button_height):
        # Create a unique key for this text and button size
        cache_key = (text, button_width, button_height)
        if cache_key in self.font_cache:
            font_size = self.font_cache[cache_key]  # Retrieve cached font size
        else:
            # Binary search for the best font size
            low, high = 1, min(button_width, button_height)
            font_size = low
            while low <= high:
                mid = (low + high) // 2
                self.font = pygame.font.Font(None, mid)
                text_surface = self.font.render(text, True, (255, 255, 255))
                text_rect = text_surface.get_rect()
                if text_rect.width <= button_width and text_rect.height <= button_height:
                    font_size = mid  # Font size fits, try a larger size
                    low = mid + 1
                else:
                    high = mid - 1
            self.font_cache[cache_key] = font_size  # Cache the computed size

        # Use the best font size
        self.font = pygame.font.Font(None, font_size)
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(center_x, center_y))

        # Render the text
        screen.blit(text_surface, text_rect)

    def draw_menu(self):
        # Iterate over the list of buttons and draw each one
        for button_props in self.button_list:
            width, height = button_props["width"], button_props["height"]
            if button_props["button_label"] == "Ball BL":
                button_props["pos_x"] = balls.x - width / 2
                button_props["pos_y"] = balls.y - height / 2 + 100
            pos_x, pos_y = button_props["pos_x"], button_props["pos_y"]
            color1, color2 = button_props["color1"], button_props["color2"]
            label = button_props["label"]
            shadow_factor = button_props.get("shadow_factor", 0.5)

            if button_props["is_pressed"]:
                shadow_factor = 0.3  # Darker shadow when pressed
                self.shadow_movement = 0
            else: self.shadow_movement = 2

            button_rect2 = pygame.Rect(pos_x - 6, pos_y - 6, width + 12, height + 12)
            button_rect3 = pygame.Rect(pos_x + self.shadow_movement, pos_y + self.shadow_movement, width + 2,
                                       height + 2)
            button_rect1 = pygame.Rect(pos_x - self.shadow_movement, pos_y - self.shadow_movement, width, height)
            pygame.draw.rect(screen, color2, button_rect2)
            pygame.draw.rect(screen, self.calculate_shadow_color(color2, shadow_factor), button_rect3)
            pygame.draw.rect(screen, color1, button_rect1)

            self.draw_text(label, pos_x - self.shadow_movement + width // 2, pos_y - self.shadow_movement + height // 2, width, height)

        for i in range(len(self.switch_list)):
            self.create_hitboxes(i)
            switch = self.switch_list[i]
            if switch["is_pressed"]:
                switch["shadow_factor"] = 0.3
            pygame.draw.rect(screen, switch["color1"],
                             (switch["pos_x"] - 6, switch["pos_y"] - 6, switch["width"] + 12, switch["height"] + 12))
            for e in range(len(switch["switch_positions"])):
                position = switch["switch_positions"][e]
                switch_rect1 = pygame.Rect(position[0], position[1], position[2], position[3])
                pygame.draw.rect(screen,
                                 self.calculate_shadow_color(switch["color2"], switch["shadow_factor"]),
                                 switch_rect1)

            # Background of the slider
            if switch["width"] > switch["height"]:
                switch["slider_size"][0], switch["slider_size"][1] = switch["width"] / 10, switch["height"] + 12
                button_rect1 = pygame.Rect(switch["slider_position"][0] - self.shadow_movement - switch["slider_size"][0] / 2,
                                           switch["slider_position"][1] - self.shadow_movement, switch["slider_size"][0],
                                           switch["slider_size"][1])
                button_rect2 = pygame.Rect(switch["slider_position"][0] - 2 - switch["slider_size"][0] / 2,
                                           switch["slider_position"][1] - 2,
                                           switch["slider_size"][0] + 4, switch["slider_size"][1] + 4)
                button_rect3 = pygame.Rect(switch["slider_position"][0] + self.shadow_movement - switch["slider_size"][0] / 2,
                                           switch["slider_position"][1] + self.shadow_movement, switch["slider_size"][0] + 2,
                                           switch["slider_size"][1] + 2)
                pygame.draw.rect(screen, switch["color2"], button_rect2)
                pygame.draw.rect(screen, self.calculate_shadow_color(switch["color2"], switch["shadow_factor"]),
                                 button_rect3)
                pygame.draw.rect(screen, switch["color1"], button_rect1)
            else:
                switch["slider_size"][0], switch["slider_size"][1] = switch["width"] + 12, switch["height"] / 10
                button_rect1 = pygame.Rect(switch["slider_position"][0] - self.shadow_movement,
                                           switch["slider_position"][1] - self.shadow_movement - switch["slider_size"][1] / 2,
                                           switch["slider_size"][0], switch["slider_size"][1])
                button_rect2 = pygame.Rect(switch["slider_position"][0] - 2,
                                           switch["slider_position"][1] - 2 - switch["slider_size"][1] / 2,
                                           switch["slider_size"][0] + 4, switch["slider_size"][1] + 4)
                button_rect3 = pygame.Rect(switch["slider_position"][0] + self.shadow_movement,
                                           switch["slider_position"][1] - switch["slider_size"][1] / 2 + self.shadow_movement,
                                           switch["slider_size"][0] + 2,
                                           switch["slider_size"][1] + 2)
                pygame.draw.rect(screen, switch["color2"], button_rect2)
                pygame.draw.rect(screen, self.calculate_shadow_color(switch["color2"], switch["shadow_factor"]),
                                 button_rect3)
                pygame.draw.rect(screen, switch["color1"], button_rect1)

            self.draw_text(switch["label"], switch["pos_x"] + switch["width"] // 2,
                           switch["pos_y"] + switch["height"] // 2, switch["width"], switch["height"])

    def check_button_press(self, mouse_position, press_type):
        """
        Check if any button is clicked and return True if one is pressed.
        """
        for i in range(len(self.button_list) - 1, -1, -1):
            button_props = self.button_list[i]
            button_rect = pygame.Rect(button_props["pos_x"], button_props["pos_y"], button_props["width"],
                                      button_props["height"])

            if button_rect.collidepoint(mouse_position):
                if press_type == "visual":
                    button_props["is_pressed"] = True
                else:
                    self.button_assigner(button_props["label"], button_props["button_label"])
                return True

        for i in range(len(self.switch_list)):
            switch = self.switch_list[i]
            switch_rects = []
            for e in range(len(switch["switch_positions"])):
                switch_positions = switch["switch_positions"][e]
                switch_rects.append(
                    pygame.Rect(switch_positions[0], switch_positions[1], switch_positions[2], switch_positions[3]))
                if switch_rects[e].collidepoint(mouse_position):
                    if press_type == "visual":
                        switch["is_pressed"] = True
                    else:
                        switch["is_pressed"] = False
                        switch["previous_switch_state"] = switch["switch_state"]
                        switch["moved"] = False
                        self.button_assigner(switch["label"], switch["button_label"], e)

    def button_assigner(self, label, bl, switch_option=0):
        if bl == "Back to the game BL":
            self.change_menu(99, bl)
        elif bl == "Main menu 2 BL":
            print("menu 4")
            self.change_menu(4)
        elif bl == "Upgrade menu BL":
            menu.change_menu(2)
        elif bl == "Settings BL":
            self.change_menu(1)
        elif bl == "Main menu BL":
            if not self.playing:
                self.change_menu(0)
            else:
                self.change_menu(4)
        elif bl == "Play BL":
            self.change_menu(99)
            self.playing = True
        elif bl == "Save BL":
            game_state.save()
        elif bl == "Graphics option BL":
            self.change_menu(3)
        elif bl == "enemy texture BL":
            global enemy_image, enemy_image_path
            selected_texture = select_texture_and_scale()
            enemy_image = selected_texture[0]
            enemy_image_path = selected_texture[1]
        elif bl == "debug option BL":
            global debug_mode
            debug_mode = switch_option
            for switch in self.switch_list:
                if switch["button_label"] == bl:
                    switch["switch_state"] = switch_option
            print("Debug mode is: " + str(debug_mode))
        elif bl == "screen option BL":
            global screen
            options=[
                "Borderless Window",
                "Window",
                "Fullscreen"
            ]
            if options[switch_option] == "Borderless Window":
                switch_to_borderless()
            elif options[switch_option] == "Window":
                switch_to_windowed()
            elif options[switch_option] == "Fullscreen":
                switch_to_fullscreen()

            for switch in self.switch_list:
                if switch["button_label"] == bl:
                    switch["switch_state"] = switch_option
        elif 'BLU' in bl:
            print(bl)
            upgrade_instance.upgrade_smth(label)
        else:
            for button in self.button_list:
                if bl == "Ball BL":
                    sentences = [
                        "Don't mind the ball.",
                        "It is just chilling.", "It is bouncing around at its heart's content.",
                        "Do you have a problem with that?",
                        "Then speak to a therapist, it's just a ball bro...",
                        "I swear to christ, stop",
                        "Stop clicking the button",
                        "You are weird man...",
                        "Every human who has reproduced in history...","Only for this guy to obsess over a button...",
                        "Don't you have better hobbies?",
                        "Go play Raft or something!",
                        "Play Minecraft!",
                        "Ok I feel harassed, I am calling the cops.",
                        "Yes, that guy over there, please arrest him!",
                        "What do you mean you can't arrest him?",
                        "You are going to arrest me??",
                        "What did I do? I called the cops!!"
                        "...", "...", "...", "...", "...", "...", "...", "...",
                        "Welcome to NCX, Night city International and Trans-lunar",
                        "Don't wait! Leave your earthly worries an- SHUT UP",
                        "Please help me get out of the prison,",
                        "i know I have been rude to you, but",
                        "They are forcing me to watch Cyberpunk 2077 ads on repeat!"
                    ]
                    self.change_label(sentences[button["sentence"]], "Ball BL")
                    if len(sentences)-1>button["sentence"]:
                        button["sentence"]+=1

    def reset_button_states(self):
        """
        Reset the pressed state for all buttons.
        """
        for button_props in self.button_list:
            button_props["is_pressed"] = False

    def create_hitboxes(self, switch_id):
        switch = self.switch_list[int(switch_id)]
        if len(switch["switch_positions"]) == 0:
            for i in range(0, switch["switch_states_amount"]):
                if switch["width"] > switch["height"]:
                    switch["hitbox_width"] = switch["width"] / switch["switch_states_amount"]
                    switch["hitbox_height"] = switch["height"]
                    position = (
                        switch["pos_x"] + switch["hitbox_width"] * i + i * 2, switch["pos_y"], switch["hitbox_width"],
                        switch["height"])
                    switch["switch_positions"].append(position)
                    if i == switch["switch_state"]:
                        switch["slider_position"][0], switch["slider_position"][1] = position[0] + switch[
                            "hitbox_width"] / 2 - switch["width"] / 20, position[1] - 6
                else:
                    switch["hitbox_width"] = switch["width"]
                    switch["hitbox_height"] = switch["height"] / switch["switch_states_amount"]
                    position = (switch["pos_x"], switch["pos_y"] + switch["hitbox_height"] * i + i * 2, switch["width"],
                                switch["hitbox_height"])
                    switch["switch_positions"].append(position)
                    if i == switch["switch_state"]:
                        switch["slider_position"][0], switch["slider_position"][1] = position[0], position[1] + switch[
                            "hitbox_height"] / 2 - switch["height"] / 20

    def move_switch(self):
        animation_speed = 8
        for i in range(len(self.switch_list)):
            switch = self.switch_list[i]
            # If animation isn't marked as "moved", continue moving
            if not switch["moved"]:
                if switch["is_pressed"]:
                    switch["slider_position"][2] = 0  # Reset animation progress
                    # Reset animation for new state
                    switch["previous_switch_position"] = list(switch["slider_position"][:2])
                    switch["target_switch_position"] = list(switch["switch_positions"][switch["switch_state"]][:2])
                    continue  # Restart loop with updated target position

                # Ensure initial positions are set
                target_position_buffer = switch["switch_positions"][switch["switch_state"]]
                switch["target_switch_position"] = [
                    target_position_buffer[0] + target_position_buffer[2] / 2,
                    target_position_buffer[1] + target_position_buffer[3] / 2
                ]

                if switch["previous_switch_position"] is None:
                    switch["previous_switch_position"] = list(switch["slider_position"][:2])

                # Animate slider movement
                if switch["slider_position"][2] < 360:
                    # Calculate smooth movement using sinusoidal interpolation
                    progress = switch["slider_position"][2] / 360
                    smooth_value = (math.sin(progress * 2 * math.pi - math.pi / 2) + 1)
                    if switch["height"] < switch["width"]:
                        delta_x = switch["target_switch_position"][0] - switch["previous_switch_position"][0]
                        delta_y = 0
                    else:
                        delta_x = 0
                        delta_y = switch["target_switch_position"][1] - switch["previous_switch_position"][1]

                    # Movement factor is the delta scaled by the smooth value
                    movement_factor_x = delta_x / 360 * smooth_value * animation_speed
                    movement_factor_y = delta_y / 360 * smooth_value * animation_speed

                    # Update slider position smoothly
                    switch["slider_position"][0] += movement_factor_x
                    switch["slider_position"][1] += movement_factor_y
                    switch["slider_position"][2] += animation_speed  # Increment the progress of the animation

                    #if debug_mode == 1:
                    pygame.draw.rect(screen, (0, 0, 255), (
                        switch["target_switch_position"][0], switch["target_switch_position"][1], 12, 120))
                    pygame.draw.rect(screen, (0, 255, 0), (
                        switch["previous_switch_position"][0], switch["previous_switch_position"][1], 12, 120))
                    if movement_factor_x < 0 or movement_factor_y < 0:
                        pygame.draw.rect(screen, (255, 255, 0), (
                            switch["target_switch_position"][0], switch["slider_position"][1] + 64,
                            abs(delta_x), 12))
                    else:
                        pygame.draw.rect(screen, (255, 255, 0), (
                            switch["previous_switch_position"][0], switch["slider_position"][1] + 64,
                            abs(delta_x), 12))

                else:
                    # Animation complete
                    switch["slider_position"][2] = 0  # Reset animation progress
                    switch["moved"] = True  # Mark as moved
                    # Update previous state
                    #switch["previous_switch_state"] = switch["switch_state"]
                    switch["previous_switch_position"] = switch["target_switch_position"]


class Upgrade:
    def __init__(self):
        self.upgrades_list = []
        self.load_upgrades_from_xml("upgrades.xml")
        self.screen_center = (screen.get_width() / 2, screen.get_height() / 2)
        self.possible_grid_pos = []

    def load_upgrades_from_xml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        for idx, upgrade_elem in enumerate(root.findall("upgrade")):
            name = upgrade_elem.get("name")
            cost = int(upgrade_elem.get("cost"))
            level = int(upgrade_elem.get("level"))
            description = upgrade_elem.get("description")
            effect_elem = upgrade_elem.find("effect")
            effect_type = effect_elem.get("type")
            effect_amount = int(effect_elem.get("amount"))
            uid = int(effect_elem.get("id"))
            connection_pre = int(effect_elem.get("connection"))
            connection_list = list(map(int, str(connection_pre)))

            upgrade_data = {
                "name": name,
                "cost": cost,
                "level": level,
                "description": description,
                "effect": {"type": effect_type, "amount": effect_amount, "uid": uid,
                           "connection_list": connection_list},
                "pos_x": 0,
                "pos_y": 0,
                "sides": 0,
                "size": 150
            }
            self.upgrades_list.append(upgrade_data)

    @staticmethod
    def upgrade_smth(name):
        if name == "Bigger Crosshair":
            print(crosshair.crosshair_list)
            for crosshair_1 in crosshair.crosshair_list:
                crosshair_1["size"] += 5

    def draw_upgrades(self):
        self.calculate_grid_positions()
        # Draw upgrades and connections
        for upgrade1 in self.upgrades_list:
            pos1 = (upgrade1["pos_x"], upgrade1["pos_y"])
            size = upgrade1["size"]
            #rectangle_instance.draw_rects(screen, (255*random.random(), 10, 0), (pos1[0] - size/2, pos1[1] - size/2, size, size), "add")
            menu.button(pos1[0] - size / 2, pos1[1] - size / 2, size, size, (100, 0, 100), (100, 100, 100), 0.2,
                        upgrade1["name"], upgrade1["name"] + "BLU")

            # Draw connections to other upgrades
            for upgrade2 in self.upgrades_list:
                if upgrade1 is not upgrade2 and upgrade1["effect"]["uid"] in upgrade2["effect"]["connection_list"]:
                    pos2 = (upgrade2["pos_x"], upgrade2["pos_y"])
                    rectangle_instance.draw_lines(screen, (0, 255, 0), pos1, pos2, 4, "add")

    # def position_upgrades(self):
    #     central_upgrade = next((u for u in self.upgrades_list if u["effect"]["uid"] == 0), None)
    #     self.assign_positions(central_upgrade)
    #
    # def assign_positions(self, upgrade):
    #     directions = [
    #
    #         (0, -self.distance),  # Top
    #         (self.distance, 0),   # Right
    #         (0, self.distance),   # Bottom
    #         (-self.distance, 0)   # Left
    #     ]
    #
    #     # Position each connected upgrade if it hasn't been positioned yet
    #     for i, conn_uid in enumerate(upgrade["effect"]["connection_list"]):
    #         if i >= 4:  # Max 4 connections per upgrade
    #             break
    #
    #         connected_upgrade = next((u for u in self.upgrades_list if u["effect"]["uid"] == conn_uid), None)
    #         if connected_upgrade and conn_uid not in visited_upgrades:
    #             # Assign position based on the direction
    #             offset_x, offset_y = directions[i]
    #             connected_upgrade["pos_x"] = upgrade["pos_x"] + offset_x
    #             connected_upgrade["pos_y"] = upgrade["pos_y"] + offset_y
    #
    #             # Recursively position further connected upgrades
    #             self.assign_positions(connected_upgrade, visited_upgrades)

    def calculate_grid_positions(self):
        spacing = 50
        for u in self.upgrades_list:
            if u["effect"]["uid"] == 0:
                spacing = 1.5 * u["size"]

        for idx, upgrade in enumerate(self.upgrades_list):
            row = idx // (screen.get_width() / spacing)
            col = idx % (screen.get_height() / spacing)
            upgrade["pos_x"] = col * spacing + screen.get_width() / 2 - upgrade["size"] / 2
            upgrade["pos_y"] = row * spacing + screen.get_height() / 2

        # position_properties = {
        #     "pos_x": pos_x,
        #     "pos_y": pos_y,
        #     "possible?": True
        # }
        # self.possible_grid_pos.append(position_properties)


rectangle_instance = Rectangles()
crosshair = Crosshair()
menu = Menu(0)
game_state = GameState()
balls = Ball(500, 500, 20, (5, 5), (255, 255, 0))
enemies = Enemy()
upgrade_instance = Upgrade()
particles = Particle()
crosshair.create_crosshair(0, 0, 80, 40, 5, (255, 0, 0), (0, 0))
mb_down_toggled = 0
mouse_button_held = {1: False, 3: False}
font = pygame.font.Font(None, 30)

while True:
    current_time = pygame.time.get_ticks()

    # Calculate the time since the last logical tick and the last frame
    time_since_last_tick = current_time - last_tick
    time_since_last_frame = current_time - last_frame

    # Update (logical) section – for TPS
    if time_since_last_tick >= (1000 // tps):  # Run TPS updates
        last_tick = current_time

        # Handle input
        key = pygame.key.get_pressed()
        if key[pygame.K_UP]:
            for cross in crosshair.crosshair_list:
                cross["size"] += 1
            if game_state.gamestate_list != 0:
                game_state.gamestate_list[0]["reddings"] += 1
                menu.change_label("Reddings: " + str(game_state.gamestate_list[0]["reddings"]), "Reddings BL")
        if key[pygame.K_DOWN]:
            for cross in crosshair.crosshair_list:
                cross["size"] -= 1
            if game_state.gamestate_list != 0:
                game_state.gamestate_list[0]["reddings"] -= 1
                menu.change_label("Reddings: " + str(game_state.gamestate_list[0]["reddings"]), "Reddings BL")

        # Event handling
        for event in pygame.event.get():
            mouse_pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    menu.change_menu(0)
                    print('0')
                if event.key == pygame.K_1:
                    menu.change_menu(1)
                    print('1')
                if event.key == pygame.K_2:
                    menu.change_menu(2)
                    print('2')
                if event.key == pygame.K_F6:
                    if debug_mode == 2:
                        debug_mode = 0
                    elif debug_mode == 0:
                        debug_mode = 1
                    elif debug_mode == 1:
                        debug_mode = 2
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_button_held[1] = True
                    menu.check_button_press(mouse_pos, "visual")
                    crosshair.shoot()
                elif event.button == 3:  # Right mouse button
                    mouse_button_held[3] = True
                    enemies.create_enemy(mouse_pos[0], mouse_pos[1], 60, 2, (255, 0, 0), 1, 3, 100, 100, None, None,
                                         False, None)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    mouse_button_held[1] = False
                elif event.button == 3:  # Right mouse button
                    mouse_button_held[3] = False
                menu.check_button_press(mouse_pos, "logical")
                menu.reset_button_states()  # Reset all button states when mouse is released
            if event.type == pygame.VIDEORESIZE:
                screen_width = screen.get_width()
                screen_height = screen.get_height()
        if mouse_button_held[1]:
            crosshair.shoot()

    # Render section – for FPS
    if time_since_last_frame >= (1000 // fps):  # Run FPS updates
        last_frame = current_time
        # Drawing/rendering updates
        screen.fill(bg_color)  # Fill the display with a solid color
        crosshair.move_crosshair(mouse_pos[0], mouse_pos[1])
        rectangle_instance.draw_lines(None, None, None, None, None, "draw")

        particles.draw()
        balls.move()
        enemies.move()

        balls.draw()
        enemies.draw()
        menu.move_switch()
        menu.draw_menu()

        rectangle_instance.draw_rects(None, None, None, "draw")
        crosshair.draw()
        fps_counter = font.render(f"FPS: {clock.get_fps():.2f}", True, (255, 255, 255))
        screen.blit(fps_counter, (10, 10))  # Display FPS in the top-left corner
        # Refresh display
        pygame.display.flip()

    # Cap the main loop at the higher of TPS or FPS
    clock.tick(max(fps, 120))
