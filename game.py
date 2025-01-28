import ctypes
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from xml.dom import minidom
from collections import defaultdict
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import BB

pygame.init()

info = pygame.display.Info()
tv_width, tv_height = info.current_w, info.current_h
screen = pygame.display.set_mode((tv_width, tv_height),
                                 pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SRCALPHA)
draw_options = pymunk.pygame_util.DrawOptions(screen)
# Create Pymunk Space
space = pymunk.Space(True)
no_physics_space = pymunk.Space(True)
space.gravity = (0, 0)
BALL_COLLISION_TYPE = 1
ENEMY_COLLISION_TYPE = 2
# Cache for rotated rectangles
rotation_cache = {}
paused_velocities = {}
paused = 0
timer = False  # binary timer to execute stuff only every second time
sdl = ctypes.CDLL("SDL2.dll")
hwnd = pygame.display.get_wm_info()['window']
user32 = ctypes.windll.user32
screen_width = screen.get_width()
screen_height = screen.get_height()
clock = pygame.time.Clock()
cell_size = 64  # Size of each spatial partitioning cell
grid = defaultdict(list)  # Grid for spatial partitioning
reddings_amount = 0
rand = random.Random()
user_name = "standard"
mouse_pos = (0, 0)
debug_mode = 3
collision_mode = 0
playing = False
enemy_image_path = None
enemy_image = screen
target_tps = 120
target_tick_time = 1 / target_tps
target_fps = 120
target_frame_time = 1 / target_fps
no_sleep_frame_time = 0.05
frame_time = 0.05
frame_start_time = time.perf_counter()
smoothed_fps = 240
smoothed_lrps = smoothed_fps
frame_counter = 0
bg_color = (8, 0, 0)


def null_window_position(x=0, y=0):
    user32.MoveWindow(hwnd, x, y, tv_width, tv_height)


def switch_to_borderless():
    global screen
    # Do it twice because of a stupid black bar bug on the bottom...
    for _ in range(2):
        screen = pygame.display.set_mode((tv_width, tv_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)
        user32.ShowWindow(hwnd, 1)  # Show window normally (not minimized)
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


def damage_enemy(red_nemesis, damage=1, cross_shoot=None):
    red_nemesis["is_attacked"] = True
    if cross_shoot is not None:
        red_nemesis["hp"] -= cross_shoot["damage"]
        if red_nemesis["hp"] < 0:
            red_nemesis["lost_hp"] = cross_shoot["damage"] + red_nemesis["hp"]
        else:
            red_nemesis["lost_hp"] = cross_shoot["damage"]
        cross_shoot["enemy_hit"] = True
        cross_shoot["last_shot_time"] = pygame.time.get_ticks()  # Store the shot time
    else:
        red_nemesis["hp"] -= damage
        if red_nemesis["hp"] < 0:
            red_nemesis["lost_hp"] = damage + red_nemesis["hp"]
        else:
            red_nemesis["lost_hp"] = damage

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


def precompute_unit_rotations():
    """
    Precompute the vertices of a unit square rotated for all 360 degrees.
    """
    unit_half_size = 0.5  # Half the size of a unit square
    corners = [
        (-unit_half_size, -unit_half_size),
        (unit_half_size, -unit_half_size),
        (unit_half_size, unit_half_size),
        (-unit_half_size, unit_half_size),
    ]

    for degree in range(720):
        angle = math.radians(degree / 2)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        rotated_corners = []
        for corner_x, corner_y in corners:
            rotated_x = cos_angle * corner_x - sin_angle * corner_y
            rotated_y = sin_angle * corner_x + cos_angle * corner_y
            rotated_corners.append((rotated_x, rotated_y))

        rotation_cache[degree] = rotated_corners


def get_rotated_vertices(top_left_x, top_left_y, size, angle):
    """
    Returns the vertices of a rotated square by applying the size and top-left
    position to the precomputed unit square rotation. 720 degrees available (0.5 degree precision)
    """
    relative_vertices = rotation_cache[int(round(angle * 2)) % 720]

    # Calculate the top-left position offset and apply the size to vertices, finally return vertices

    return [
        (top_left_x + x * size, top_left_y + y * size)
        for x, y in relative_vertices
    ]


def check_collision(shape1, shape2):
    """
    Checks if two Pymunk shapes collide.

    :param shape1: A pymunk.Shape (e.g., Poly, Circle).
    :param shape2: A pymunk.Shape (e.g., Poly, Circle).
    :return: True if the shapes collide, False otherwise.
    """

    # You could add a check for bounding box intersection first, as it's cheaper than checking full collisions
    if shape1.bb.intersects(shape2.bb):
        # Use the built-in collision query
        collision_info = shape1.shapes_collide(shape2)
        # Check if there is any collision
        return collision_info.points != []
    return False


def check_aabb_to_point_collision(obj_shape, point, point_radius=0):
    """
    Check if a given point collides with an object shape (AABB or any Pymunk shape).

    Args:
        obj_shape (pymunk.Shape): The shape (AABB or any Pymunk shape) to check collision with.
        point (tuple): The (x, y) coordinates of the point to check.
        point_radius (float): The radius around the point to consider for collision.

    Returns:
        bool: True if the point collides with the shape, False otherwise.
    """
    point = (float(point[0]), float(point[1]))
    result = obj_shape.point_query(point)
    if result.distance <= point_radius:
        return True  # Collision detected

    return False  # No collision


def create_bb(position, size):
    return BB(position[0], position[1], position[0] + size[0], position[1] + size[1])


def draw_ball_debug_info(a_ball):
    pygame.draw.circle(screen, (255, 0, 0), (int(a_ball.x), int(a_ball.y)), a_ball.radius, 1)


def draw_enemy_debug_info():
    for enemy in enemies.red_nemesis_list:
        # Draw the square's collision boundaries
        pygame.draw.polygon(screen, (0, 255, 0), enemy["vertices"], 1)


def ball_hits_enemy(arbiter, _shape, _data):
    # Get the shapes involved in the collision
    ball_shape, enemy_shape = arbiter.shapes

    # Find the enemy in your list
    for enemy in enemies.red_nemesis_list:
        if enemy["shape"] == enemy_shape:
            damage_enemy(enemy, 13)
            return True
    return False


class GameState:
    def __init__(self):
        # Use a fixed filename in the current working directory
        # Get the user's Documents folder
        user_folder = os.path.expanduser("~")
        documents_folder = os.path.join(user_folder, "Documents")
        global user_name
        user_name = os.path.basename(user_folder)
        menu.change_label("Hello there " + str(user_name) + "!", "Username BL")
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
        self.precomputed_particles = {}

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

    def precompute_particles(self, particle_type, size, color):
        """
        Precomputes rotated particle surfaces for all 360 degrees of rotation.
        """
        self.precomputed_particles[particle_type] = {}

        for angle in range(360):
            # Create a transparent surface for the particle
            particle_surface = pygame.Surface((size, size), pygame.SRCALPHA)

            # Define the stretched diamond shape
            diamond = [
                (size // 2, 0),  # Top point
                (3 * size // 4, size // 2),  # Right point (closer to center horizontally)
                (size // 2, size - 1),  # Bottom point
                (size // 4, size // 2)  # Left point (closer to center horizontally)
            ]
            pygame.draw.polygon(particle_surface, color, diamond)
            pygame.draw.polygon(particle_surface, (255, 255, 0), diamond, width=2)

            # Add shine to the particle
            base_color = color
            brightness_factor = int(255 * 0.3)  # Static shine factor
            shine_color = (
                min(255, base_color[0] + brightness_factor),
                min(255, base_color[1] + brightness_factor),
                min(255, base_color[2] + brightness_factor),
                255  # Semi-transparent alpha
            )
            shine_triangle = [
                (size // 2, size // 4),  # Near the top-center
                (3 * size // 5, size // 2),  # Right point
                (size // 2, 3 * size // 4)  # Near the bottom-center
            ]
            pygame.draw.polygon(particle_surface, shine_color, shine_triangle)

            # Rotate the particle surface
            rotated_surface = pygame.transform.rotate(particle_surface, angle)
            self.precomputed_particles[particle_type][angle] = rotated_surface

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
                distance = get_distance(pos_x, pos_y, cross_x, cross_y) - 10

                # Avoid division by zero for very small distances
                if distance > size:
                    # Gravity-like force: proportional to 1 / (distance ** 2)
                    strength = 200 / (distance ** 2)  # Adjust 10 to control the strength of attraction
                    direction_x = (cross_x - pos_x) / distance
                    direction_y = (cross_y - pos_y) / distance
                    force_x += strength * direction_x
                    force_y += strength * direction_y
                else:
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
                if label not in self.precomputed_particles:
                    try:
                        self.precompute_particles(label, size, particle["color"])
                    except ValueError:
                        raise ValueError(f"Particle type '{label}' is not precomputed.")

                angle = int(particle["rotation"]) % 360
                particle_surface = self.precomputed_particles[label][angle]

                # Scale particle if necessary
                size = (
                     int(particle_surface.get_width() * scaled_width*0.05), int(particle_surface.get_height()))
                particle_surface = pygame.transform.scale(particle_surface, size)

                # Blit the particle
                particle_rect = particle_surface.get_rect(center=(pos_x, pos_y))
                screen.blit(particle_surface, particle_rect)


class Ball:
    def __init__(self, x, y, radius, ball_velocity, color):
        # Create the pymunk body and shape
        self.radius = radius
        self.color = color
        self.x = x
        self.y = y
        self.hp = 100

        mass = 0.01
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, inertia)
        self.body.position = (x, y)
        self.body.velocity = ball_velocity

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 1  # Makes it bouncy
        self.shape.friction = 0
        space.add(self.body, self.shape)
        self.shape.collision_type = 1

    def move(self):
        ball_pos = self.body.position
        ball_vel = self.body.velocity
        # Reflect velocity if outside the boundaries
        if ball_pos.x < 0 or ball_pos.x > screen_width:
            self.body.velocity = (-ball_vel.x, ball_vel.y)  # Reverse X velocity
            self.body.position = (
                max(0, int(min(ball_pos.x, screen_width))), ball_pos.y)  # Clamp position inside bounds

        if ball_pos.y < 0 or ball_pos.y > screen_height:
            self.body.velocity = (ball_vel.x, -ball_vel.y)  # Reverse Y velocity
            self.body.position = (
                ball_pos.x, max(0, int(min(ball_pos.y, screen_height))))  # Clamp position inside bounds

    def draw(self):
        # Draw the ball using pygame
        pygame.draw.circle(screen, self.color, (int(self.body.position.x), int(self.body.position.y)), self.radius)


class Enemy:
    def __init__(self):
        self.red_nemesis_list = []

    def move(self):
        global timer, collision_mode
        max_velocity = 20
        velocity_limit = 10
        screen_limit_x = screen_width * 2
        screen_limit_y = screen_height * 2

        for i in reversed(range(len(self.red_nemesis_list))):  # Iterate in reverse
            red_nemesis = self.red_nemesis_list[i]
            enemy_body = red_nemesis["body"]
            shape = red_nemesis["shape"]
            speed = red_nemesis["speed"]

            # Update position and rotation
            enemy_body.position += enemy_body.velocity
            red_nemesis["pos_x"], red_nemesis["pos_y"] = enemy_body.position

            if not timer:
                red_nemesis["rotation"] += speed / 3 if red_nemesis["rotation"] > 0 else -speed / 3
                red_nemesis["rotation"] %= 360  # Normalize rotation to [0, 360)
                enemy_body.angle = math.radians(red_nemesis["rotation"])
                timer = True
            else:
                timer = False

            # Limit velocity
            vx, vy = enemy_body.velocity
            enemy_body.velocity = (
                max(-velocity_limit, min(vx, velocity_limit)) if abs(vx) > max_velocity else vx,
                max(-velocity_limit, min(vy, velocity_limit)) if abs(vy) > max_velocity else vy,
            )

            # Remove enemies outside screen bounds
            if abs(red_nemesis["pos_x"]) > screen_limit_x or abs(red_nemesis["pos_y"]) > screen_limit_y:
                space.remove(enemy_body, shape)
                self.red_nemesis_list.pop(i)
                continue

            # Handle enemy health
            if red_nemesis["hp"] <= 0:
                space.remove(enemy_body, shape)
                particles.create_particle(
                    red_nemesis["pos_x"], red_nemesis["pos_y"], 30, 3, red_nemesis["color"], "damage", 2
                )
                particles.create_particle(
                    red_nemesis["pos_x"], red_nemesis["pos_y"], 20, 0.1, (10, 240, 5), "reddings", 0
                )
                self.red_nemesis_list.pop(i)
                continue

        # Handle wave logic if enemy count drops below threshold
        if 50 > len(self.red_nemesis_list) > 0:
            collision_mode = 1 - collision_mode  # Toggle collision mode
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

                if collision_mode == 1:
                    red_nemesis["og_color"] = (255, 255, 0)
                    mass = 1
                    inertia = pymunk.moment_for_box(mass, (red_nemesis["size"], red_nemesis["size"]))
                    enemy_body = pymunk.Body(mass, inertia)
                else:
                    red_nemesis["og_color"] = (255, 0, 0)
                    enemy_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                enemy_body.position = (red_nemesis["pos_x"], red_nemesis["pos_y"])
                enemy_body.velocity = (red_nemesis["vel_x"], red_nemesis["vel_y"])
                # Create a rectangular shape for the enemy
                shape = pymunk.Poly.create_box(enemy_body, (red_nemesis["size"], red_nemesis["size"]))
                shape.elasticity = 1
                shape.friction = 0
                space.add(enemy_body, shape)
                red_nemesis["body"] = enemy_body
                red_nemesis["shape"] = shape
                red_nemesis["shape"].collision_type = 2
                red_nemesis["random_props_chosen"] = True

    def wave(self):
        #if self.red_nemesis_list: self.red_nemesis_list.clear()
        enemy_amount = rand.randint(20, 80)
        for _ in range(enemy_amount):
            self.create_enemy(None, None, None, None, (255, 0, 0), (255, 200, 0), "random_guy",
                              30, 100, 100, None, None, False, None)
        #next_round =

    def create_enemy(self, pos_x, pos_y, size, speed, color, edge_color, label, age, hp, max_hp, rotation, direction,
                     is_attacked,
                     weirdness):

        enemy_properties = {
            "body": None,
            "shape": None,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "vel_x": 0,
            "vel_y": 0,
            "size": size,
            "speed": speed,
            "color": color,
            "og_color": color,
            "edge_color": edge_color,
            "label": label,
            "age": age,
            "hp": hp,
            "max_hp": max_hp,
            "lost_hp": 0,
            "prev_lost_hp": 0,
            "animated_lost_hp": 0,
            "rotation": rotation,
            "vertices": [],
            "last_rotation": 0,
            "direction": direction,
            "is_attacked": is_attacked,
            "weirdness": weirdness,
            "random_props_chosen": False,
            "surface": None,
            "image": None,
            "rotated_surface": pygame.Surface((40, 40), pygame.SRCALPHA),
            "previous_framecount": 0
        }

        self.red_nemesis_list.append(enemy_properties)
        self.choose_random_props()

    def draw(self):
        global debug_mode, enemy_image, enemy_image_path
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
            edge_color = red_nemesis["edge_color"]
            if red_nemesis["is_attacked"]:
                hp_factor = hp / max_hp
                pos_x_half_size = pos_x - size / 2
                pos_y_size = pos_y + size
                tenth_size = size * 0.1

                # Draw the full HP bar (red background and green HP bar)
                pygame.draw.rect(screen, (255, 0, 0), (pos_x_half_size, pos_y_size, size, tenth_size))
                pygame.draw.rect(screen, (0, 255, 0), (pos_x_half_size, pos_y_size, size * hp_factor + 1, tenth_size))

                if red_nemesis["lost_hp"] > 0:
                    # Combine current and previous lost HP for seamless animation
                    if red_nemesis["prev_lost_hp"] != red_nemesis["lost_hp"] and red_nemesis["prev_lost_hp"] != 0:
                        # RUDELY INTERRUPTED!!! WHO DARES WAKE THE FLYING DUTCHMAN!!
                        print("RUDELY INTERRUPTED")
                        red_nemesis["lost_hp"] += red_nemesis["prev_lost_hp"]
                        red_nemesis["previous_framecount"] = 0

                    if red_nemesis["previous_framecount"] == 0:
                        red_nemesis["previous_framecount"] = frame_counter

                    # Calculate the total lost size
                    lost_size = size * (red_nemesis["lost_hp"] / 100)
                    elapsed_frames = frame_counter - red_nemesis["previous_framecount"]
                    # Calculate the animation progress (clamped to [0, 1])
                    animation_progress = min(elapsed_frames / 300, 1)
                    # Update the actual lost_hp value gradually
                    initial_lost_hp = red_nemesis["lost_hp"]
                    red_nemesis["lost_hp"] = max(0, initial_lost_hp * (1 - animation_progress))
                    # Calculate the size of the animated loss bar
                    animated_size = size * (red_nemesis["lost_hp"] / 100)
                    # Draw the animated loss bar
                    pygame.draw.rect(screen, (200, 255, 200),
                                     (max(pos_x_half_size, pos_x_half_size + size * hp_factor + 1), pos_y_size, animated_size, tenth_size))
                    print(red_nemesis["lost_hp"], elapsed_frames, animation_progress)
                    # Reset animation if finished
                    if animation_progress >= 1:
                        red_nemesis["previous_framecount"] = 0
                        red_nemesis["lost_hp"] = 0
                    else:
                        # Store the current lost HP for the next frame
                        red_nemesis["prev_lost_hp"] = red_nemesis["lost_hp"]

            if debug_mode == 1:
                pygame.draw.polygon(screen, color, vertices, 3)
            if debug_mode == 0 or debug_mode == 3:
                pygame.draw.polygon(screen, color, vertices)
                pygame.draw.polygon(screen, edge_color, vertices, 3)
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
        crosshair_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        crosshair_body.position = (pos_x + size / 2, pos_y + size / 2)
        shape = pymunk.Poly.create_box(crosshair_body, (size, size))
        no_physics_space.add(crosshair_body, shape)
        #space.add(crosshair_body, shape)
        crosshair_props = {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "size": size,
            "shooting_speed": shooting_speed,
            "damage": damage,
            "color": color,
            "shot_allowed": True,
            "enemy_hit": False,
            "offset": offset,
            "body": crosshair_body,
            "shape": shape,
            "surface": pygame.Surface((screen_width, screen_height), pygame.SRCALPHA),
            "og_size": 0
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

    def shoot(self, visual=False):
        cross_shoot = self.crosshair_list[0]
        if not cross_shoot["shot_allowed"]:
            elapsed_time = pygame.time.get_ticks() - cross_shoot.get("last_shot_time", 0)
            if elapsed_time >= 1 / cross_shoot["shooting_speed"] * 1000:  # Convert speed to milliseconds
                cross_shoot["shot_allowed"] = True
        for red_nemesis in enemies.red_nemesis_list:
            for cross_collision in self.crosshair_list:
                if check_collision(
                        cross_collision["shape"],
                        red_nemesis["shape"]):
                    if visual is False:
                        red_nemesis["color"] = tuple(int(c * 0.5) for c in red_nemesis["og_color"])
                        if cross_shoot["shot_allowed"]:
                            damage_enemy(red_nemesis, 1, cross_shoot)
                    else:
                        red_nemesis["color"] = tuple(int(c * 0.5) for c in red_nemesis["og_color"])
                else:
                    red_nemesis["color"] = red_nemesis["og_color"]
        if cross_shoot["enemy_hit"]:
            cross_shoot["shot_allowed"] = False
            cross_shoot["enemy_hit"] = False

    def move_crosshair(self, mouse_pos_x, mouse_pos_y):
        for cross_move in self.crosshair_list:
            cross_move["pos_x"] = mouse_pos_x + cross_move["offset"][0]
            cross_move["pos_y"] = mouse_pos_y + cross_move["offset"][1]
            cross_move["body"].position = pymunk.Vec2d(cross_move["pos_x"],
                                                       cross_move["pos_y"])
            for shape in cross_move["body"].shapes:
                shape.cache_bb()  # Update the bounding box of the shape to match the new position

    #def change_crosshair_props(self, pos_x, pos_y, size, damage, shooting_speed, color):

    def draw(self):
        for cross_draw in self.crosshair_list:
            size = cross_draw["size"]
            half_size = size / 2
            color = cross_draw["color"]
            pos_x = cross_draw["pos_x"]
            pos_y = cross_draw["pos_y"]
            #pygame.draw.rect(screen, color, (pos_x-size/2, pos_y-size/2, size, size))

            # Pre-calculate sizes to avoid repeating division
            if cross_draw["size"] != cross_draw["og_size"]:
                cross_draw["og_size"] = cross_draw["size"]
                third_size = size * 0.33
                sixth_size = size * 0.16
                pos_x_plus_half_size = size + half_size
                pos_x_minus_half_size = size - half_size
                pos_y_plus_half_size = size + half_size
                pos_y_minus_half_size = size - half_size

                # Create a surface to hold all the rectangles
                cross_draw["surface"] = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

                # Pre-calculate the rectangles once
                rects = [
                    (pos_x_plus_half_size - third_size, pos_y_plus_half_size - sixth_size, third_size, sixth_size),
                    (pos_x_minus_half_size, pos_y_plus_half_size - sixth_size, third_size, sixth_size),
                    (pos_x_plus_half_size - third_size, pos_y_minus_half_size, third_size, sixth_size),
                    (pos_x_minus_half_size, pos_y_minus_half_size, third_size, sixth_size),
                    (pos_x_plus_half_size - sixth_size, pos_y_plus_half_size - third_size, sixth_size, third_size),
                    (pos_x_plus_half_size - sixth_size, pos_y_minus_half_size, sixth_size, third_size),
                    (pos_x_minus_half_size, pos_y_plus_half_size - third_size, sixth_size, third_size),
                    (pos_x_minus_half_size, pos_y_minus_half_size, sixth_size, third_size)
                ]
                # Fill the surface with the rectangles
                for rect in rects:
                    pygame.draw.rect(cross_draw["surface"], color, rect)

            # Blit the surface to the screen
            screen.blit(cross_draw["surface"], (pos_x - size, pos_y - size))


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
        global user_name, debug_mode
        if rectangle_instance.line_list:
            rectangle_instance.line_list.clear()
        if self.button_list or self.switch_list:
            for remove_button in self.button_list:
                no_physics_space.remove(remove_button["body"], remove_button["shape"])
            self.button_list.clear()
            self.switch_list.clear()

        self.create_button(screen.get_width() / 100, 10, 300, 40, (220, 10, 0), (220, 100, 80), 0.5, "Reddings: 0",
                           "Reddings BL")
        self.create_button(0.94 * screen.get_width(), 10, 100, 40, (220, 40, 0), (220, 100, 80), 0.5, "Save",
                           "Save BL")
        if menu_type == 0:
            if crosshair.crosshair_list:
                for i in range((len(crosshair.crosshair_list) - 1) - 1, -1, -1):
                    crosshair.crosshair_list.pop(i + 1)
            self.create_button(50, 500, 500, 60, (0, 0, 255),
                               (100, 100, 80), 0.5, "Ball", "Ball BL")
            self.create_button(screen.get_width() / 2 - 175, screen.get_height() / 2 - 300, 350, 70, (220, 10, 0),
                               (220, 100, 80), 0.5, "Play", "Play BL")
            self.create_button(screen.get_width() / 2 - 175, screen.get_height() / 2 - 200, 350, 60, (220, 10, 0),
                               (220, 100, 80), 0.5, "Settings", "Settings BL")
            self.create_button(screen.get_width() / 2 - 175, 50, 350, 60, bg_color,
                               bg_color, 0.5, "Hello there " + str(user_name) + "!", "Username BL")
            print('Main menu opened')

        elif menu_type == 1:
            self.create_button(screen.get_width() / 10, screen.get_height() / 2 - 300, 140, 100, (220, 10, 0),
                               (220, 100, 80),
                               0.5, "Main menu", "Main menu BL")
            self.create_button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 300, 300, 50, (220, 10, 0),
                               (220, 100, 80), 0.5, "Graphics", "Graphics option BL")
            self.create_button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 200, 300, 50, (220, 10, 0),
                               (220, 100, 80), 0.5, "Audio", "Audio option BL")
            self.create_button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 100, 300, 50, (220, 10, 0),
                               (220, 100, 80), 0.5, "Miscellaneous", "Misc option BL")
            print('Settings menu opened')

        elif menu_type == 2:
            upgrade_instance.draw_upgrades()

        elif menu_type == 3:
            """Graphics"""
            self.create_switch(screen.get_width() / 2 - 150, screen.get_height() / 2 - 300, 300, 50, (220, 10, 0),
                               (220, 100, 80), 0.5, "Switch Debug Mode", "debug option BL", debug_mode, 4)
            self.create_switch(screen.get_width() / 2 - 150, screen.get_height() / 2 - 100, 300, 50, (220, 10, 0),
                               (220, 100, 80), 0.5, "Switch Screen Mode", "screen option BL", screen, 3)
            self.create_button(screen.get_width() / 2 - 150, screen.get_height() / 2 - 200, 300, 50, (220, 10, 0),
                               (220, 100, 80), 0.5, "Set enemy texture", "enemy texture BL")
            self.create_button(screen.get_width() / 10, screen.get_height() / 2 - 300, 140, 100, (220, 10, 0),
                               (220, 100, 80),
                               0.5, "Settings", "Settings BL")
            print("transforming image to scale: " + str(enemy_image_path))
        elif menu_type == 4:
            self.create_button(screen.get_width() / 2 - 175, screen.get_height() / 2 - 300, 350, 60, (220, 10, 0),
                               (220, 100, 80), 0.5, "Settings", "Settings BL")
        elif menu_type == 5:
            """Miscellaneous"""
            self.create_switch(screen.get_width() / 2 - 150, screen.get_height() / 2 - 300, 300, 50, (220, 10, 0),
                               (220, 100, 80), 0.5, "Collision Mode", "collision mode BL", collision_mode, 2)
            self.create_button(screen.get_width() / 10, screen.get_height() / 2 - 300, 140, 100, (220, 10, 0),
                               (220, 100, 80),
                               0.5, "Settings", "Settings BL")
        if menu_type == 99:
            self.create_button(screen.get_width() / 7, 10, 100, 40, (220, 10, 0), (220, 100, 80),
                               0.5, "Main menu", "Main menu 2 BL")
            self.create_button(screen.get_width() / 5, 10, 100, 40, (220, 10, 0), (220, 100, 80),
                               0.5, "Upgrades", "Upgrade menu BL")
            if bl != "Back to the game BL":
                game_state.apply_load_ingame()
                enemies.wave()
        else:
            self.create_button(screen.get_width() / 1.5, screen.get_height() / 2 - 300, 170, 70, (220, 100, 0),
                               (160, 100, 80), 0.5, "Back to the game", "Back to the game BL")

    def create_button(self, pos_x, pos_y, width, height, color1, color2, shadow_factor, label, button_label,
                      dependent_position=None):

        # Create a static body (not affected by physics, just for position)
        button_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        button_body.position = (pos_x + width / 2, pos_y + height / 2)
        shape = pymunk.Poly.create_box(button_body, (width, height))
        no_physics_space.add(button_body, shape)

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
            "dependent_position": dependent_position,
            "previous_rect": (0, 0, 0, 0, 0),
            "button_rects": [],
            "body": button_body,
            "shape": shape
        }
        self.button_list.append(button_properties)

    def create_switch(self, pos_x, pos_y, width, height, color1, color2, shadow_factor, label, button_label,
                      switch_state, switch_states_amount):
        # Create a dynamic body for the switch
        switch_body = pymunk.Body(body_type=pymunk.Body.STATIC)  # Static body since we're not using physics forces

        # Set the initial position of the switch body
        switch_body.position = (pos_x, pos_y)

        # Store hitboxes and switch positions
        hitboxes = []
        switch_positions = []
        slider_position = [pos_x, pos_y, 0]

        # Create hitboxes based on the switch dimensions and states
        hitbox_width = 10
        hitbox_height = 10

        for i in range(switch_states_amount):
            if width > height:
                hitbox_width = width / switch_states_amount
                hitbox_height = height
                position = (
                    pos_x + hitbox_width * i + i * 2, pos_y, hitbox_width, height)
            else:
                hitbox_width = width
                hitbox_height = height / switch_states_amount
                position = (pos_x, pos_y + hitbox_height * i + i * 2, width, hitbox_height)

            # Store the switch position
            switch_positions.append(position)

            # Create Pymunk shapes for each hitbox
            hitbox = pymunk.Poly.create_box(switch_body, (hitbox_width, hitbox_height))
            hitbox.offset = (position[0] - pos_x, position[1] - pos_y)  # Adjust offset based on switch position
            hitboxes.append(hitbox)

            # Set initial slider position
            if i == switch_state:
                slider_position = [position[0] + hitbox_width / 2 - width / 20, position[1] - 6, 0]

        # Define the switch properties
        switch_properties = {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "width": width,
            "height": height,
            "hitbox_height": hitbox_height,
            "hitbox_width": hitbox_width,
            "slider_position": slider_position,
            "slider_size": [10, 10],
            "color1": color1,
            "color2": color2,
            "shadow_factor": shadow_factor,
            "label": label,
            "button_label": button_label,
            "is_pressed": False,
            "moved": True,
            "previous_switch_state": 0,
            "switch_state": switch_state,
            "target_switch_position": None,
            "previous_switch_position": None,
            "switch_positions": switch_positions,
            "switch_states_amount": switch_states_amount,
            "movement_cache": 0,
            "body": switch_body,
            "shapes": hitboxes  # Store the shapes for collision detection
        }

        self.switch_list.append(switch_properties)

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
        """
        Draw text on the screen, with caching to optimize repeated rendering.

        :param text: The string to render.
        :param center_x: X-coordinate for the text's center.
        :param center_y: Y-coordinate for the text's center.
        :param button_width: The width constraint for the text.
        :param button_height: The height constraint for the text.
        """
        # Create a unique key for this text and button size
        cache_key = (text, round(center_x), round(center_y), round(button_width), round(button_height))

        if cache_key in self.font_cache:
            # Retrieve cached text surface and rect
            text_surface, text_rect = self.font_cache[cache_key]
        else:
            # Binary search for the best font size
            low, high = 1, min(button_width, button_height)
            font_size = low
            while low <= high:
                mid = (low + high) // 2
                text_font = pygame.font.Font(None, mid)
                text_surface = text_font.render(text, False, (255, 255, 255))
                text_rect = text_surface.get_rect()
                if text_rect.width <= button_width and text_rect.height <= button_height:
                    font_size = mid  # Font size fits, try a larger size
                    low = mid + 1
                else:
                    high = mid - 1

            # Use the best font size to render the final surface
            text_font = pygame.font.Font(None, font_size)
            text_surface = text_font.render(text, False, (255, 255, 255)).convert()
            text_rect = text_surface.get_rect(center=(center_x, center_y))

            # Cache the rendered text surface and rect
            self.font_cache[cache_key] = (text_surface, text_rect)

        # Render the cached text
        screen.blit(text_surface, text_rect)

    def draw_menu(self):
        # Iterate over the list of buttons and draw each one
        for button_props in self.button_list:
            width, height = button_props["width"], button_props["height"]
            if button_props["button_label"] == "Ball BL":
                half_width, half_height = width / 2, height / 2
                button_props["pos_x"] = balls[0].body.position[0] - half_width
                button_props["pos_y"] = balls[0].body.position[1] - half_height + 100
                button_props["body"].position = pymunk.Vec2d(button_props["pos_x"] + half_width,
                                                             button_props["pos_y"] + half_height)
                for shape in button_props["body"].shapes:
                    shape.cache_bb()  # Update the bounding box of the shape to match the new position
            pos_x, pos_y = button_props["pos_x"], button_props["pos_y"]
            color1, color2 = button_props["color1"], button_props["color2"]
            label = button_props["label"]
            shadow_factor = button_props.get("shadow_factor", 0.5)

            if button_props["is_pressed"]:
                shadow_factor = 0.3  # Darker shadow when pressed
                self.shadow_movement = 0
            else:
                self.shadow_movement = 2

            if not (pos_x, pos_y, width, height, self.shadow_movement) == button_props["previous_rect"]:
                button_props["button_rects"].clear()
                button_rect2 = pygame.Rect(pos_x - 6, pos_y - 6, width + 12, height + 12)
                button_rect3 = pygame.Rect(pos_x + self.shadow_movement, pos_y + self.shadow_movement, width + 2,
                                           height + 2)
                button_rect1 = pygame.Rect(pos_x - self.shadow_movement, pos_y - self.shadow_movement, width, height)
                button_props["button_rects"] += [button_rect1, button_rect2, button_rect3]

            pygame.draw.rect(screen, color2, button_props["button_rects"][1])
            pygame.draw.rect(screen, self.calculate_shadow_color(color2, shadow_factor),
                             button_props["button_rects"][2])
            pygame.draw.rect(screen, color1, button_props["button_rects"][0])

            self.draw_text(label, pos_x - self.shadow_movement + width // 2, pos_y - self.shadow_movement + height // 2,
                           width, height)

        for i in range(len(self.switch_list)):
            #self.create_hitboxes(i)
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
                button_rect1 = pygame.Rect(
                    switch["slider_position"][0] - self.shadow_movement - switch["slider_size"][0] / 2,
                    switch["slider_position"][1] - self.shadow_movement, switch["slider_size"][0],
                    switch["slider_size"][1])
                button_rect2 = pygame.Rect(switch["slider_position"][0] - 2 - switch["slider_size"][0] / 2,
                                           switch["slider_position"][1] - 2,
                                           switch["slider_size"][0] + 4, switch["slider_size"][1] + 4)
                button_rect3 = pygame.Rect(
                    switch["slider_position"][0] + self.shadow_movement - switch["slider_size"][0] / 2,
                    switch["slider_position"][1] + self.shadow_movement, switch["slider_size"][0] + 2,
                    switch["slider_size"][1] + 2)
                pygame.draw.rect(screen, switch["color2"], button_rect2)
                pygame.draw.rect(screen, self.calculate_shadow_color(switch["color2"], switch["shadow_factor"]),
                                 button_rect3)
                pygame.draw.rect(screen, switch["color1"], button_rect1)
            else:
                switch["slider_size"][0], switch["slider_size"][1] = switch["width"] + 12, switch["height"] / 10
                button_rect1 = pygame.Rect(switch["slider_position"][0] - self.shadow_movement,
                                           switch["slider_position"][1] - self.shadow_movement - switch["slider_size"][
                                               1] / 2,
                                           switch["slider_size"][0], switch["slider_size"][1])
                button_rect2 = pygame.Rect(switch["slider_position"][0] - 2,
                                           switch["slider_position"][1] - 2 - switch["slider_size"][1] / 2,
                                           switch["slider_size"][0] + 4, switch["slider_size"][1] + 4)
                button_rect3 = pygame.Rect(switch["slider_position"][0] + self.shadow_movement,
                                           switch["slider_position"][1] - switch["slider_size"][
                                               1] / 2 + self.shadow_movement,
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

        for button_props in self.button_list:
            if check_aabb_to_point_collision(button_props["shape"], pygame.mouse.get_pos()):
                if press_type == "visual":
                    button_props["is_pressed"] = True
                else:
                    self.button_assigner(button_props["label"], button_props["button_label"])

        # Check switches - updated to use spatial partitioning
        for switch in self.switch_list:
            # Check all switch positions (states)
            for idx, switch_pos in enumerate(switch["switch_positions"]):
                # Create a rectangle for the current switch position
                switch_rect = pygame.Rect(switch_pos[0], switch_pos[1], switch_pos[2], switch_pos[3])

                # If the mouse position is over any of the switch's hitboxes
                if switch_rect.collidepoint(mouse_position):
                    if press_type == "visual":
                        switch["is_pressed"] = True
                    else:
                        # Update the switch state based on the clicked hitbox index
                        switch["is_pressed"] = False
                        switch["previous_switch_state"] = switch["switch_state"]
                        switch["switch_state"] = idx
                        switch["moved"] = False
                        self.button_assigner(switch["label"], switch["button_label"], switch["switch_state"])

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
        elif bl == "Misc option BL":
            self.change_menu(5)
        elif bl == "enemy texture BL":
            global enemy_image, enemy_image_path
            selected_texture = select_texture_and_scale()
            enemy_image = selected_texture[0]
            enemy_image_path = selected_texture[1]
        elif bl == "debug option BL":
            global debug_mode
            debug_mode = switch_option
            #print("Debug mode is: " + str(debug_mode))
        elif bl == "screen option BL":
            global screen
            options = [
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
        elif bl == "collision mode BL":
            global collision_mode
            collision_mode = switch_option
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
                        "Every human who has reproduced in history...", "Only for this guy to obsess over a button...",
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
                    if len(sentences) - 1 > button["sentence"]:
                        button["sentence"] += 1

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
                # if switch is pressed statement is only for interrupted sequence during animation
                if switch["is_pressed"]:
                    switch["slider_position"][2] = 0  # Reset animation progress
                    # Reset animation for new state
                    switch["previous_switch_position"] = list(switch["slider_position"][:2])
                    switch["target_switch_position"] = list(switch["switch_positions"][switch["switch_state"]][:2])
                    continue  # Restart loop with updated target position

                # Ensure initial positions are set
                target_position_buffer = switch["switch_positions"][switch["switch_state"]]
                switch["target_switch_position"] = [
                    target_position_buffer[0] + target_position_buffer[2] / 2,  # x + half width
                    target_position_buffer[1] + target_position_buffer[3] / 2  # y + half height
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
        self.load_upgrades_from_xml('upgrades.xml')
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
            for crosshair_1 in crosshair.crosshair_list:
                crosshair_1["size"] += 5

    def draw_upgrades(self):
        self.calculate_grid_positions()
        # Draw upgrades and connections
        for upgrade1 in self.upgrades_list:
            pos1 = (upgrade1["pos_x"], upgrade1["pos_y"])
            size = upgrade1["size"]
            #rectangle_instance.draw_rects(screen, (255*random.random(), 10, 0), (pos1[0] - size/2, pos1[1] - size/2, size, size), "add")
            menu.create_button(pos1[0] - size / 2, pos1[1] - size / 2, size, size, (100, 0, 100), (100, 100, 100), 0.2,
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
balls = [Ball(100, 100, 15, (200, 200), (255, 255, 0))]
enemies = Enemy()
upgrade_instance = Upgrade()
particles = Particle()
crosshair.create_crosshair(0, 0, 80, 40, 5, (255, 0, 0), (0, 0))
mb_down_toggled = 0
mouse_button_held = {1: False, 3: False}
font = pygame.font.Font(None, 30)
precompute_unit_rotations()


def call_both_spaces(param):
    no_physics_space.step(param)
    space.step(param)


def main():
    global no_sleep_frame_time, frame_time, frame_start_time, frame_counter, mouse_pos, smoothed_fps, paused, debug_mode, screen_width, screen_height, smoothed_lrps
    running = True
    while running:
        frame_start_time = time.perf_counter()
        frame_counter += 1
        # Update (logical) section – for TPS
        if frame_time >= target_tick_time:  # Run TPS updates
            enemies.move()
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
            movement_speed = frame_time  # Scale by frame time for consistent movement

            # Directly update the ball's position
            if paused == 2 or paused == 3:
                balls[0].body.velocity = (0, 0)
                if key[pygame.K_w]:
                    balls[0].body.position = (balls[0].body.position[0], balls[0].body.position[1] - movement_speed)
                if key[pygame.K_s]:
                    balls[0].body.position = (balls[0].body.position[0], balls[0].body.position[1] + movement_speed)
                if key[pygame.K_a]:
                    balls[0].body.position = (balls[0].body.position[0] - movement_speed, balls[0].body.position[1])
                if key[pygame.K_d]:
                    balls[0].body.position = (balls[0].body.position[0] + movement_speed, balls[0].body.position[1])

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
                        if debug_mode == 3:
                            debug_mode = 0
                        elif debug_mode == 0:
                            debug_mode = 1
                        elif debug_mode == 1:
                            debug_mode = 2
                        elif debug_mode == 2:
                            debug_mode = 3
                    if event.key == pygame.K_k:
                        if paused == 3:
                            paused = 0
                        elif paused == 0:
                            paused = 1
                        elif paused == 1:
                            paused = 2
                        elif paused == 2:
                            paused = 3
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_button_held[1] = True
                        menu.check_button_press(mouse_pos, "visual")
                    elif event.button == 3:  # Right mouse button
                        mouse_button_held[3] = True
                        enemies.create_enemy(mouse_pos[0], mouse_pos[1], 60, 0, (255, 0, 0), (255, 200, 0), 1, 3, 100,
                                             100, 45, None,
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
            else:
                crosshair.shoot(visual=True)

        # Render section – for FPS
        if frame_time >= target_frame_time:  # Run FPS updates
            # Drawing/rendering updates
            screen.fill(bg_color)  # Fill the display with a solid color
            crosshair.move_crosshair(mouse_pos[0], mouse_pos[1])
            rectangle_instance.draw_lines(None, None, None, None, None, "draw")

            particles.draw()
            for ball in balls:
                ball.move()
                if debug_mode == 0 or debug_mode == 2 or debug_mode == 3:
                    ball.draw()
                elif debug_mode == 1:
                    draw_ball_debug_info(ball)

            enemies.draw()
            #no_physics_space.debug_draw(draw_options)
            #space.debug_draw(draw_options)
            menu.move_switch()
            menu.draw_menu()

            rectangle_instance.draw_rects(None, None, None, "draw")
            crosshair.draw()

            # Update the pymunk space (run the physics simulation)
            handler = space.add_collision_handler(1, 2)
            handler.begin = ball_hits_enemy

            if paused == 0:  # Unpaused state
                # Restore saved velocities for all bodies
                for body, (velocity, angular_velocity) in paused_velocities.items():
                    body.velocity = velocity
                    body.angular_velocity = angular_velocity
                paused_velocities.clear()  # Clear the saved velocities
                # Normal Pymunk physics step
                call_both_spaces(frame_time)

            elif paused == 1:  # Partially paused state (non-player bodies freeze)
                for body in space.bodies:
                    if body != balls[0].body:  # Skip the player's body
                        if body not in paused_velocities:
                            # Save the body's current velocities
                            paused_velocities[body] = (body.velocity, body.angular_velocity)
                        body.velocity = (0, 0)
                        body.angular_velocity = 0
                # Perform a minimal physics step for collision detection
                call_both_spaces(0)

            elif paused == 2:  # Fully paused state (all bodies freeze except the player)
                for body in space.bodies:
                    if body == balls[0].body:  # Skip freezing the player's body
                        paused_velocities[body] = (body.velocity, body.angular_velocity)
                        continue
                    if body not in paused_velocities:
                        # Save the body's current velocities
                        paused_velocities[body] = (body.velocity, body.angular_velocity)
                    body.velocity = (0, 0)
                    body.angular_velocity = 0
                # Perform a minimal physics step for collision detection
                call_both_spaces(0)
            elif paused == 3:
                # Restore saved velocities for all bodies
                for body, (velocity, angular_velocity) in paused_velocities.items():
                    body.velocity = velocity
                    body.angular_velocity = angular_velocity
                # Normal Pymunk physics step
                call_both_spaces(frame_time)

            if debug_mode == 3:
                #draw_debug_grid(mouse_pos)
                pass
            # Enforce frame delay to match target FPS

            smoothing_factor = 0.0  # Adjust this between 0 and 1; closer to 1 means more smoothing

            # Update smoothed FPS based on time since last frame
            if frame_time > 0:  # Prevent division by zero
                current_fps = 1 / frame_time  # Convert ms to FPS
                smoothed_fps = (smoothing_factor * smoothed_fps) + ((1 - smoothing_factor) * current_fps)
            #    current_lrps  = 1 / no_sleep_frame_time
            #    smoothed_lrps = (smoothing_factor * smoothed_lrps) + ((1 - smoothing_factor) * current_lrps)

            # Render and display smoothed FPS
            fps_display1 = font.render(f"FPS: {smoothed_fps:.0f}", True, (255, 55, 255), (0, 0, 0))
            fps_display2 = font.render(f"Frametime: {frame_time * 1000:.1f}",
                                       True, (255, 255, 55), (0, 0, 0))
            fps_display3 = font.render(f"Logic/Rendering Frametime: {no_sleep_frame_time * 1000:.1f}",
                                       True, (255, 255, 55), (0, 0, 0))
            #fps_display4 = font.render(f"L/R FPS: {smoothed_lrps:.1f}", True, (255, 255, 55), (0, 0, 0))
            screen.blit(fps_display1, (10, 10))  # Display FPS in the top-left corner
            screen.blit(fps_display2, (10, 30))
            screen.blit(fps_display3, (200, 10))
            #screen.blit(fps_display4, (200, 30))
            # Refresh display
            pygame.display.flip()

        frame_time = time.perf_counter() - frame_start_time
        no_sleep_frame_time = frame_time

        if frame_time < target_frame_time:
            time.sleep(target_frame_time - frame_time)
            frame_time = time.perf_counter() - frame_start_time
    pygame.quit()


if __name__ == "__main__":
    main()
