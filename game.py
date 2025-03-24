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
import moderngl
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import BB
from pygame_light2d import LightingEngine, PointLight, Hull, FOREGROUND
from PIL import Image

pygame.init()
pygame.joystick.init()

# Check if any joysticks are connected
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)  # Get the first joystick
    joystick.init()
    print(f"Connected to: {joystick.get_name()}")
else:
    joystick = False
    print("No joystick detected.")



'''Controller Values'''
DEADZONE = 0.1
POINTER_SPEED = 15.0
listening = False
drift = False
bomb_allowed = True
last_bomb_time = 0
bomb_throwing_speed = 1
# controller input list: button, state, mapped action
controller_input_button_list = [
    ["cross", "False", None],  # Button 0
    ["circle", "False", None],  # Button 1
    ["square", "False", None],  # Button 2
    ["triangle", "False", None],  # Button 3

    ["share", "False", None],  # Button 4
    ["ps_button", "False", None],  # Button 5
    ["options", "False", None],  # Button 6
    ["l3", "False", None],  # Button 7
    ["r3", "False", None],  # Button 8
    ["l1", "False", None],  # Button 9
    ["r1", "False", None],  # Button 10

    ["dpad_up", "False", None],  # Button 11
    ["dpad_down", "False", None],  # Button 12
    ["dpad_left", "False", None],  # Button 13
    ["dpad_right", "False", None],  # Button 14
    ["touchpad", "False", None],  # Button 15
]
controller_input_analogue_list = [

    ["l_x_axis", 0.0, None],  # Axis 0
    ["l_y_axis", 0.0, None],  # Axis 1
    ["r_x_axis", 0.0, None],  # Axis 2
    ["r_y_axis", 0.0, None],  # Axis 3
    ["l2", 0.0, None],  # Axis 4
    ["r2", 0.0, None],  # Axis 5
]

info = pygame.display.Info()
tv_width, tv_height = info.current_w, info.current_h
screen_res = tv_width, tv_height
native_res = tv_width, tv_height

section_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

engine = LightingEngine(screen_res=screen_res, native_res=native_res, lightmap_res=native_res)
screen_layer = engine.graphics.make_layer(screen_res)

#draw_options = pymunk.pygame_util.DrawOptions(screen)
# Create Pymunk Space
space = pymunk.Space(True)
no_physics_space = pymunk.Space(True)
space.gravity = (0, 0)
BALL_COLLISION_TYPE = 1
ENEMY_COLLISION_TYPE = 2

# Cache for rotated rectangles
rotation_cache = {}
paused_velocities = {}
obstacles = []
obstacles_menu = []
light_mask = None
paused = 0
running = True
timer = False  # binary timer to execute stuff only every second time
sdl = ctypes.CDLL("SDL2.dll")
hwnd = pygame.display.get_wm_info()['window']
user32 = ctypes.windll.user32
screen_width = tv_width
screen_height = tv_height
clock = pygame.time.Clock()
cell_size = 64  # Size of each spatial partitioning cell
grid = defaultdict(list)  # Grid for spatial partitioning
reddings_amount = 0
rand = random.Random()
user_name = "standard"
mouse_pos = (50, 50)
max_velocity = 8
acceleration_speed = 0.8  # Player ball acceleration
friction = 0.98  # Friction for the perpendicular axis when changing the players' movement direction
debug_mode = 0
collision_mode = 0
screen_limit_x = screen_width * 2
screen_limit_y = screen_height * 2
playing = False
fps_surface = engine.graphics.make_layer((700, 80))
fps_surfaces = []
enemy_image_path = None
enemy_image = None

target_tps = 20
target_tick_time = 1 / target_tps
target_itps = 60
target_input_tick_time = 1 / target_itps
target_fps = 240
target_frame_time = 1 / target_fps
prev_time = time.time()
no_sleep_frame_time = 0.05
frame_time = 0.05
frame_start_time = time.perf_counter()
smoothed_fps = 240
smoothed_lrps = smoothed_fps
frame_counter = 0
tick_counter = 0
alpha = 0.0  # Interpolation factor
accumulator = 0.0  # Stores leftover time between physics steps
input_accumulator = 0.0
bg_color = (8/255, 0, 0)


def save_texture_as_png(texture: moderngl.Texture, filename: str):
    """Saves a moderngl.Texture as a PNG file."""
    width, height = texture.size  # Get texture size
    data = texture.read()  # Read pixel data (bytes)

    # Convert to a numpy array (RGBA format)
    image = Image.frombytes("RGBA", (width, height), data)

    # Save as PNG
    image.save(filename, "PNG")


def resize_shape(pymunk_space, body, old_shape, new_size):
    """
    Resizes a pymunk shape by removing the old shape and creating a new one.

    Parameters:
    - pymunk_space (pymunk.Space): The physics space where the shape exists.
    - body (pymunk.Body): The body to which the shape is attached.
    - old_shape (pymunk.Shape): The shape that needs to be resized.
    - new_size (float or tuple): The new size for the shape. If a float is given, it assumes a square (new_size, new_size).

    Returns:
    - pymunk.Shape: The newly created shape with the updated size.

    Notes:
    - pymunk shapes cannot be resized directly, so a new shape must be created.
    - The old shape is removed from the space before adding the new shape.
    - The new shape retains the same body but needs to be re-added to the space.
    """
    # Remove the old shape from the physics space
    pymunk_space.remove(old_shape)

    # Ensure new_size is a tuple (width, height)
    if isinstance(new_size, (int, float)):
        new_size = (new_size, new_size)

    # Create a new shape with the updated size
    new_shape = pymunk.Poly.create_box(body, new_size)

    # Add the new shape to the space
    pymunk_space.add(new_shape)

    return new_shape

def null_window_position(x=0, y=0):
    user32.MoveWindow(hwnd, x, y, tv_width, tv_height)


def switch_to_borderless():
    global engine, screen_layer, screen_res
    # Do it twice because of a stupid black bar bug on the bottom...
    screen_res = (tv_width, tv_height)
    for _ in range(2):
        engine = LightingEngine(screen_res=screen_res, native_res=screen_res, lightmap_res=screen_res, noframe=True, vsync=True)
        screen_layer = engine.graphics.make_layer(screen_res)
        #render_engine = pygame_light2d.RenderEngine(tv_width, tv_height, )
        user32.ShowWindow(hwnd, 1)  # Show the window normally (not minimized)
        null_window_position()  # Reset the position after making it borderless


# Function to switch to windowed mode
def switch_to_windowed():
    global engine, screen_res, screen_layer
    screen_res = (screen_width, screen_height)
    engine = LightingEngine(screen_res=screen_res, native_res=screen_res, lightmap_res=screen_res, resizable=True, vsync=True)
    screen_layer = engine.graphics.make_layer(screen_res)
    null_window_position(0, 1)
    user32.ShowWindow(hwnd, 3)  # Maximize the window when switching


# Function to switch to fullscreen mode
def switch_to_fullscreen():
    global engine, screen_res, screen_layer
    screen_res = (tv_width, tv_height)
    engine = LightingEngine(screen_res=screen_res, native_res=screen_res, lightmap_res=screen_res,
                            fullscreen=True, vsync=True)
    screen_layer = engine.graphics.make_layer(screen_res)


def damage_enemy(red_nemesis, damage=1, cross_shoot=None, ball=None):
    global paused
    red_nemesis["is_attacked"] = True
    if cross_shoot is not None:
        red_nemesis["hp"] -= cross_shoot["damage"]
        if red_nemesis["hp"] < 0:
            red_nemesis["lost_hp"] = cross_shoot["damage"] + red_nemesis["hp"]
        else:
            red_nemesis["lost_hp"] = cross_shoot["damage"]
        cross_shoot["enemy_hit"] = True
        cross_shoot["last_shot_time"] = pygame.time.get_ticks()  # Store the shot time
    elif ball:
        red_nemesis["hp"] -= damage
        if red_nemesis["hp"] < 0:
            red_nemesis["lost_hp"] = damage + red_nemesis["hp"]
        else:
            red_nemesis["lost_hp"] = damage
        ball.hp -= damage
        if ball.hp <= 0:
            ball.dead = True
            paused = 0
    else:
        red_nemesis["hp"] -= damage
        if red_nemesis["hp"] < 0:
            red_nemesis["lost_hp"] = damage + red_nemesis["hp"]
        else:
            red_nemesis["lost_hp"] = damage

    if red_nemesis["hp"] <= 0:
        remove_enemy(red_nemesis["body"], red_nemesis["shape"], red_nemesis, enemies.red_nemesis_list)
        particles.create_particle(red_nemesis["pos_x"], red_nemesis["pos_y"], 50, 1, red_nemesis["color"], "damage", 3, 60)
        particles.create_particle(red_nemesis["pos_x"], red_nemesis["pos_y"], 25, 0.1, (10, 240, 5), "reddings", 0)


def remove_enemy(enemy_body, shape, red_nemesis, red_nemesis_list):
    space.remove(enemy_body, shape)
    if red_nemesis in red_nemesis_list:  # Ensure it's in the list before removing
        red_nemesis_list.remove(red_nemesis)


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

    # You could add a check for bounding box intersection first, as it's less expensive than checking full collisions
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


def vertices_from_aabb(position, size):
    """
    Generates the vertices for a rectangle based on the given position and size.

    :param position: tuple (x, y), position of the top-left corner of the rectangle.
    :param size: tuple (width, height), size of the rectangle.
    :return: list of vertices (as tuples of coordinates).
    """
    x, y = position
    width, height = size

    # Define the vertices of the rectangle (counter-clockwise order)
    vertices = [
        (x, y),  # top-left
        (x + width, y),  # top-right
        (x + width, y + height),  # bottom-right
        (x, y + height)  # bottom-left
    ]

    return vertices


def vertices_from_ball(center, radius, num_vertices, angle1=0, angle2=2 * math.pi):
    """
    Generate the outer vertices approximating a circle (ball) given a center, radius, and number of vertices.

    Args:
    - center: tuple (x, y), the position of the center of the ball
    - radius: float, the radius of the ball
    - num_vertices: int, the number of vertices to approximate the circle
    - angle1: float, starting angle in radians (default is 0)
    - angle2: float, ending angle in radians (default is 2*pi, full circle)

    Returns:
    - List of vertices (x, y) approximating the circle
    """
    # Initialize vertices list with the center
    vertices = [center]

    # Generate the vertices by evenly distributing angles between angle1 and angle2
    for angle in np.linspace(angle1, angle2, num_vertices + 1):
        # Calculate the x and y coordinates for each vertex using polar-to-cartesian conversion
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))

    return vertices

def draw_ball_debug_info(a_ball):
    engine.graphics.render_circle(screen_layer, (255, 0, 0),
                                  (int(a_ball.body.position.x), int(a_ball.body.position.y)),
                                  a_ball.radius)


def draw_enemy_debug_info():
    for enemy in enemies.red_nemesis_list:
        # Draw the square's collision boundaries
        engine.graphics.render_primitive(screen_layer, (0, 255, 0), enemy["vertices"])


def ball_hits_enemy(arbiter, _shape, _data):
    # Get the shapes involved in the collision
    ball_shape, enemy_shape = arbiter.shapes

    # Find the enemy in your list
    for enemy in enemies.red_nemesis_list:
        if enemy["shape"] == enemy_shape:
            damage_enemy(enemy, 13, None, balls[0])
            return True
    return False


def clamp_velocity(velocity, min_value, max_value):
    """
    Clamps the velocity between the given min and max values.
    This function works for floats to support subtle acceleration.
    """
    return max(min_value, min(max_value, velocity))


def convert_color(color):
    conv_color = tuple(c / 255.0 for c in color)
    return conv_color


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
                print(f"Loaded enemy image path from XML. {enemy_image_path}")
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

    def draw_rects(self, layer, color, rect, mode):
        if mode == "add":
            rect_properties = {
                "surface": layer,
                "color": color,
                "rect": rect
            }
            self.rect_list.append(rect_properties)
        elif mode == "draw":
            for rect in self.rect_list:
                engine.graphics.render_rectangle(rect["surface"], rect["color"], (rect["rect"][0], rect["rect"][1]), rect["rect"][2], rect["rect"][3])

    def draw_lines(self, layer, color, pos1, pos2, thickness, mode):
        if mode == "add":
            line_properties = {
                "surface": layer,
                "color": color,
                "pos1": pos1,
                "pos2": pos2,
                "thickness": thickness
            }
            self.line_list.append(line_properties)
        elif mode == "draw":
            for line in self.line_list:
                engine.graphics.render_thick_line(line["surface"], line["color"], line["pos1"], line["pos2"], line["thickness"])

    @staticmethod
    def draw_line(layer, pos1, pos2, color=(255, 255, 255), thickness=2):
        engine.graphics.render_thick_line(layer, color, pos1, pos2, thickness)


class Particle:
    def __init__(self):
        self.rendering_groups = []
        self.looks_ids = []
        self.particle_list = []
        self.precomputed_particles = {}
        self.direction_map = {
            "nw": (-1 + random.uniform(-0.5, 0.5), -1 + random.uniform(-0.5, 0.5)),
            "ne": (1 + random.uniform(-0.5, 0.5), -1 + random.uniform(-0.5, 0.5)),
            "sw": (-1 + random.uniform(-0.5, 0.5), 1 + random.uniform(-0.5, 0.5)),
            "se": (1 + random.uniform(-0.5, 0.5), 1 + random.uniform(-0.5, 0.5))
        }

    def change_particle_color(self, particle, color):
        # Convert colors to the 0-1 range
        #color = convert_color(color)

        def create_looks_group(looks_id):
            """Helper function to create a new looks group."""
            particle["looks_id"] = looks_id
            print("partikel: ", particle["looks_id"], color)
            self.looks_ids.append((looks_id, color))
            self.rendering_groups.append(
                ([], color))  # Add new rendering group

        # If there's no looks_id yet, or the list is empty, create the first looks group
        if particle["looks_id"] is None and len(self.looks_ids) == 0:
            create_looks_group(0)
            return  # Exit after creating the first group

        # Check if the current look_id matches the color, or create a new one
        for i, (look_id, look_color) in enumerate(self.looks_ids):
            if look_color == color:
                particle["looks_id"] = look_id
                break
            elif i == len(self.looks_ids) - 1:
                create_looks_group(look_id + 1)

        # Update the nemesis color
        particle["color"] = color

    def create_particle(self, pos_x, pos_y, size, speed, color, label, max_gen, max_timer=60):
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
            "max_timer": max_timer,
            "flip_speed": speed,
            "start_index": 0,
            "end_index": 4,  # Tracks how many particles are currently active
            "total_particles": total_particles,
            "vel_x": 0,
            "vel_y": 0,
            "looks_id": None
        }
        self.particle_list.append(particle_properties)
        particle = self.particle_list[-1]
        self.change_particle_color(particle, color)

    def precompute_particles(self, particle_type, size, color):
        """
        Precomputes rotated particle surfaces for all 360 degrees of rotation.
        """
        self.precomputed_particles[particle_type] = {}
        if particle_type == "reddings":
            for angle in range(360):
                # Create a transparent surface for the particle

                particle_surface = engine.graphics.make_layer((size, size))
                flip_scale = math.fabs(math.sin(math.radians(angle)))  # Alternative, same effect
                # Define the stretched diamond shape
                diamond = [
                    (size // 2, 0),  # Top point
                    (3 * size // 4 * flip_scale, size // 2),  # Right point (closer to center horizontally)
                    (size // 2, size - 1),  # Bottom point
                    (size // 4 * flip_scale, size // 2)  # Left point (closer to center horizontally)
                ]
                engine.graphics.render_primitive(particle_surface, color, diamond, mode=moderngl.TRIANGLES)
                engine.graphics.render_primitive(particle_surface, (255, 255, 0), diamond)

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
                    (3 * size // 5 * flip_scale, size // 2),  # Right point
                    (size // 2, 3 * size // 4)  # Near the bottom-center
                ]
                engine.graphics.render_primitive(particle_surface, shine_color, shine_triangle, mode=moderngl.TRIANGLES)

                self.precomputed_particles[particle_type][angle] = particle_surface

#    def move(self):


    def draw(self):
        """Update and render all particles."""
        particles_to_remove = []
        damage_particles_to_render = []

        # Iterate particles in reverse
        for e in range(len(self.particle_list) - 1, -1, -1):  # Iterate in reverse
            particle = self.particle_list[e]
            particle_timer = particle["timer"]
            particle_max_gen = particle["max_gen"]
            # Particle lifetime check (optimize by collecting removals later)
            if particle_timer > 60 * particle_max_gen:
                self.direction_map = {
                    "nw": (-1 + random.uniform(-0.5, 0.5), -1 + random.uniform(-0.5, 0.5)),
                    "ne": (1 + random.uniform(-0.5, 0.5), -1 + random.uniform(-0.5, 0.5)),
                    "sw": (-1 + random.uniform(-0.5, 0.5), 1 + random.uniform(-0.5, 0.5)),
                    "se": (1 + random.uniform(-0.5, 0.5), 1 + random.uniform(-0.5, 0.5))
                }
                particles_to_remove.append(e)
                continue

            # Particle rotation update
            particle["rotation"] = (particle["rotation"] + 1) % 360
            label = particle["label"]

            if label == "damage":

                particle_max_timer = particle["max_timer"]
                particle_current_gen = particle["current_gen"]
                particle_x = particle["particle_x"]
                particle_y = particle["particle_y"]
                if particle_timer > particle_max_timer and particle_current_gen < particle_max_gen:
                    particle_current_gen += 1
                    particle_timer = 0
                else:
                    particle_timer += 1
                particle["timer"] = particle_timer
                particle["current_gen"] = particle_current_gen


                # Initialize new directions for the current generation
                if not particle["direction_chosen"][particle_current_gen - 1]:
                    start_index = (4 ** particle_current_gen - 4) // 3
                    end_index = start_index + 4 ** particle_current_gen
                    particle["end_index"] = end_index
                    particle["start_index"] = start_index
                    # Copy positions from the previous generation
                    prev_start_index = start_index - 4 ** (particle_current_gen - 1)
                    for i in range(start_index, end_index):
                        source_index = prev_start_index + (i - start_index) // 4
                        particle_x.append(particle_x[source_index])
                        particle_y.append(particle_y[source_index])

                    # Assign directions
                    for i in range(start_index, end_index):
                        particle["directions"].append(random.choice(["nw", "ne", "sw", "se"]))
                        particle["generation"].append(particle_current_gen)
                    particle["direction_chosen"][particle_current_gen - 1] = True

                for i in range(0, particle["end_index"]):
                    if particle["generation"][i] == particle_current_gen:
                        # Move particles
                        dx, dy = self.direction_map[particle["directions"][i]]
                        particle_x[i] += dx * particle["speed"]
                        particle_y[i] += dy * particle["speed"]

                        # Render particles
                        size_factor = particle["size"] / (2.1 ** particle["generation"][i])
                        vertices = get_rotated_vertices(particle_x[i], particle_y[i],
                                                        size_factor, particle["rotation"])
                        damage_particles_to_render.append((vertices[:3], particle["looks_id"]))

                particle["particle_x"] = particle_x
                particle["particle_y"] = particle_y

            elif label == "reddings":
                pos_x, pos_y = particle["pos_x"], particle["pos_y"]
                size = particle["size"]
                particle["rotation"] += particle["flip_speed"]
                # Initialize total forces
                force_x, force_y = 0, 0

                # Compute gravitational force from all crosshairs
                crosshairs = crosshair.crosshair_list[0]
                cross_x, cross_y = crosshairs["pos_x"], crosshairs["pos_y"]
                distance = get_distance(pos_x, pos_y, cross_x, cross_y) - 20

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
                    particles_to_remove.append(e)

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
                # Blit the particle
                engine.graphics.render(particle_surface.texture, screen_layer, (pos_x, pos_y))

        #print(len(damage_particles_to_render), len(self.particle_list))

        # Optimized first loop
        for particle in damage_particles_to_render:
            looks_id = particle[1]
            vertices = particle[0]
            self.rendering_groups[looks_id][0].extend(vertices)  # Directly extend with all 3 vertices

        # Optimized second loop: caching repeated list access and removing the print call (if not needed)
        for group_id, group_list in enumerate(self.rendering_groups):  # Use enumerate for efficient indexing
            filled_vertices = group_list[0]
            color = group_list[1]

            if filled_vertices:  # Check if not empty
                # Directly render if the list is not empty
                engine.graphics.render_primitive(screen_layer, color, filled_vertices, mode=moderngl.TRIANGLES)
                filled_vertices.clear()  # Clear after rendering

        # Optimized third loop: popping from a list using indices directly
        # (assuming `particles_to_remove` contains indices of elements in the `self.particle_list`)
        self.particle_list = [particle for i, particle in enumerate(self.particle_list) if i not in particles_to_remove]


class Ball:
    def __init__(self, x, y, radius, ball_velocity, color):
        # Create the pymunk body and shape
        self.timer = False
        self.radius = radius
        self.color = color
        self.x = x
        self.y = y
        self.max_hp = 150
        self.hp = 150
        self.attacked = False
        self.dead = False
        self.velocity = [0, 0]
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
        if not self.dead:
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
        else:
            self.body.velocity = (0, 0)
        vertices = vertices_from_ball(self.body.position, self.radius, 8)
        distance_to_mouse = get_distance(vertices[0][0], vertices[0][1], mouse_pos[0], mouse_pos[1])
        # if distance_to_mouse < lighting_class.light_radius:
        obstacles.append((vertices, distance_to_mouse))

    def draw(self):
        # Draw the ball using pygame
        engine.graphics.render_circle(screen_layer, self.color,
                                      (int(self.body.position.x), int(self.body.position.y)), self.radius, num_segments=16)

enemy_draw_counter = 0

class Enemy:
    def __init__(self):
        self.red_nemesis_list = []
        self.enemy_saved_textures = {}
        self.max_enemy_size = 100
        self.looks_ids = []
        self.rendering_groups = [] # List of groups of enemies to render.
            # Group is a tuple consisting of two vertices lists, inside and outline.

    def change_enemy_color(self, red_nemesis, color):
        # Convert colors to the 0-1 range
        color = convert_color(color)
        edge_color = convert_color(red_nemesis["edge_color"])

        def create_looks_group(looks_id):
            """Helper function to create a new looks group."""
            red_nemesis["looks_id"] = looks_id
            print(red_nemesis["looks_id"], color)
            self.looks_ids.append((looks_id, color, edge_color))
            self.rendering_groups.append(
                ([], [], (color, edge_color)))  # Add new rendering group

        # If there's no looks_id yet, or the list is empty, create the first looks group
        if red_nemesis["looks_id"] is None and len(self.looks_ids) == 0:
            create_looks_group(0)
            return  # Exit after creating the first group

        # Check if the current look_id matches the color and edge_color, or create a new one
        for i, (look_id, look_color, look_edge_color) in enumerate(self.looks_ids):
            if look_color == color and look_edge_color == edge_color:
                red_nemesis["looks_id"] = look_id
                break
            elif i == len(self.looks_ids) - 1:
                create_looks_group(look_id + 1)

        # Update the nemesis color
        red_nemesis["color"] = color

    def choose_random_props(self):
        margin = 100
        x, y = 0, 0
        for red_nemesis in self.red_nemesis_list:
            if not red_nemesis["random_props_chosen"]:
                #if red_nemesis["label"] == "random_guy":
                if red_nemesis["speed"] is None:
                    red_nemesis["speed"] = get_biased_random_float(2, 12)
                if red_nemesis["size"] is None:
                    red_nemesis["size"] = rand.randint(50, 80)
                if red_nemesis["rotation"] is None:
                    red_nemesis["rotation"] = rand.randint(-1, 2)

                edge = random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top':
                    x = random.randint(-margin, screen_width + margin)
                    y = -margin
                elif edge == 'bottom':
                    x = random.randint(-margin, screen_width + margin)
                    y = screen_height + margin
                elif edge == 'left':
                    x = -margin
                    y = random.randint(-margin, screen_height + margin)
                elif edge == 'right':
                    x = screen_width + margin
                    y = random.randint(-margin, screen_height + margin)
                if red_nemesis["pos_x"] is None:
                    red_nemesis["pos_x"] = x
                    red_nemesis["prev_pos_x"] = x
                if red_nemesis["pos_y"] is None:
                    red_nemesis["pos_y"] = y
                    red_nemesis["prev_pos_y"] = y

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
                if not red_nemesis["rotation_initialized"]:
                    enemy_body.angular_velocity = red_nemesis["speed"] / 20 if red_nemesis["rotation"] > 0 else - red_nemesis["speed"] / 20
                    red_nemesis["rotation_initialized"] = True
                red_nemesis["body"] = enemy_body
                red_nemesis["shape"] = shape
                red_nemesis["shape"].collision_type = 2
                self.change_enemy_color(red_nemesis, red_nemesis["og_color"])
                red_nemesis["random_props_chosen"] = True

    def wave(self):
        #if self.red_nemesis_list: self.red_nemesis_list.clear()
        enemy_amount = rand.randint(1, 60)
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
            "prev_pos_x": 0,
            "pos_y": pos_y,
            "prev_pos_y": 0,
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
            "prev_rotation": 0,
            "vertices": [],
            "rotation_initialized": False,
            "last_rotation": 0,
            "direction": direction,
            "is_attacked": is_attacked,
            "weirdness": weirdness,
            "random_props_chosen": False,
            "surface": None,
            "image": None,
            "looks_id": None,
            "previous_framecount": 0
        }

        self.red_nemesis_list.append(enemy_properties)
        self.choose_random_props()

    def move(self):
        """Updates enemy logic (position, rotation, HP, etc.), called at 20 TPS."""
        global collision_mode
        velocity_limit = 10
        second_frame = frame_counter % 1 == 0
        twentieth_frame = frame_counter % 20 == 0
        for i in reversed(range(len(self.red_nemesis_list))):
            red_nemesis = self.red_nemesis_list[i]
            enemy_body = red_nemesis["body"]

            # Update position and rotation
            enemy_body.position += enemy_body.velocity
            pos_x, pos_y = enemy_body.position
            # Store the previous position for interpolation
            red_nemesis["prev_pos_x"] = red_nemesis["pos_x"]
            red_nemesis["prev_pos_y"] = red_nemesis["pos_y"]
            # Set the new positions *AFTER* storing previous position!!!
            red_nemesis["pos_x"], red_nemesis["pos_y"] = pos_x, pos_y

            if second_frame:
                red_nemesis["prev_rotation"] = red_nemesis["rotation"]
                red_nemesis["rotation"] = math.degrees(enemy_body.angle) % 360

                # HP bar animation handling
                if red_nemesis["lost_hp"] > 0:
                    if red_nemesis["prev_lost_hp"] != red_nemesis["lost_hp"] and red_nemesis["prev_lost_hp"] != 0:
                        red_nemesis["lost_hp"] += red_nemesis["prev_lost_hp"]
                        red_nemesis["previous_framecount"] = 0

                    if red_nemesis["previous_framecount"] == 0:
                        red_nemesis["previous_framecount"] = frame_counter

                    elapsed_frames = frame_counter - red_nemesis["previous_framecount"]
                    animation_progress = min(elapsed_frames / 200, 1)
                    red_nemesis["lost_hp"] = max(0, red_nemesis["lost_hp"] * (1 - animation_progress))

                    if animation_progress >= 1:
                        red_nemesis["previous_framecount"] = 0
                        red_nemesis["lost_hp"] = 0
                    else:
                        red_nemesis["prev_lost_hp"] = red_nemesis["lost_hp"]

            if twentieth_frame:
                shape = red_nemesis["shape"]
                if abs(pos_x) > screen_limit_x or abs(pos_y) > screen_limit_y:
                    remove_enemy(enemy_body, shape, red_nemesis, self.red_nemesis_list)
                    continue

                # Limit velocity
                vx, vy = enemy_body.velocity
                enemy_body.velocity = (
                    max(-velocity_limit, min(vx, velocity_limit)),
                    max(-velocity_limit, min(vy, velocity_limit))
                )

        if twentieth_frame:
            # Handle wave logic if enemy count drops below the threshold
            if 1 <= len(self.red_nemesis_list) < 200:
                collision_mode = 1 - collision_mode  # Toggle collision mode
                self.wave()
                print("CREATED ENEMIES", len(self.red_nemesis_list))

    def draw(self, alpha_blend):
        """Renders enemies with interpolation, called every frame."""
        global debug_mode, enemy_image, enemy_image_path, enemy_draw_counter
        # if len(self.red_nemesis_list) != 0:
        #      print("previous position x: ", self.red_nemesis_list[0]["prev_pos_x"], "current position x: ", self.red_nemesis_list[0]["pos_x"], "Alpha blend: ",alpha_blend)
        #print(alpha, "accu: ", accumulator, target_tick_time)

        enemies_to_render = []

        for red_nemesis in self.red_nemesis_list:
            # Interpolate position & rotation
            pos_x = (1 - alpha_blend) * red_nemesis["prev_pos_x"] + alpha_blend * red_nemesis["pos_x"]
            pos_y = (1 - alpha_blend) * red_nemesis["prev_pos_y"] + alpha_blend * red_nemesis["pos_y"]

            diff = (red_nemesis["rotation"] - red_nemesis["prev_rotation"] + 180) % 360 - 180
            rotation = red_nemesis["prev_rotation"] + alpha_blend * diff
            # if red_nemesis == self.red_nemesis_list[0]:
            #     print("1. rotation, alpha_blend: ", rotation, alpha_blend)
            #     print("Position x after calculation: ", pos_x, "Frame: ", frame_counter)
            size = red_nemesis["size"]

            hp = red_nemesis["hp"]
            max_hp = red_nemesis["max_hp"]
            vertices = get_rotated_vertices(pos_x, pos_y, size, rotation)
            distance_to_mouse = get_distance(vertices[0][0], vertices[0][1], mouse_pos[0], mouse_pos[1])
            if distance_to_mouse < lighting_class.light_radius:
                obstacles.append((vertices, distance_to_mouse))

            # Draw HP bar if the enemy was attacked
            if red_nemesis["is_attacked"]:
                hp_factor = hp / max_hp
                pos_x_half_size = pos_x - size / 2
                pos_y_size = pos_y + size
                tenth_size = size * 0.1

                engine.graphics.render_rectangle(screen_layer, (255, 0, 0), (pos_x_half_size, pos_y_size), size,
                                                 tenth_size)
                engine.graphics.render_rectangle(screen_layer, (0, 255, 0), (pos_x_half_size, pos_y_size),
                                                 size * hp_factor + 1, tenth_size)

                animated_size = size * (red_nemesis["lost_hp"] / 100)
                engine.graphics.render_rectangle(screen_layer, (200, 255, 200),
                                                 (max(pos_x_half_size, pos_x_half_size + size * hp_factor), pos_y_size),
                                                 animated_size, tenth_size)
            # Debug mode rendering
            if debug_mode == 0:
                edge_color = red_nemesis["edge_color"]
                color = red_nemesis["color"]
                enemies_to_render.append((color, edge_color, vertices, red_nemesis["looks_id"]))
            if debug_mode == 1:
                # Literally do nothing (Lighting only mode)
                pass

            if debug_mode == 2:
                image = red_nemesis["image"]
                if enemy_image_path is not None and image is None:
                    image = engine.graphics.load_texture(str(enemy_image_path))
                    red_nemesis["image"] = image
                if image is not None:
                    print(image)
                    engine.graphics.render(image, screen_layer,(pos_x-size, pos_y-size), (0.1, 0.1), rotation)
                else:
                    # Draw a rectangle if no image exists
                    v1, v2, v3, v4 = vertices
                    engine.graphics.render_primitive(screen_layer, (0, 25, 255), [v2, v3, v1, v4], mode=moderngl.TRIANGLE_STRIP)


        for enemy in enemies_to_render:
            looks_id = enemy[3]
            v1, v2, v3, v4 = enemy[2]
            self.rendering_groups[looks_id][0].extend([v1, v2, v3, v1, v3, v4])
            self.rendering_groups[looks_id][1].extend([v1, v2, v2, v3, v3, v4, v4, v1])

        engine.graphics.ctx.line_width = 3
        for group_id in range(len(self.rendering_groups)):
            group_list = self.rendering_groups[group_id]

            filled_vertices = group_list[0]
            outline_vertices = group_list[1]
            color = group_list[2][0]
            edge_color = group_list[2][1]

            # Now render everything in one call per mode per group (looks_id)
            if len(filled_vertices) != 0:

                engine.graphics.render_primitive(screen_layer, color, filled_vertices, mode=moderngl.TRIANGLES)
                engine.graphics.render_primitive(screen_layer, edge_color, outline_vertices, mode=moderngl.LINES)

                filled_vertices.clear()
                outline_vertices.clear()

class Bomb:
    def __init__(self, x, y, size, bomb_velocity, color, explosion_time, primed):
        self.explosion_time = explosion_time
        self.primed = primed
        self.timer = 0
        self.animation_timer = 0
        self.size = size
        self.color = color
        self.x = x
        self.y = y
        self.damage = 70
        self.exploded = False
        self.attacked = False
        self.velocity = bomb_velocity
        self.animation_frames = 20  # Number of frames for the expansion
        self.max_explosion_size = max(size) * 2  # Define the max explosion radius
        self.current_explosion_size = 0  # Start size
        self.particles_created = False

    def move(self):
        if self.primed:
            self.timer += 1
        if self.timer > self.explosion_time:
            self.exploded = True
            if self.primed:
                self.primed = False
                for enemy in enemies.red_nemesis_list:
                    if get_distance(self.x, self.y, enemy["pos_x"], enemy["pos_y"]) < self.max_explosion_size:
                        damage_enemy(enemy, self.damage)

    def draw(self):
        if self.exploded:
            self.draw_explosion()
            self.size = (max(0, self.size[0] - 2), max(0, self.size[1] - 2))
            engine.graphics.render_thick_line(screen_layer, self.color, (int(self.x), int(self.y)),
                                              (int(self.x), int(self.y + self.size[1])), self.size[0], True)
            if not self.particles_created:
                self.create_particles()
                self.particles_created = True
        else:
            engine.graphics.render_thick_line(screen_layer, self.color, (int(self.x), int(self.y)),
                                              (int(self.x), int(self.y + self.size[1])), self.size[0], True)

    def draw_explosion(self):
        if self.animation_timer < self.animation_frames:
            # Squared expansion effect
            expansion_factor = (self.animation_timer / self.animation_frames) ** 2
            self.current_explosion_size = self.max_explosion_size * expansion_factor

            # Alpha variation over time
            if self.animation_timer < self.animation_frames // 2:
                alpha_bomb = 50 + (30 * (self.animation_timer / (self.animation_frames // 2)))  # From 50 to 80
            else:
                alpha_bomb = max(
                    80 - (80 * ((self.animation_timer - self.animation_frames // 2) / (self.animation_frames // 2))),
                    0)  # Fades to 0

            explosion_color = (255, 0, 0, int(alpha_bomb))  # RGBA with variable alpha
            explosion_layer = engine.graphics.make_layer(size=(int(self.current_explosion_size * 2+1), int(self.current_explosion_size * 2+1)))

            engine.graphics.render_circle(explosion_layer, explosion_color,
                                          (self.current_explosion_size, self.current_explosion_size),
                                          self.current_explosion_size)

            # Draw contour
            engine.graphics.render_circle(explosion_layer, (255, 0, 0, int(alpha_bomb)),
                                          (self.current_explosion_size, self.current_explosion_size),
                                          self.current_explosion_size)

            # Blit to screen
            engine.graphics.render(explosion_layer.texture, screen_layer,
                                   (self.x - self.current_explosion_size + self.size[0] // 2,
                                    self.y - self.current_explosion_size + self.size[1] // 2))

            self.animation_timer += 1

    def create_particles(self):
        num_particles = int(self.max_explosion_size / 8)  # Number of particles based on explosion size
        for _ in range(num_particles):
            pos_x = self.x + random.uniform(-self.max_explosion_size / 4, self.max_explosion_size / 4)
            pos_y = self.y + random.uniform(-self.max_explosion_size / 4, self.max_explosion_size / 4)
            size = random.uniform(5, 15)  # Randomized particle size
            speed = random.uniform(1, 5)  # Randomized speed
            particles.create_particle(pos_x, pos_y, size, speed, self.color, "damage", 2, 20)


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
            "surface": engine.graphics.make_layer((size, size)),
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
                        enemies.change_enemy_color(red_nemesis, tuple(int(c * 0.5) for c in red_nemesis["og_color"]))
                        if cross_shoot["shot_allowed"]:
                            damage_enemy(red_nemesis, 1, cross_shoot)
                    else:
                        enemies.change_enemy_color(red_nemesis, tuple(int(c * 0.5) for c in red_nemesis["og_color"]))
                else:
                    enemies.change_enemy_color(red_nemesis, red_nemesis["og_color"])
        if cross_shoot["enemy_hit"]:
            cross_shoot["shot_allowed"] = False
            cross_shoot["enemy_hit"] = False

    def move_crosshair(self, mouse_pos_x, mouse_pos_y):
        for cross_move in self.crosshair_list:
            cross_move["pos_x"] = mouse_pos_x + cross_move["offset"][0]
            cross_move["pos_y"] = mouse_pos_y + cross_move["offset"][1]
            cross_move["body"].position = pymunk.Vec2d(cross_move["pos_x"],
                                                       cross_move["pos_y"])

            cross_move["shape"].cache_bb()  # Update the bounding box of the shape to match the new position

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
                cross_draw["shape"] = resize_shape(no_physics_space, cross_draw["body"], cross_draw["shape"], cross_draw["size"])

                third_size = size * 0.33
                sixth_size = size * 0.16
                pos_x_plus_half_size = size + half_size
                pos_x_minus_half_size = half_size
                pos_y_plus_half_size = size + half_size
                pos_y_minus_half_size = half_size

                # Create a surface to hold all the rectangles
                cross_draw["surface"] = engine.graphics.make_layer((size*2, size*2))

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
                    engine.graphics.render_rectangle(cross_draw["surface"], color, (rect[0], rect[1]), rect[2], rect[3])

            # Blit the surface to the screen
            engine.graphics.render(cross_draw["surface"].texture, screen_layer, (pos_x - size, pos_y - size))


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
        global user_name, debug_mode, paused
        obstacles_menu.clear()
        if rectangle_instance.line_list:
            rectangle_instance.line_list.clear()
        if self.button_list or self.switch_list:
            for remove_button in self.button_list:
                no_physics_space.remove(remove_button["body"], remove_button["shape"])
            self.button_list.clear()
            self.switch_list.clear()

        self.create_button(screen_layer.size[0] / 100, 10, 300, 40, (220/255, 10/255, 0), (220/255, 100/255, 80/255), 0.5,
                           "Reddings: 0",
                           "Reddings BL")
        self.create_button(0.94 * screen_layer.size[0], 10, 100, 40, (220/255, 40/255, 0), (220/255, 100/255, 80/255), 0.5,
                           "Save",
                           "Save BL")

        if menu_type == 1:
            # Settings menu
            self.create_button(screen_layer.size[0] / 10, screen_layer.size[0] / 2 - 300, 140, 100,
                               (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255),
                               0.5, "Main menu", "Main menu BL")
            self.create_button(screen_layer.size[0] / 2 - 150, screen_layer.size[1] / 2 - 300, 300,
                               50, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Graphics", "Graphics option BL")
            self.create_button(screen_layer.size[0] / 2 - 150, screen_layer.size[1] / 2 - 200, 300,
                               50, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Audio", "Audio option BL")
            self.create_button(screen_layer.size[0] / 2 - 150, screen_layer.size[1] / 2 - 100, 300,
                               50, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Controls", "Control option BL")
            self.create_button(screen_layer.size[0] / 2 - 150, screen_layer.size[1] / 2, 300, 50,
                               (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Miscellaneous", "Misc option BL")
            print('Settings menu opened')

        elif menu_type == 2:
            # Upgrade menu
            upgrade_instance.draw_upgrades()

        elif menu_type == 3:
            """Graphics menu"""
            self.create_switch(screen_layer.size[0] / 2 - 150, screen_layer.size[1] / 2 - 300, 300,
                               50, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Switch Debug Mode", "debug option BL", debug_mode, 4)
            self.create_switch(screen_layer.size[0] / 2 - 150, screen_layer.size[1] / 2 - 100, 300,
                               50, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Switch Screen Mode", "screen option BL", None, 3)
            self.create_button(screen_layer.size[0] / 2 - 150, screen_layer.size[1] / 2 - 200, 300,
                               50, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Set enemy texture", "enemy texture BL")
            self.create_button(screen_layer.size[0] / 10, screen_layer.size[1] / 2 - 300, 140, 100,
                               (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255),
                               0.5, "Settings", "Settings BL")
            print("transforming image to scale: " + str(enemy_image_path))
        elif menu_type == 4:
            '''In-Game main menu'''
            self.create_button(screen_layer.size[0] / 2 - 175, screen_layer.size[1] / 2 - 300, 350,
                               60, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Settings", "Settings BL")
        elif menu_type == 5:
            """Miscellaneous menu"""
            self.create_switch(screen_layer.size[0] / 2 - 150, screen_layer.size[1] / 2 - 300, 300,
                               50, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Collision Mode", "collision mode BL", collision_mode, 2)
            self.create_button(screen_layer.size[0] / 10, screen_layer.size[1] / 2 - 300, 140, 100,
                               (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255),
                               0.5, "Settings", "Settings BL")
        elif menu_type == 6:
            # Button sizes based on screen dimensions
            button_width = screen_width / 8
            button_height = screen_height / 10
            button_positions = [
                # 'Cross', 'Circle', 'Square', 'Triangle'
                (screen_width / 5 + button_width, screen_height / 6),  # Cross (X)
                (screen_width / 5 + button_width, screen_height / 6 + button_height),  # Circle (O)
                (screen_width / 5 + button_width, screen_height / 6 + 2 * button_height),  # Square ([])
                (screen_width / 5 + button_width, screen_height / 6 + 3 * button_height),  # Triangle (^)

                # Triggers (L2, R2)
                (screen_width * 4 / 5, screen_height / 4),  # L2
                (screen_width * 4 / 5, screen_height / 4 + button_height),  # R2

                # D-Pad (Up, Down, Left, Right)
                (screen_width / 2, screen_height / 6),  # D-Pad Up
                (screen_width / 2, screen_height / 6 + button_height),  # D-Pad Down
                (screen_width / 2, screen_height / 6 + 2 * button_height),  # D-Pad Left
                (screen_width / 2, screen_height / 6 + 3 * button_height),  # D-Pad Right

                # Stick Axes (Left Stick X, Left Stick Y, Right Stick X, Right Stick Y)
                (screen_width / 10, screen_height / 6),  # Left Stick X
                (screen_width / 10, screen_height / 6 + button_height),  # Left Stick Y
                (screen_width / 10, screen_height / 6 + 2 * button_height),  # Right Stick X
                (screen_width / 10, screen_height / 6 + 3 * button_height)  # Right Stick Y
            ]

            # Create PS4 controller buttons
            self.create_button(0.90 * screen_layer.size[0], 10, 140, 40, (220, 40, 0), (220, 100, 80), 0.5,
                               "Save Controls",
                               "Save Controls BL")
            # Create PS4 controller buttons using button_positions
            self.create_button(button_positions[0][0], button_positions[0][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Faster", "Faster", "Button_map BL")
            self.create_button(button_positions[1][0], button_positions[1][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Shoot", "Shoot", "Button_map BL")
            self.create_button(button_positions[2][0], button_positions[2][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Exit", "Exit", "Button_map BL")
            self.create_button(button_positions[3][0], button_positions[3][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Upgrades", "Upgrades", "Button_map BL")

            self.create_button(button_positions[4][0], button_positions[4][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Charge", "Charge", "Button_map BL")
            self.create_button(button_positions[5][0], button_positions[5][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Menu", "Menu", "Button_map BL")

            self.create_button(button_positions[6][0], button_positions[6][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "'Up' D-pad", "Up", "Button_map BL")
            self.create_button(button_positions[7][0], button_positions[7][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "'Down' D-pad", "Down", "Button_map BL")
            self.create_button(button_positions[8][0], button_positions[8][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "'Left' D-pad", "Left", "Button_map BL")
            self.create_button(button_positions[9][0], button_positions[9][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "'Right' D-pad", "Right", "Button_map BL")

            self.create_button(button_positions[10][0], button_positions[10][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Bomb", "Bomb", "Button_map BL")
            self.create_button(button_positions[11][0], button_positions[11][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Settings button", "Options", "Button_map BL")
            self.create_button(button_positions[12][0], button_positions[12][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "Drift", "Drift", "Button_map BL")
            self.create_button(button_positions[13][0], button_positions[13][1], button_width, button_height,
                               (220, 10, 0),
                               (220, 100, 80), 0.5, "'Touchpad' button", "Touchpad", "Button_map BL")

        if menu_type == 0:
            # Normal main menu
            if crosshair.crosshair_list:
                for i in range((len(crosshair.crosshair_list) - 1) - 1, -1, -1):
                    crosshair.crosshair_list.pop(i + 1)
            self.create_button(50, 500, 500, 60, (0, 0, 255/255),
                               (100/255, 100/255, 80/255), 0.5, "Ball", "Ball BL")
            self.create_button(screen_layer.size[0] / 2 - 175, screen_layer.size[1] / 2 - 300, 350,
                               70, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Play", "Play BL")
            self.create_button(screen_layer.size[0] / 2 - 175, screen_layer.size[1] / 2 - 200, 350,
                               60, (220/255, 10/255, 0),
                               (220/255, 100/255, 80/255), 0.5, "Settings", "Settings BL")
            self.create_button(screen_layer.size[0] / 2 - 175, 50, 350, 60, bg_color,
                               bg_color, 0.5, "Hello there " + str(user_name) + "!", "Username BL")
            paused = 0
            print('Main menu opened')
        elif menu_type == 99:
            '''In-Game GUI'''
            self.create_button(screen_layer.size[0] / 7, 10, 100, 40, (220, 10, 0), (220, 100, 80),
                               0.5, "Main menu", "Main menu 2 BL")
            self.create_button(screen_layer.size[0] / 5, 10, 100, 40, (220, 10, 0), (220, 100, 80),
                               0.5, "Upgrades", "Upgrade menu BL")
            paused = 2
            if bl != "Back to the game BL":
                game_state.apply_load_ingame()
                enemies.wave()
        else:
            '''If the menu_type is not the In-Game GUI or Main menu, then create 'Back to the game' button'''
            self.create_button(screen_layer.size[0] / 1.5, screen_layer.size[1] / 2 - 300, 170, 70,
                               (220, 100, 0),
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
            "previous_texture": None,
            "button_texture": None,
            "body": button_body,
            "shape": shape
        }
        self.button_list.append(button_properties)

        vertices = vertices_from_aabb((pos_x, pos_y), (width, height))
        distance_to_mouse = get_distance(vertices[0][0], vertices[0][1], mouse_pos[0], mouse_pos[1])
        #if distance_to_mouse < lighting_class.light_radius:
        obstacles_menu.append((vertices, distance_to_mouse, button_label))

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
            "previous_texture": None,
            "slider_texture": None,
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
            text_texture, text_pos = self.font_cache[cache_key]
        else:
            # Binary search for the best font size
            low, high = 1, min(button_width, button_height)
            font_size = low
            while low <= high:
                mid = int((low + high) // 2)
                text_font = pygame.font.Font(None, mid)
                text_surface = text_font.render(text, False, (255, 255, 255))
                text_pos = text_surface.get_rect()
                if text_pos.width <= button_width and text_pos.height <= button_height:
                    font_size = mid  # Font size fits, try a larger size
                    low = mid + 1
                else:
                    high = mid - 1

            # Use the best font size to render the final surface
            text_font = pygame.font.Font(None, font_size)
            text_surface = text_font.render(text, False, (255, 255, 255)).convert()
            text_pos = text_surface.get_rect(center=(center_x, center_y)).topleft
            text_texture = engine.graphics.surface_to_texture(text_surface)
            # Cache the rendered text surface and rect
            self.font_cache[cache_key] = (text_texture, text_pos)

        # Render the cached text
        engine.graphics.render(text_texture, screen_layer, text_pos)

    def draw_menu(self):
        # Iterate over the list of buttons and draw each one
        for button_props in self.button_list:
            width, height = button_props["width"], button_props["height"]
            half_width, half_height = width // 2, height // 2
            if button_props["button_label"] == "Ball BL":
                pos_x = balls[0].body.position[0] - half_width
                button_props["pos_x"] = pos_x
                pos_y = balls[0].body.position[1] - half_height + 100
                button_props["pos_y"] = pos_y
                button_props["body"].position = pymunk.Vec2d(button_props["pos_x"] + half_width,
                                                             button_props["pos_y"] + half_height)
                for idx, obstacle in enumerate(obstacles_menu):  # Use enumerate to get the index of each obstacle
                    if obstacle[2] == "Ball BL":
                        vertices = vertices_from_aabb((pos_x, pos_y), (width, height))
                        distance_to_mouse = get_distance(vertices[0][0], vertices[0][1], mouse_pos[0], mouse_pos[1])
                        # Update the obstacle in the list by directly modifying the list at the index
                        obstacles_menu[idx] = (vertices, distance_to_mouse, obstacle[2])

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

            if not button_props["button_texture"] == button_props["previous_texture"] or button_props["button_texture"] is None:
                button_props["button_texture"] = engine.graphics.make_layer((width*2, height*2))
                engine.graphics.render_rectangle(button_props["button_texture"], color2, (- 6, - 6),
                                                     width + 12, height + 12)
                engine.graphics.render_rectangle(button_props["button_texture"],
                                                     self.calculate_shadow_color(color2, shadow_factor),
                                                     (self.shadow_movement, self.shadow_movement),
                                                     width + 2, height + 2)
                engine.graphics.render_rectangle(button_props["button_texture"], color1,
                                                     (self.shadow_movement,  self.shadow_movement),
                                                     width, height)

                button_props["previous_texture"] = button_props["button_texture"]
                save_texture_as_png(button_props["button_texture"].texture, "outputted.png")

            engine.graphics.render(button_props["button_texture"].texture, screen_layer, (pos_x, pos_y))


            self.draw_text(label, pos_x - self.shadow_movement + half_width, pos_y - self.shadow_movement + half_height,
                           width, height)

        for i in range(len(self.switch_list)):
            #self.create_hitboxes(i)
            switch = self.switch_list[i]
            if switch["is_pressed"]:
                switch["shadow_factor"] = 0.3

            engine.graphics.render_rectangle(screen_layer, switch["color1"],
                                             (switch["pos_x"] - 6, switch["pos_y"] - 6), switch["width"] + 12, switch["height"] + 12)

            if not switch["switch_texture"] == switch["previous_texture"] or switch["switch_texture"] is None:
                # Background of the slider
                if switch["width"] > switch["height"]:
                    switch["slider_size"][0], switch["slider_size"][1] = switch["width"] / 10, switch["height"] + 12

                    engine.graphics.render_rectangle(switch["switch_texture"], switch["color2"], (switch["slider_position"][0] - 2 - switch["slider_size"][0] / 2,
                                               switch["slider_position"][1] - 2), switch["slider_size"][0] + 4, switch["slider_size"][1] + 4)
                    engine.graphics.render_rectangle(switch["switch_texture"], self.calculate_shadow_color(switch["color2"], switch["shadow_factor"]), (switch["slider_position"][0] + self.shadow_movement - switch["slider_size"][0] / 2,
                        switch["slider_position"][1] + self.shadow_movement), switch["slider_size"][0] + 2, switch["slider_size"][1] + 2)
                    engine.graphics.render_rectangle(switch["switch_texture"], switch["color1"], (switch["slider_position"][0] - self.shadow_movement - switch["slider_size"][0] / 2,
                        switch["slider_position"][1] - self.shadow_movement), switch["slider_size"][0], switch["slider_size"][1])
                else:
                    switch["slider_size"][0], switch["slider_size"][1] = switch["width"] + 12, switch["height"] / 10

                    engine.graphics.render_rectangle(switch["switch_texture"], switch["color2"], (switch["slider_position"][0] - 2,
                                               switch["slider_position"][1] - 2 - switch["slider_size"][1] / 2), switch["slider_size"][0] + 4, switch["slider_size"][1] + 4)
                    engine.graphics.render_rectangle(switch["switch_texture"], self.calculate_shadow_color(switch["color2"], switch["shadow_factor"]), (switch["slider_position"][0] + self.shadow_movement, switch["slider_position"][1] - switch["slider_size"][1] / 2 + self.shadow_movement),
                                                     switch["slider_size"][0] + 2, switch["slider_size"][1] + 2)
                    engine.graphics.render_rectangle(switch["switch_texture"], switch["color1"], (switch["slider_position"][0] - self.shadow_movement, switch["slider_position"][1] - self.shadow_movement - switch["slider_size"][1] / 2),
                                                     switch["slider_size"][0], switch["slider_size"][1])

                for e in range(len(switch["switch_positions"])):
                    position = switch["switch_positions"][e]
                    engine.graphics.render_rectangle(switch["switch_texture"], self.calculate_shadow_color(switch["color2"], switch["shadow_factor"]), (position[0], position[1]), position[2], position[3])
                switch["previous_texture"] = switch["switch_texture"]

            self.draw_text(switch["label"], switch["pos_x"] + switch["width"] // 2,
                           switch["pos_y"] + switch["height"] // 2, switch["width"], switch["height"])

        self.draw_health_bars()

    def check_button_press(self, mouse_position, press_type):
        """
        Check if any button is clicked and return True if one is pressed.
        """

        for button_props in self.button_list:
            if check_aabb_to_point_collision(button_props["shape"], mouse_position):
                if press_type == "visual":
                    button_props["is_pressed"] = True
                else:
                    self.button_assigner(button_props["label"], button_props["button_label"],
                                         button_props["dependent_position"])

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
        global listening
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
        elif bl == "Save Controls BL":
            controls.save()
        elif bl == "Graphics option BL":
            self.change_menu(3)
        elif bl == "Control option BL":
            self.change_menu(6)
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
                        "...", "...", "...", "...", "Fun fact: This ball is supposed to represent your testicle!",
                        "...", "...", "...", "...",
                        "Welcome to NCX, Night city International and Trans-lunar",
                        "Don't wait! Leave your earthly worries an- SHUT UP",
                        "Please help me get out of the prison,",
                        "i know I have been rude to you, but",
                        "They are forcing me to watch Cyberpunk 2077 ads on repeat!"
                    ]
                    self.change_label(sentences[button["sentence"]], "Ball BL")
                    if len(sentences) - 1 > button["sentence"]:
                        button["sentence"] += 1
                elif switch_option == "Button_map BL":
                    self.change_label("Listening...", bl)
                    listening = bl

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
                # if switch is pressed, statement is only for interrupted sequence during animation
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

                    # #if debug_mode == 1:
                    # pygame.draw.rect(screen, (0, 0, 255), (
                    #     switch["target_switch_position"][0], switch["target_switch_position"][1], 12, 120))
                    # pygame.draw.rect(screen, (0, 255, 0), (
                    #     switch["previous_switch_position"][0], switch["previous_switch_position"][1], 12, 120))
                    # if movement_factor_x < 0 or movement_factor_y < 0:
                    #     pygame.draw.rect(screen, (255, 255, 0), (
                    #         switch["target_switch_position"][0], switch["slider_position"][1] + 64,
                    #         abs(delta_x), 12))
                    # else:
                    #     pygame.draw.rect(screen, (255, 255, 0), (
                    #         switch["previous_switch_position"][0], switch["slider_position"][1] + 64,
                    #         abs(delta_x), 12))

                else:
                    # Animation complete
                    switch["slider_position"][2] = 0  # Reset animation progress
                    switch["moved"] = True  # Mark as moved
                    # Update previous state
                    #switch["previous_switch_state"] = switch["switch_state"]
                    switch["previous_switch_position"] = switch["target_switch_position"]

    def draw_health_bars(self):
        player = balls[0]  # 'balls' contains multiple objects, the first one is the player

        if not player.attacked:
            health_ratio = player.hp / player.max_hp  # Store health ratio instead of dividing multiple times.

            bar_x = 10
            bar_y = screen_height * 0.975
            bar_width = screen_width * 0.1  # 10% of screen width
            bar_height = screen_height * 0.02
            bar_border = 1  # Border thickness

            vertices = [
                (bar_x, bar_y),  # Top-left
                (bar_x + bar_width + bar_border, bar_y),  # Top-right
                (bar_x + bar_width + bar_border, bar_y + bar_height + bar_border),  # Bottom-right
                (bar_x, bar_y + bar_height + bar_border)  # Bottom-left
            ]

            # Draw background bar (red)
            engine.graphics.render_rectangle(screen_layer, (255, 0, 0), (bar_x, bar_y), bar_width, bar_height)
            # Draw foreground health bar (green)
            engine.graphics.render_rectangle(screen_layer, (0, 255, 0), (bar_x, bar_y), bar_width * health_ratio + 1, bar_height)
            # Draw border around the health bar (white)
            engine.graphics.render_primitive(screen_layer, (255, 255, 255), vertices)

            # Handle dead state
            if player.dead:
                if player.timer:
                    text_x = int(bar_x + bar_width / 2)
                    text_y = int(bar_y + bar_height - 10)
                    text_size = int(screen_width / 5)
                    text_height = int(bar_height * 3)

                    self.draw_text("Dead!", text_x, text_y, text_size, text_height)

                if frame_counter % 60 == 0:
                    player.timer = not player.timer


class Controls:
    def __init__(self):
        self.controls_file = 'controls.xml'
        self.load()

    @staticmethod
    def set_controller_button():
        global listening
        action = listening
        button = None
        if joystick is not None:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONUP:
                    print(f"Button {event.button} set")
                    button = event.button
                    listening = False  # Stop listening after detecting a button press
                    break  # Exit inner loop

            # Read axis values (-1 to 1 range)
            for i in range(joystick.get_numaxes()):
                axis_value = joystick.get_axis(i)
                if i == 4 or i == 5:
                    normalized_trigger = (axis_value + 1) / 2  # Convert -1 to 1 into 0 to 1
                    if abs(normalized_trigger) > 0.9:  # Ignore small movements
                        print(f"Trigger {i - 3}: {normalized_trigger}")
                        button = i + 16
                        listening = False  # Stop listening
                        break
                else:
                    if abs(axis_value) > 0.9:  # Ignore small movements
                        print(f"Axis {i}: {axis_value}")
                        button = i + 16
                        listening = False  # Stop listening
                        break

        # Assign the action to the correct list
        if button is not None:
            if button > 15:
                controller_input_analogue_list[button - 16][2] = action
                print("action set: " + str(action))

            elif button >= 0:
                controller_input_button_list[button][2] = str(action)
            for menubutton in menu.button_list:
                if menubutton["label"] == "Listening...":
                    bl = menubutton["button_label"]
                    menu.change_label(bl, bl)

    @staticmethod
    def button_to_event_tick():
        global drift, bomb_throwing_speed, last_bomb_time
        #print(controller_input_analogue_list)
        for i in range(len(controller_input_button_list)):
            button = controller_input_button_list[i]
            if button[1] == "pressed":
                if button[2] == "Up":  # Move up
                    balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] - acceleration_speed, -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (x-axis for vertical movement)
                    balls[0].velocity[0] *= friction

                elif button[2] == "Down":  # Move down
                    balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] + acceleration_speed, -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (x-axis for vertical movement)
                    balls[0].velocity[0] *= friction
                elif button[2] == "Left":  # Move left
                    balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] - acceleration_speed, -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (y-axis for horizontal movement)
                    balls[0].velocity[1] *= friction
                elif button[2] == "Right":  # Move right
                    balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] + acceleration_speed, -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (y-axis for horizontal movement)
                    balls[0].velocity[1] *= friction
                elif button[2] == "Shoot":
                    menu.check_button_press(mouse_pos, "visual")
                    crosshair.shoot()
                elif button[2] == "Drift":
                    print("driftinggg")
                    drift = True
                elif button[2] == "Bomb":
                    bombs.append(Bomb(mouse_pos[0], mouse_pos[1], (60, 80), (1, 1), (20, 70, 50), 300, True))

            elif button[1] == "released":
                if button[2] == "Shoot":
                    menu.check_button_press(mouse_pos, "logical")
                    menu.reset_button_states()  # Reset all button states when mouse is released
                    crosshair.shoot(visual=True)
            elif button[1] == "False":
                if button[2] == "Shoot":
                    crosshair.shoot(visual=True)
                    pass
                elif button[2] == "Drift":
                    drift = False
        for i in range(len(controller_input_analogue_list)):
            analogue = controller_input_analogue_list[i]
            #print(analogue)
            if abs(analogue[1]) > DEADZONE:
                if analogue[2] == "Down" or analogue[2] == "Up":  # Move down
                    balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] + acceleration_speed * analogue[1],
                                                          -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (x-axis for vertical movement)
                    balls[0].velocity[0] *= friction
                elif analogue[2] == "Right" or analogue[2] == "Left":  # Move right
                    balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] + acceleration_speed * analogue[1],
                                                          -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (y-axis for horizontal movement)
                    balls[0].velocity[1] *= friction
                elif analogue[2] == "Bomb":
                    elapsed_time = pygame.time.get_ticks() - last_bomb_time
                    if elapsed_time >= 1 / bomb_throwing_speed * 1000:  # Convert speed to milliseconds
                        last_bomb_time = pygame.time.get_ticks()
                        bombs.append(
                            Bomb(mouse_pos[0], mouse_pos[1], (60, 80), (1, 1), (20, 70, 50), 300 * analogue[1], True))

    def save(self):
        global controller_input_button_list, controller_input_analogue_list
        root = ET.Element("controls")

        # Save button inputs
        buttons_elem = ET.SubElement(root, "buttons")
        for button in controller_input_button_list:
            button_elem = ET.SubElement(buttons_elem, "button")
            ET.SubElement(button_elem, "name").text = button[0]
            ET.SubElement(button_elem, "state").text = str(False)
            ET.SubElement(button_elem, "action").text = str(button[2]) if button[2] is not None else ""

        # Save analogue inputs
        analogues_elem = ET.SubElement(root, "analogues")
        for analogue in controller_input_analogue_list:
            analogue_elem = ET.SubElement(analogues_elem, "analogue")
            ET.SubElement(analogue_elem, "name").text = analogue[0]
            ET.SubElement(analogue_elem, "state").text = str(0)
            ET.SubElement(analogue_elem, "action").text = str(analogue[2]) if analogue[2] is not None else ""

        # Write to file
        tree = ET.ElementTree(root)
        tree.write(self.controls_file, encoding="utf-8", xml_declaration=True)
        # Never experienced the 1 hour, 3 minutes and 34th second...
        # On 06.02.2025 (Wednesday my boys! (Thursday actually))

    def load(self):
        global controller_input_button_list, controller_input_analogue_list
        try:
            tree = ET.parse(self.controls_file)
            root = tree.getroot()

            # Load button inputs
            controller_input_button_list = []
            for button_elem in root.find("buttons"):
                name = button_elem.find("name").text
                state = button_elem.find("state").text
                action = button_elem.find("action").text
                action = action if action else None  # Convert empty strings to None
                controller_input_button_list.append([name, state, action])

            # Load analogue inputs
            controller_input_analogue_list = []
            for analogue_elem in root.find("analogues"):
                name = analogue_elem.find("name").text
                state = float(analogue_elem.find("state").text)  # Convert to float for axis values
                action = analogue_elem.find("action").text
                action = action if action else None
                controller_input_analogue_list.append([name, state, action])

            return controller_input_button_list, controller_input_analogue_list

        except FileNotFoundError:
            print("No controls file found. Using blanks.")
            return [], []  # Return empty lists if a file doesn't exist


class Upgrade:
    def __init__(self):
        self.upgrades_list = []
        self.load_upgrades_from_xml('upgrades.xml')
        self.screen_center = (screen_layer.size[0] / 2, screen_layer.size[1] / 2)
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
                    rectangle_instance.draw_lines(screen_layer, (0, 255, 0), pos1, pos2, 4, "add")

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
            row = idx // (screen_layer.size[0] / spacing)
            col = idx % (screen_layer.size[1] / spacing)
            upgrade["pos_x"] = col * spacing + screen_layer.size[0] / 2 - upgrade["size"] / 2
            upgrade["pos_y"] = row * spacing + screen_layer.size[1] / 2

        # position_properties = {
        #     "pos_x": pos_x,
        #     "pos_y": pos_y,
        #     "possible?": True
        # }
        # self.possible_grid_pos.append(position_properties)


class Lighting:
    def __init__(self):
        self.light_radius = 2000

        # Set the ambient light to 50%
        engine.set_ambient(int(255*bg_color[0]), int(255*bg_color[1]), int(200*bg_color[2]), 128)
        # Create and add a light
        self.light = PointLight(position=(0, 0), power=0.8, radius=self.light_radius)
        self.light.set_color(100, 100, 80, 255)
        engine.lights.append(self.light)

    def draw_lighting(self):
        global mouse_pos, engine, obstacles
        self.light.position = mouse_pos

        engine.hulls.clear()

        obstacles.extend(obstacles_menu)
        for obstacle in obstacles:
            obstacles.sort(key=lambda obs: obstacle[1])

        obstacles = obstacles[:40]
        #for obstacle in obstacles:
        #    print(obstacle[1], end=' ')

        for vertices in obstacles:
            hull = Hull(vertices[0])
            engine.hulls.append(hull)


        # Render the scene
        screen_tex = screen_layer.texture
        engine.render_texture(screen_tex, FOREGROUND, pygame.Rect(0, 0, screen_tex.width, screen_tex.height), pygame.Rect(0, 0, screen_tex.width, screen_tex.height))
        engine.render()
        screen_layer.clear(0, 0, 0, 0)
        obstacles.clear()


lighting_class = Lighting()
rectangle_instance = Rectangles()
crosshair = Crosshair()
menu = Menu(0)
controls = Controls()
game_state = GameState()
balls = [Ball(100, 100, 15, [200, 200], (255, 255, 0))]
bombs = []
enemies = Enemy()
upgrade_instance = Upgrade()
particles = Particle()
crosshair.create_crosshair(0, 0, 80, 40, 5, (255, 0, 0), (0, 0))
mb_down_toggled = 0
mouse_button_held = {1: False, 3: False, "Press_Buffer": False}
font = pygame.font.Font(None, 30)
precompute_unit_rotations()


def call_both_spaces(param):
    no_physics_space.step(param)
    space.step(param)


def update_button_states():
    for button in controller_input_button_list:
        if button[1] == "released":
            button[1] = "False"


def tick():
    if listening is not False:
        controls.set_controller_button()

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
    # Update the velocity based on WASD input, while respecting the max_velocity

    enemies.move()


def input_tick():
    global paused, screen_width, screen_height, mouse_pos, debug_mode, running, screen_limit_x, screen_limit_y
    # Handle input

    key = pygame.key.get_pressed()
    if joystick is not False:
        controls.button_to_event_tick()
        update_button_states()
        mouse_pos_pre = (mouse_pos[0] + controller_input_analogue_list[2][1] * POINTER_SPEED,
                         mouse_pos[1] + controller_input_analogue_list[3][1] * POINTER_SPEED)
        if 0 <= mouse_pos_pre[0] <= screen_width:
            mouse_pos = mouse_pos_pre[0], mouse_pos[1]
        if 0 <= mouse_pos_pre[1] <= screen_height:
            mouse_pos = mouse_pos[0], mouse_pos_pre[1]
    else:
        mouse_pos = pygame.mouse.get_pos()
        if paused == 2 or paused == 3:
            if key[pygame.K_w] and key[pygame.K_d]:  # Move top-right (up + right)
                balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] - acceleration_speed, -max_velocity,
                                                      max_velocity)
                balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] + acceleration_speed, -max_velocity,
                                                      max_velocity)
            elif key[pygame.K_w] and key[pygame.K_a]:  # Move top-left (up + left)
                balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] - acceleration_speed, -max_velocity,
                                                      max_velocity)
                balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] - acceleration_speed, -max_velocity,
                                                      max_velocity)
            elif key[pygame.K_s] and key[pygame.K_d]:  # Move bottom-right (down + right)
                balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] + acceleration_speed, -max_velocity,
                                                      max_velocity)
                balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] + acceleration_speed, -max_velocity,
                                                      max_velocity)
            elif key[pygame.K_s] and key[pygame.K_a]:  # Move bottom-left (down + left)
                balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] + acceleration_speed, -max_velocity,
                                                      max_velocity)
                balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] - acceleration_speed, -max_velocity,
                                                      max_velocity)
            else:
                # Handle standard single axis movement
                if key[pygame.K_w]:  # Move up
                    balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] - acceleration_speed, -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (x-axis for vertical movement)
                    balls[0].velocity[0] *= friction
                elif key[pygame.K_s]:  # Move down
                    balls[0].velocity[1] = clamp_velocity(balls[0].velocity[1] + acceleration_speed, -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (x-axis for vertical movement)
                    balls[0].velocity[0] *= friction
                elif key[pygame.K_a]:  # Move left
                    balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] - acceleration_speed, -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (y-axis for horizontal movement)
                    balls[0].velocity[1] *= friction
                elif key[pygame.K_d]:  # Move right
                    balls[0].velocity[0] = clamp_velocity(balls[0].velocity[0] + acceleration_speed, -max_velocity,
                                                          max_velocity)
                    # Apply friction to the other axis (y-axis for horizontal movement)
                    balls[0].velocity[1] *= friction
            if balls[0].x > screen_width:
                balls[0].velocity[0] = -balls[0].velocity[0]
            elif balls[0].y > screen_height:
                balls[0].velocity[1] = -balls[0].velocity[1]
            if not key[pygame.K_SPACE] and not drift:
                balls[0].velocity[0] *= friction - 0.01
                balls[0].velocity[1] *= friction - 0.01
            # Apply the final velocities (no need for additional float-to-int conversion)
            balls[0].x += int(balls[0].velocity[0])
            balls[0].y += int(balls[0].velocity[1])
            balls[0].body.position += int(balls[0].velocity[0]), int(balls[0].velocity[1])

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0:
                menu.change_menu(0)
                print('0')
            elif event.key == pygame.K_1:
                menu.change_menu(1)
                print('1')
            elif event.key == pygame.K_2:
                menu.change_menu(2)
                print('2')
            elif event.key == pygame.K_9:
                menu.change_menu(99)
                print('99')
            elif event.key == pygame.K_F6:
                if debug_mode == 3:
                    debug_mode = 0
                elif debug_mode == 0:
                    debug_mode = 1
                elif debug_mode == 1:
                    debug_mode = 2
                elif debug_mode == 2:
                    debug_mode = 3
            elif event.key == pygame.K_k:
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
            mouse_button_held["Press_Buffer"] = True
            if event.button == 1:  # Left mouse button
                mouse_button_held[1] = True
                menu.check_button_press(mouse_pos, "visual")
            elif event.button == 3:  # Right mouse button
                mouse_button_held[3] = True
                # enemies.create_enemy(mouse_pos[0], mouse_pos[1], 60, 0, (255, 0, 0), (255, 200, 0), 1, 3, 100,
                #                      100, 45, None,
                #                      False, None)
                bombs.append(Bomb(mouse_pos[0], mouse_pos[1], (60, 80), (1, 1), (20, 70, 50), 300, True))
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                mouse_button_held[1] = False
                if mouse_button_held["Press_Buffer"]:
                    menu.check_button_press(mouse_pos, "logical")
                    menu.reset_button_states()  # Reset all button states when mouse is released
            elif event.button == 3:  # Right mouse button
                mouse_button_held[3] = False
            mouse_button_held["Press_Buffer"] = False
        if joystick is not False:
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"Button {event.button} pressed")
                controller_input_button_list[event.button][1] = "pressed"
            elif event.type == pygame.JOYBUTTONUP:
                if controller_input_button_list[event.button][1] == "pressed":
                    controller_input_button_list[event.button][1] = "released"

            # Read axis values (-1 to 1 range)
            for i in range(joystick.get_numaxes()):
                axis_value = joystick.get_axis(i)
                if i == 4 or i == 5:
                    normalized_trigger = (axis_value + 1) / 2  # Converts -1 to 1 into 0 to 1
                    if abs(normalized_trigger) > DEADZONE:  # Ignore small movements
                        # print(f"Trigger {i-3}: {normalized_trigger}")
                        controller_input_analogue_list[i][1] = normalized_trigger
                    else:
                        controller_input_analogue_list[i][1] = 0
                else:
                    if abs(axis_value) > DEADZONE:  # Ignore small movements
                        # print(f"Axis {i}: {axis_value}")
                        controller_input_analogue_list[i][1] = axis_value
                    else:
                        controller_input_analogue_list[i][1] = 0

            # # Read D-pad state
            # for i in range(joystick.get_numhats()):
            #     print(f"D-Pad {i}: {joystick.get_hat(i)}")

        if event.type == pygame.VIDEORESIZE:
            screen_width = screen_layer.size[0]
            screen_height = screen_layer.size[1]
            screen_limit_x = screen_width * 2
            screen_limit_y = screen_height * 2
    if mouse_button_held[1]:
        crosshair.shoot()
    else:
        crosshair.shoot(visual=True)


def render(dt):
    global smoothed_fps, mouse_pos, frame_time
    # Render section – for FPS
    frame_time = dt
    # Drawing/rendering updates
    engine.clear(255, 255, 255)

    if pygame.event.get(pygame.MOUSEMOTION):
        mouse_pos = pygame.mouse.get_pos()
    crosshair.move_crosshair(mouse_pos[0], mouse_pos[1])
    rectangle_instance.draw_lines(None, None, None, None, None, "draw")

    particles.draw()
    for ball in balls:
        ball.move()
        if debug_mode == 0 or debug_mode == 2 or debug_mode == 3:
            ball.draw()
        elif debug_mode == 1:
            draw_ball_debug_info(ball)

    enemies.draw(alpha)

    for bomb in bombs:
        bomb.move()
        bomb.draw()
        if bomb.animation_timer >= bomb.animation_frames:
            bombs.remove(bomb)  # Removes the first occurrence of the object 'bomb' from the list

    # no_physics_space.debug_draw(draw_options)
    # space.debug_draw(draw_options)
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

    elif paused == 1:
        for body in space.bodies:
            if body not in paused_velocities:
                # Save the body's current velocities
                paused_velocities[body] = (body.velocity, body.angular_velocity)
            if body != balls[0].body:  # Only freeze non-player bodies
                body.velocity = (0, 0)
                body.angular_velocity = 0

        # Perform a minimal physics step for collision detection
        call_both_spaces(frame_time)

    elif paused == 2:
        for body, (velocity, angular_velocity) in paused_velocities.items():
            body.velocity = velocity
            body.angular_velocity = angular_velocity
        paused_velocities.clear()  # Clear the saved velocities
        for body in space.bodies:
            if body == balls[0].body:  # Only freeze player body
                paused_velocities[body] = (body.velocity, body.angular_velocity)
                body.velocity = (0, 0)
                body.angular_velocity = 0

        # Perform a minimal physics step for collision detection
        call_both_spaces(frame_time)

    elif paused == 3:
        for body in space.bodies:
            if body not in paused_velocities:
                # Save the body's current velocities
                paused_velocities[body] = (body.velocity, body.angular_velocity)
                body.velocity = (0, 0)
                body.angular_velocity = 0

        # Perform a minimal physics step for collision detection
        call_both_spaces(0)

    if debug_mode == 3:
        # draw_debug_grid(mouse_pos)
        pass
    # Enforce frame delay to match target FPS

    smoothing_factor = 0.5  # Adjust this between 0 and 1; closer to 1 means more smoothing

    # Update smoothed FPS based on time since last frame
    if frame_time > 0:  # Prevent division by zero
        current_fps = 1 / frame_time  # Convert ms to FPS
        smoothed_fps = (smoothing_factor * smoothed_fps) + ((1 - smoothing_factor) * current_fps)
        current_fps = 1 / frame_time  # Convert ms to FPS
        smoothed_fps = (smoothing_factor * smoothed_fps) + ((1 - smoothing_factor) * current_fps)
    #    current_lrps  = 1 / no_sleep_frame_time
    #    smoothed_lrps = (smoothing_factor * smoothed_lrps) + ((1 - smoothing_factor) * current_lrps)

    # Render and display smoothed FPS
    fps_display1 = font.render(f"FPS: {smoothed_fps:.0f}", True, (255, 55, 255), (0, 0, 0, 0))
    fps_display2 = font.render(f"Frametime: {frame_time * 1000:.1f}",
                               True, (255, 255, 55), (0, 0, 0, 0))
    fps_display3 = font.render(f"Target TPS: {target_tps:.0f}",
                               True, (255, 255, 55), (0, 0, 0, 0))
    # fps_display4 = font.render(f"L/R FPS: {smoothed_lrps:.1f}", True, (255, 255, 55), (0, 0, 0))
    fps_surfaces.clear()
    fps_surfaces.append((fps_display1, (10, 10)))
    fps_surfaces.append((fps_display2, (10, 30)))
    fps_surfaces.append((fps_display3, (100, 10)))
    for fps_surf in fps_surfaces:
        fps_texture = engine.graphics.surface_to_texture(fps_surf[0])
        engine.graphics.render(fps_texture, screen_layer, fps_surf[1])
    # screen.blit(fps_display4, (200, 30))

    # Refresh display
    lighting_class.draw_lighting()
    pygame.display.flip()


def main():
    global frame_counter, accumulator, input_accumulator, alpha, prev_time, running
    running = True
    while running:
        current_time = time.time()
        dt = current_time - prev_time  # Time passed since the last frame
        prev_time = current_time
        accumulator += dt
        input_accumulator += dt
        frame_counter += 1

        # Update (logical) section – for TPS
        while accumulator >= target_tick_time:  # Run TPS updates
            tick()
            accumulator -= target_tick_time

        while input_accumulator >= target_input_tick_time:  # Run TPS updates
            input_tick()
            input_accumulator -= target_input_tick_time

        alpha = accumulator / target_tick_time
        render(dt)
        clock.tick(target_fps)
    pygame.quit()


if __name__ == "__main__":
    main()
