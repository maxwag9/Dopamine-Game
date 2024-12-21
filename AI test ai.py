import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

epoch = 0

# Define the neural network for the AI ball
class AIBallNet(nn.Module):
    def __init__(self):
        super(AIBallNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),  # Input: 10 features (position, distances, rotations)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # Output: 2 features (vel_x, vel_y)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the AI ball network and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AIBallNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training data (mock data for demonstration purposes)
# Each sample: [ball_x, ball_y, dist_enemy1, rot_enemy1, dist_enemy2, rot_enemy2, dist_enemy3, rot_enemy3, wall_x, wall_y]
data = np.array([
    [50, 50, 30, 0.1, 40, 0.3, 20, -0.2, 10, 90],
    [30, 70, 20, -0.5, 50, 0.2, 10, 0.4, 5, 80],
    # Add more samples here
])
labels = np.array([
    [1, -1],  # vel_x, vel_y
    [0, 1],
    # Corresponding velocities for above samples
])

# Convert data to PyTorch tensors
inputs = torch.tensor(data, dtype=torch.float32).to(device)
targets = torch.tensor(labels, dtype=torch.float32).to(device)

# Training loop
def train_ai_ball():
    global epoch
    epoch += 1
    net.train()
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{69420}], Loss: {loss.item():.4f}")


train_ai_ball()

# AI ball movement logic in the game
def ai_ball_update(ball_x, ball_y, distances, rotations, wall_dists):
    # Prepare input for the network
    input_data = torch.tensor([
        ball_x, ball_y,
        *distances, *rotations,
        *wall_dists
    ], dtype=torch.float32).unsqueeze(0).to(device)

    net.eval()
    with torch.no_grad():
        output = net(input_data)

    vel_x, vel_y = output[0].cpu().numpy()
    return vel_x, vel_y

# Example usage in your game loop
# ball_x, ball_y = current ball position
distances = [30, 40, 20]  # Distances to 3 closest enemies
rotations = [0.1, 0.3, -0.2]  # Rotations of those enemies
wall_dists = [10, 90]  # Distances to the 4 walls (mocked as 2 here for simplicity)
vel_x, vel_y = ai_ball_update(50, 50, distances, rotations, wall_dists)
print(f"AI Ball Velocity: ({vel_x:.2f}, {vel_y:.2f})")
