import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import string
import time

plt.ion()
plt.subplots_adjust(bottom=0.12)

MAX_POP = 120
TIME_LIMIT = 60
START_TIME = time.time()

def generate_name():
    return "A_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

class Brain:
    def __init__(self):
        self.weights = np.random.randn(3, 2)

    def forward(self, inputs):
        return np.tanh(np.dot(inputs, self.weights))

    def mutate(self):
        self.weights += np.random.randn(3, 2) * 0.1

class Creature:
    def __init__(self):
        self.name = generate_name()
        self.x = np.random.rand()
        self.y = np.random.rand()
        self.energy = 1.0
        self.brain = Brain()
        self.speed = 0.03

    def move(self, food_list, obstacles):
        vision_radius = 0.25

        visible_food = [
            f for f in food_list
            if np.linalg.norm([self.x - f[0], self.y - f[1]]) < vision_radius
        ]

        if visible_food:
            nearest = min(
                visible_food,
                key=lambda f: np.linalg.norm([self.x - f[0], self.y - f[1]])
            )

            dx = nearest[0] - self.x
            dy = nearest[1] - self.y
            dist = np.linalg.norm([dx, dy]) + 1e-6

            inputs = np.array([dx/dist, dy/dist, self.energy])
            output = self.brain.forward(inputs)

            self.x += output[0] * self.speed
            self.y += output[1] * self.speed
        else:
            self.x += np.random.uniform(-0.02, 0.02)
            self.y += np.random.uniform(-0.02, 0.02)

        self.energy -= 0.04

        self.x = np.clip(self.x, 0, 1)
        self.y = np.clip(self.y, 0, 1)

        for ox, oy in obstacles:
            if np.linalg.norm([self.x - ox, self.y - oy]) < 0.06:
                self.energy -= 0.3

    def eat(self, food_list):
        for i, f in enumerate(food_list):
            if np.linalg.norm([self.x - f[0], self.y - f[1]]) < 0.04:
                self.energy += 0.9
                del food_list[i]
                break

def reproduce(parent):
    child = Creature()
    child.x = parent.x
    child.y = parent.y
    child.brain.weights = np.copy(parent.brain.weights)
    child.brain.mutate()
    child.name = generate_name()
    return child

# ---------- INITIAL SETUP ----------
obstacles = [(0.3, 0.3), (0.7, 0.7), (0.5, 0.2)]
creatures = [Creature() for _ in range(50)]
food = [np.random.rand(2) for _ in range(40)]

dead_count = 0
score = 0
winner = None
winner_energy = 0

# ---------- MAIN LOOP ----------
while True:
    plt.clf()
    ax = plt.gca()
    ax.set_facecolor('#0b0f1a')
    ax.set_aspect('equal')

    # FOOD
    for f in food:
        size = 0.015
        diamond = patches.Polygon([
            (f[0], f[1] + size),
            (f[0] + size, f[1]),
            (f[0], f[1] - size),
            (f[0] - size, f[1])
        ], color='#FFD700')
        ax.add_patch(diamond)

    # OBSTACLES
    for o in obstacles:
        body = plt.Circle(o, 0.05, color='#2b0000')
        border = plt.Circle(o, 0.05, fill=False, edgecolor='#ff3b3b', linewidth=3)
        ax.add_patch(body)
        ax.add_patch(border)

    new_creatures = []
    alive_creatures = []
    best_creature = None
    max_energy = 0

    for c in creatures:
        c.move(food, obstacles)
        c.eat(food)

        if c.energy > 2.2:
            new_creatures.append(reproduce(c))
            c.energy -= 1.0

        if c.energy > 0:
            alive_creatures.append(c)

            if c.energy > max_energy:
                max_energy = c.energy
                best_creature = c

            energy_ratio = min(c.energy / 2.5, 1)
            color = (
                0.2 + 0.5 * energy_ratio,
                0.4 + 0.5 * (1 - energy_ratio),
                1.0
            )

            size = 0.02 + c.energy * 0.01
            body = plt.Circle((c.x, c.y), size, color=color)
            ax.add_patch(body)
        else:
            dead_count += 1

    creatures = alive_creatures + new_creatures

    if len(creatures) > MAX_POP:
        creatures = sorted(creatures, key=lambda c: c.energy, reverse=True)[:MAX_POP]

    for _ in range(10):
        food.append(np.random.rand(2))

    score += 1
    alive_count = len(creatures)

    if best_creature:
        winner = best_creature
        winner_energy = max_energy

    # TOP AGENTS
    top_agents = sorted(creatures, key=lambda c: c.energy, reverse=True)[:3]
    for agent in top_agents:
        plt.text(agent.x, agent.y, agent.name, color='white', fontsize=6)

    # BEST AGENT HIGHLIGHT
    if best_creature:
        highlight = plt.Circle((best_creature.x, best_creature.y), 0.05,
                               fill=False, edgecolor='gold', linewidth=2)
        ax.add_patch(highlight)

    # ---------- CLEAN UI ----------
    plt.figtext(0.5, 0.08,
                f"Alive: {alive_count}    Dead: {dead_count}    Score: {score}",
                ha="center", fontsize=11, color="black")

    plt.figtext(0.5, 0.04,
                "● Agents     ◆ Food     ● Obstacles     ◎ Best Agent",
                ha="center", fontsize=10, color="black")

    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.pause(0.03)

    if time.time() - START_TIME > TIME_LIMIT or alive_count == 0:
        break

# ---------- WINNER SCREEN ----------
plt.clf()
ax = plt.gca()
ax.set_facecolor('#0b0f1a')

if winner:
    plt.text(0.3, 0.65, "🏆 WINNER", color='gold', fontsize=18)
    plt.text(0.3, 0.55, f"Name: {winner.name}", color='cyan', fontsize=12)
    plt.text(0.3, 0.48, f"Energy: {round(winner_energy,2)}", color='white')
    plt.text(0.3, 0.42, f"Score: {score}", color='#FFD700')

plt.xticks([])
plt.yticks([])
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.ioff()
plt.show()
