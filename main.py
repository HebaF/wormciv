import pygame
import sys
import random
import numpy as np
from neural import NeuralNetwork
from ui import UI, WHITE, BLACK, RED, GREEN, BLUE, PINK, YELLOW, DARK_GRAY

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1200, 800  # Increased size to accommodate panels
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Worm Civilization: Neural Evolution")
clock = pygame.time.Clock()
FPS = 30
font = pygame.font.SysFont("Arial", 16)  # Define font for generation info

# Initialize UI
ui_system = UI(WIDTH, HEIGHT)

# --- Environmental Elements ---
# Adjust positions for new window size and simulation panel
simulation_area = ui_system.simulation_panel.rect
heat_zone = pygame.Rect(
    simulation_area.x + 100, 
    simulation_area.y + 100, 
    200, 150
)
food_patch = pygame.Rect(
    simulation_area.x + simulation_area.width - 300,
    simulation_area.y + simulation_area.height - 200,
    150, 100
)

#Generation Variables
generation = 1
generation_duration = 30  # Generation duration in seconds.
generation_timer = 0

# Track selected worm for neural visualization
selected_worm_index = 0

# --- Worm Class ---
class Worm:
    def __init__(self, x, y, worm_type="Default", genes=None):
        self.x = x
        self.y = y
        self.worm_type = worm_type
        # Initialize gene profiles with enhanced learning capabilities
        if genes is None:
            self.genes = {
                "speed": 1.0,      
                "resistance": 1.0, 
                "metabolism": 1.0,
                "neural_plasticity": 1.0,
                "sensory_range": 1.0,
                "learning_rate": 1.0,
                "memory_capacity": 1.0,
                "adaptation_speed": 1.0,
                "social_learning": 1.0,  # New gene for social learning capability
                "teaching_ability": 1.0  # New gene for ability to share knowledge
            }
        else:
            self.genes = genes
            
        # Initialize learning metrics
        self.last_health = 100
        self.cumulative_reward = 0
        self.learning_performance = 0
        
        self.base_speed = 2  # Base movement speed.
        self.speed = self.base_speed * self.genes["speed"]
        self.radius = 5      # Visual size.
        self.health = 100    # Starting health.
        self.fitness_score = 0  # Will be used later in evolution.
        
        # Enhanced neural system
        self.brain = NeuralNetwork()
        self.brain.set_plasticity(self.genes["neural_plasticity"])
        self.last_movement = {"forward": 0, "turn": 0}
        self.direction = 0  # Angle in radians
        self.previous_state = None
    
    def sense_environment(self, nearby_worms=None):
        """Process environmental and social inputs for the neural network."""
        sensory_range = 100 * self.genes["sensory_range"]  # Base range of 100 pixels
        
        # Calculate distances to food and heat
        food_center = food_patch.center
        heat_center = heat_zone.center
        
        # Distance to food (chemical gradient)
        food_dist = np.sqrt((self.x - food_center[0])**2 + (self.y - food_center[1])**2)
        food_signal = max(0, 1 - (food_dist / sensory_range))
        
        # Distance to heat (temperature gradient)
        heat_dist = np.sqrt((self.x - heat_center[0])**2 + (self.y - heat_center[1])**2)
        heat_signal = max(0, 1 - (heat_dist / sensory_range))
        
        # Process social information from nearby worms
        social_inputs = None
        if nearby_worms:
            social_inputs = {}
            for worm in nearby_worms:
                # Calculate social influence based on distance and teaching ability
                dist = np.sqrt((self.x - worm.x)**2 + (self.y - worm.y)**2)
                if dist < sensory_range:
                    influence = (1 - dist/sensory_range) * worm.genes["teaching_ability"]
                    # Share neural states weighted by influence
                    worm_state = worm.brain.get_network_state()
                    for neuron_id, activation in worm_state.items():
                        if neuron_id not in social_inputs:
                            social_inputs[neuron_id] = 0
                        social_inputs[neuron_id] += activation * influence
        
        # Prepare sensory inputs
        return {
            "ASEL": food_signal,  # Left chemical sensor
            "ASER": food_signal,  # Right chemical sensor
            "AFD": heat_signal    # Temperature sensor
        }, social_inputs
    
    def calculate_reward(self) -> float:
        """Calculate reward based on health change and environmental success."""
        health_change = self.health - self.last_health
        self.last_health = self.health
        
        # Base reward on health change
        reward = health_change * 0.1
        
        # Reward for finding food when health is low
        if food_patch.collidepoint(self.x, self.y) and self.health < 50:
            reward += 0.5
            
        # Penalty for staying in heat when health is low
        if heat_zone.collidepoint(self.x, self.y) and self.health < 50:
            reward -= 0.3
            
        # Scale reward by adaptation speed gene
        reward *= self.genes["adaptation_speed"]
        
        return reward

    def update(self, nearby_worms=None):
        # Get sensory and social inputs
        sensory_inputs, social_inputs = self.sense_environment(nearby_worms)
        reward = self.calculate_reward()
        
        # Process through neural network with learning and social modulation
        motor_outputs = self.brain.process_sensory_input(
            sensory_inputs, 
            reward,
            social_inputs if self.genes["social_learning"] > 0.5 else None
        )
        
        # Update learning performance
        self.learning_performance = self.brain.get_learning_performance()
        self.cumulative_reward += reward
        
        # Convert neural outputs to movement
        forward_strength = motor_outputs.get("forward", 0)
        turn_strength = motor_outputs.get("turn", 0)
        
        # Update direction and position
        self.direction += turn_strength * 0.1  # Scale turn rate
        speed = self.base_speed * self.genes["speed"] * forward_strength
        
        self.x += np.cos(self.direction) * speed
        self.y += np.sin(self.direction) * speed
        
        # Store last movement for animation/visualization
        self.last_movement = {"forward": forward_strength, "turn": turn_strength}

        # Ensure the worm stays within simulation panel boundaries
        self.x = max(simulation_area.x, min(simulation_area.right, self.x))
        self.y = max(simulation_area.y, min(simulation_area.bottom, self.y))
        
        # Hazard Effects: If in heat zone, lose health; if in food patch, gain health.
        if heat_zone.collidepoint(self.x, self.y):
            # Damage is reduced if resistance is high; metabolism modulates the rate.
            damage = 0.2 * (1.0 / self.genes["resistance"]) * self.genes["metabolism"]
            self.health -= damage
        if food_patch.collidepoint(self.x, self.y):
            gain = 0.1 * self.genes["metabolism"]
            self.health = min(self.health + gain, 100)
        
        # Update fitness score based on health and neural activity
        neural_state = self.brain.get_network_state()
        neural_activity = sum(neural_state.values()) / len(neural_state)
        self.fitness_score = self.health * (0.8 + 0.2 * neural_activity)  # Include neural activity in fitness

    def draw(self, surface, offset_x=0, offset_y=0, selected=False):
        # Draw worm body with position offset
        adjusted_x = self.x - offset_x
        adjusted_y = self.y - offset_y
        
        # Draw selection indicator if this is the selected worm
        if selected:
            pygame.draw.circle(surface, YELLOW, (int(adjusted_x), int(adjusted_y)), self.radius + 3, 2)
        
        pygame.draw.circle(surface, PINK, (int(adjusted_x), int(adjusted_y)), self.radius)
        
        # Draw direction indicator
        end_x = adjusted_x + np.cos(self.direction) * (self.radius + 2)
        end_y = adjusted_y + np.sin(self.direction) * (self.radius + 2)
        pygame.draw.line(surface, YELLOW, (adjusted_x, adjusted_y), (end_x, end_y), 2)
        
        # Draw health bar
        bar_width = 20
        bar_height = 4
        health_width = (bar_width * self.health) / 100
        bar_x = adjusted_x - bar_width/2
        bar_y = adjusted_y - self.radius - 6
        
        pygame.draw.rect(surface, RED, (bar_x, bar_y, bar_width, bar_height), 1)
        pygame.draw.rect(surface, GREEN, (bar_x, bar_y, health_width, bar_height))

# --- Create a Population of Worms ---
worms = []
for _ in range(10):  # Create 10 worms at random positions
    x = random.randint(simulation_area.x, simulation_area.right)
    y = random.randint(simulation_area.y, simulation_area.bottom)
    worms.append(Worm(x, y))

# --- Main Simulation Loop ---
running = True
while running:
    dt = clock.tick(FPS) / 1000  # Delta time in seconds.
    generation_timer += dt
    
    # Handle UI events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if click is in simulation area
            if simulation_area.collidepoint(event.pos):
                # Find closest worm to click position
                click_x, click_y = event.pos
                distances = [(i, np.sqrt((w.x - click_x)**2 + (w.y - click_y)**2)) 
                           for i, w in enumerate(worms)]
                selected_worm_index = min(distances, key=lambda x: x[1])[0]
        
        # Process UI events and get any changes
        changes = ui_system.handle_event(event)
        # Apply genetic changes from UI
        if changes:
            for worm in worms:
                if "Memory Capacity" in changes:
                    worm.genes["memory_capacity"] = changes["Memory Capacity"]
                if "Exploration" in changes:
                    worm.genes["adaptation_speed"] = changes["Exploration"]
                if "Neural Plasticity" in changes:
                    worm.genes["neural_plasticity"] = changes["Neural Plasticity"]
                    worm.brain.set_plasticity(changes["Neural Plasticity"])
                if "Decision Speed" in changes:
                    worm.genes["speed"] = changes["Decision Speed"]

    # Update each worm's position and health with social learning
    for worm in worms:
        # Find nearby worms for social learning
        nearby_worms = [w for w in worms if w != worm and 
                       np.sqrt((w.x - worm.x)**2 + (w.y - worm.y)**2) < 
                       100 * worm.genes["sensory_range"]]
        worm.update(nearby_worms)
    
    # Get neural network state and connections from selected worm
    selected_worm = worms[selected_worm_index]
    network_state = selected_worm.brain.get_network_state()
    connections = selected_worm.brain.get_connections()
    
    # Draw UI with neural network visualization
    screen.fill(DARK_GRAY)  # Background
    ui_system.evolution_panel.draw(screen)
    ui_system.simulation_panel.draw(screen)
    ui_system.connectome_panel.draw(screen, network_state, connections)
    
    # Draw simulation in its panel
    simulation_surface = screen.subsurface(simulation_area)
    simulation_surface.fill(WHITE)
    
    # Draw environment with adjusted positions
    adjusted_heat = pygame.Rect(
        heat_zone.x - simulation_area.x,
        heat_zone.y - simulation_area.y,
        heat_zone.width,
        heat_zone.height
    )
    adjusted_food = pygame.Rect(
        food_patch.x - simulation_area.x,
        food_patch.y - simulation_area.y,
        food_patch.width,
        food_patch.height
    )
    
    pygame.draw.rect(simulation_surface, RED, adjusted_heat)
    heat_label = font.render("Heat Zone", True, BLACK)
    simulation_surface.blit(heat_label, (adjusted_heat.x + 5, adjusted_heat.y + 5))
    
    pygame.draw.rect(simulation_surface, GREEN, adjusted_food)
    food_label = font.render("Food Patch", True, BLACK)
    simulation_surface.blit(food_label, (adjusted_food.x + 5, adjusted_food.y + 5))
    
    # Draw worms with adjusted positions
    for i, worm in enumerate(worms):
        worm.draw(simulation_surface, simulation_area.x, simulation_area.y, i == selected_worm_index)
    
    # Display stats in evolution panel
    stats_y = ui_system.evolution_panel.rect.bottom - 120
    stats = [
        f"Generation: {generation}",
        f"Time: {generation_timer:.1f}s",
        f"Worms: {len(worms)}",
        f"Best Fitness: {max(w.fitness_score for w in worms):.1f}",
        f"Best Learning: {max(w.learning_performance for w in worms):.2f}"
    ]
    
    for stat in stats:
        stat_surface = font.render(stat, True, BLACK)
        screen.blit(stat_surface, 
                   (ui_system.evolution_panel.rect.x + 10, stats_y))
        stats_y += 20
    
    pygame.display.flip()
    
    # Check if generation time is over.
    if generation_timer >= generation_duration:
        # Enhanced evolution with learning performance
        worms.sort(key=lambda w: w.fitness_score + w.learning_performance * 10, reverse=True)
        survivors = worms[:len(worms)//2]  # Top 50% survive.
        
        new_worms = []
        for worm in survivors:
            # Each survivor produces 2 offspring with intelligent mutations
            for _ in range(2):
                # Analyze which genes contributed to success
                learning_boost = max(0, worm.learning_performance)
                social_boost = max(0, worm.cumulative_reward * worm.genes["social_learning"])
                mutation_rate = 0.05 * (1.0 + learning_boost + social_boost * 0.5)
                
                new_genes = {}
                for key, value in worm.genes.items():
                    # Adaptive mutation based on performance
                    if key in ["learning_rate", "neural_plasticity", "memory_capacity"] and learning_boost > 0.5:
                        # Successful learners get targeted mutations in learning genes
                        mutation = random.uniform(-0.1, 0.15)
                    elif key in ["social_learning", "teaching_ability"] and social_boost > 0.5:
                        # Successful social learners get targeted mutations in social genes
                        mutation = random.uniform(-0.05, 0.2)
                    else:
                        mutation = random.uniform(-mutation_rate, mutation_rate)
                    
                    new_genes[key] = max(0.1, min(2.0, value + mutation))
                
                x = random.randint(simulation_area.x, simulation_area.right)
                y = random.randint(simulation_area.y, simulation_area.bottom)
                new_worm = Worm(x, y, worm_type=worm.worm_type, genes=new_genes)
                new_worms.append(new_worm)
        
        worms = new_worms  # Replace old generation.
        generation += 1
        generation_timer = 0
        selected_worm_index = 0  # Reset selected worm to first in new generation

pygame.quit()
sys.exit()
