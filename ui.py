import pygame
import pygame.gfxdraw
from typing import Dict, List, Tuple, Optional

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
BLUE = (0, 122, 255)
RED = (255, 59, 48)
GREEN = (52, 199, 89)
YELLOW = (255, 255, 0)
PINK = (255, 179, 222)

class Panel:
    """Base class for UI panels"""
    def __init__(self, rect: pygame.Rect, title: str):
        self.rect = rect
        self.title = title
        self.active = False
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 18, bold=True)
        
    def draw(self, surface: pygame.Surface):
        """Draw panel background and title"""
        # Draw panel background
        pygame.draw.rect(surface, WHITE, self.rect)
        pygame.draw.rect(surface, GRAY, self.rect, 1)
        
        # Draw title background
        title_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 30)
        pygame.draw.rect(surface, LIGHT_GRAY, title_rect)
        pygame.draw.rect(surface, GRAY, title_rect, 1)
        
        # Draw title text
        title_surface = self.title_font.render(self.title, True, BLACK)
        title_pos = (self.rect.x + 10, self.rect.y + 5)
        surface.blit(title_surface, title_pos)

class Slider:
    """UI slider control"""
    def __init__(self, rect: pygame.Rect, label: str, min_val: float, max_val: float, value: float):
        self.rect = rect
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.active = False
        self.font = pygame.font.SysFont("Arial", 14)
        
    def draw(self, surface: pygame.Surface):
        # Draw label
        label_surface = self.font.render(f"{self.label}: {self.value:.2f}", True, BLACK)
        surface.blit(label_surface, (self.rect.x, self.rect.y - 20))
        
        # Draw slider track
        track_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height//2 - 2, 
                               self.rect.width, 4)
        pygame.draw.rect(surface, GRAY, track_rect)
        
        # Draw slider handle
        handle_pos = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        handle_rect = pygame.Rect(handle_pos - 6, self.rect.y, 12, self.rect.height)
        pygame.draw.rect(surface, BLUE, handle_rect)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events and return True if value changed"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.active = True
                return self.update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.active = False
        elif event.type == pygame.MOUSEMOTION and self.active:
            return self.update_value(event.pos[0])
        return False
        
    def update_value(self, x_pos: int) -> bool:
        """Update slider value based on mouse position"""
        old_value = self.value
        self.value = self.min_val + (x_pos - self.rect.x) / self.rect.width * (self.max_val - self.min_val)
        self.value = max(self.min_val, min(self.max_val, self.value))
        return self.value != old_value

class EvolutionPanel(Panel):
    """Left panel for evolution controls"""
    def __init__(self, rect: pygame.Rect):
        super().__init__(rect, "Evolution Panel")
        
        # Create sliders
        slider_width = rect.width - 40
        slider_height = 20
        y_start = rect.y + 50
        y_spacing = 50
        
        self.sliders = {
            "Memory Capacity": Slider(
                pygame.Rect(rect.x + 20, y_start, slider_width, slider_height),
                "Memory Capacity", 0.0, 2.0, 1.0
            ),
            "Exploration": Slider(
                pygame.Rect(rect.x + 20, y_start + y_spacing, slider_width, slider_height),
                "Exploration", 0.0, 2.0, 1.0
            ),
            "Neural Plasticity": Slider(
                pygame.Rect(rect.x + 20, y_start + y_spacing * 2, slider_width, slider_height),
                "Neural Plasticity", 0.0, 2.0, 1.0
            ),
            "Decision Speed": Slider(
                pygame.Rect(rect.x + 20, y_start + y_spacing * 3, slider_width, slider_height),
                "Decision Speed", 0.0, 2.0, 1.0
            )
        }
        
    def draw(self, surface: pygame.Surface):
        super().draw(surface)
        for slider in self.sliders.values():
            slider.draw(surface)
            
    def handle_event(self, event: pygame.event.Event) -> Dict[str, float]:
        """Handle events and return dict of changed values"""
        if not self.active:
            return {}
            
        changes = {}
        for name, slider in self.sliders.items():
            if slider.handle_event(event):
                changes[name] = slider.value
        return changes

class SimulationPanel(Panel):
    """Center panel for simulation visualization"""
    def __init__(self, rect: pygame.Rect):
        super().__init__(rect, "Simulation Tank")
        
    def draw(self, surface: pygame.Surface):
        super().draw(surface)
        # Main simulation rendering is handled in main.py
        # This panel just provides the container

class ConnectomePanel(Panel):
    """Right panel for neural network visualization"""
    def __init__(self, rect: pygame.Rect):
        super().__init__(rect, "Neural Connectome")
        self.neuron_positions: Dict[str, Tuple[int, int]] = {}
        self.selected_neuron: Optional[str] = None
        self.hovered_neuron: Optional[str] = None
        self.neuron_radius = 15
        self.connection_colors = {
            'sensory': BLUE,
            'inter': GREEN,
            'motor': RED
        }
        self.tooltip_font = pygame.font.SysFont("Arial", 12)
        self.last_click_time = 0
        self.network_state = {}
        self.connections = {}
        
        # Initialize fixed positions for neurons
        margin = 40
        width = rect.width - margin * 2
        height = rect.height - margin * 2
        
        # Sensory layer (top)
        sensory_y = rect.y + margin + 60
        sensory_neurons = ['ASEL', 'ASER', 'AFD']
        x_spacing = width / (len(sensory_neurons) + 1)
        for i, neuron_id in enumerate(sensory_neurons, 1):
            self.neuron_positions[neuron_id] = (
                rect.x + margin + i * x_spacing,
                sensory_y
            )
        
        # Inter layer (middle)
        inter_y = rect.y + margin + height/2
        inter_neurons = ['AIY', 'AIZ', 'RIA', 'AIM', 'RIM']
        x_spacing = width / (len(inter_neurons) + 1)
        for i, neuron_id in enumerate(inter_neurons, 1):
            self.neuron_positions[neuron_id] = (
                rect.x + margin + i * x_spacing,
                inter_y
            )
        
        # Motor layer (bottom)
        motor_y = rect.y + margin + height - 60
        motor_neurons = ['SMBD', 'SMBV']
        x_spacing = width / (len(motor_neurons) + 1)
        for i, neuron_id in enumerate(motor_neurons, 1):
            self.neuron_positions[neuron_id] = (
                rect.x + margin + i * x_spacing,
                motor_y
            )
    
    def get_neuron_at_position(self, pos: Tuple[int, int]) -> Optional[str]:
        """Find neuron at the given position"""
        x, y = pos
        for neuron_id, (nx, ny) in self.neuron_positions.items():
            if (x - nx) ** 2 + (y - ny) ** 2 <= self.neuron_radius ** 2:
                return neuron_id
        return None
    
    def get_neuron_type(self, neuron_id: str) -> str:
        """Get the type of a neuron by its ID"""
        if neuron_id in ['ASEL', 'ASER', 'AFD']:
            return 'sensory'
        elif neuron_id in ['SMBD', 'SMBV']:
            return 'motor'
        else:
            return 'inter'
    
    def handle_event(self, event: pygame.event.Event):
        """Handle mouse events for the connectome panel"""
        if not self.active:
            return
            
        if event.type == pygame.MOUSEMOTION:
            self.hovered_neuron = self.get_neuron_at_position(event.pos)
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            clicked_neuron = self.get_neuron_at_position(event.pos)
            if clicked_neuron:
                current_time = pygame.time.get_ticks()
                if current_time - self.last_click_time < 500:  # Double click
                    self.handle_double_click(clicked_neuron)
                else:  # Single click
                    self.selected_neuron = clicked_neuron
                self.last_click_time = current_time
    
    def handle_double_click(self, neuron_id: str):
        """Handle double-click on a neuron"""
        if not self.selected_neuron or neuron_id == self.selected_neuron:
            return
            
        # Find connection between selected and clicked neurons
        if self.selected_neuron in self.connections:
            for i, (target, weight) in enumerate(self.connections[self.selected_neuron]):
                if target == neuron_id:
                    # For now, just increase weight by 0.1 on double-click
                    # In future, could show a dialog for precise adjustment
                    new_weight = min(2.0, weight + 0.1)
                    self.connections[self.selected_neuron][i] = (target, new_weight)
                    break
    
    def draw_tooltip(self, surface: pygame.Surface, neuron_id: str):
        """Draw tooltip with neuron information"""
        pos = self.neuron_positions[neuron_id]
        neuron_type = self.get_neuron_type(neuron_id)
        activation = self.network_state.get(neuron_id, 0.0)
        
        # Prepare tooltip text
        lines = [
            f"Neuron: {neuron_id}",
            f"Type: {neuron_type}",
            f"Activation: {activation:.2f}"
        ]
        
        # Calculate tooltip dimensions
        line_height = 15
        max_width = max(self.tooltip_font.size(line)[0] for line in lines)
        height = line_height * len(lines)
        
        # Position tooltip above neuron
        x = pos[0] - max_width // 2
        y = pos[1] - height - self.neuron_radius - 10
        
        # Draw tooltip background
        pygame.draw.rect(surface, WHITE, 
                        (x - 5, y - 5, max_width + 10, height + 10))
        pygame.draw.rect(surface, GRAY, 
                        (x - 5, y - 5, max_width + 10, height + 10), 1)
        
        # Draw tooltip text
        for i, line in enumerate(lines):
            text = self.tooltip_font.render(line, True, BLACK)
            surface.blit(text, (x, y + i * line_height))
    
    def draw_neuron(self, surface: pygame.Surface, pos: Tuple[int, int], 
                   neuron_id: str, neuron_type: str, activation: float):
        """Draw a single neuron with its activation level"""
        # Draw neuron circle
        color = self.connection_colors[neuron_type]
        x, y = pos
        
        # Draw activation fill
        fill_radius = int(self.neuron_radius * activation)
        if fill_radius > 0:
            pygame.draw.circle(surface, color, (int(x), int(y)), fill_radius)
        
        # Draw selection/hover indicators
        if neuron_id == self.selected_neuron:
            pygame.draw.circle(surface, YELLOW, (int(x), int(y)), 
                             self.neuron_radius + 2, 2)
        elif neuron_id == self.hovered_neuron:
            pygame.draw.circle(surface, LIGHT_GRAY, (int(x), int(y)), 
                             self.neuron_radius + 2, 1)
        
        # Draw outline
        pygame.draw.circle(surface, BLACK, (int(x), int(y)), self.neuron_radius, 2)
        
        # Draw label
        label = self.font.render(neuron_id, True, BLACK)
        surface.blit(label, (x - label.get_width()/2, y + self.neuron_radius + 5))
    
    def draw_connection(self, surface: pygame.Surface, start_pos: Tuple[int, int],
                       end_pos: Tuple[int, int], weight: float):
        """Draw a connection between neurons with weight visualization"""
        # Calculate line width based on weight (1-5 pixels)
        line_width = max(1, min(5, int(weight * 3)))
        
        # Draw the connection line
        pygame.draw.line(surface, GRAY, start_pos, end_pos, line_width)
        
    def draw(self, surface: pygame.Surface, network_state: Optional[Dict[str, float]] = None,
            connections: Optional[Dict[str, List[Tuple[str, float]]]] = None):
        """Draw the complete neural network visualization"""
        super().draw(surface)
        
        if not network_state or not connections:
            # Draw "No Data" message if network state isn't provided
            info_text = self.font.render("Waiting for neural network data...", True, BLACK)
            text_pos = (self.rect.centerx - info_text.get_width()//2, 
                       self.rect.centery - info_text.get_height()//2)
            surface.blit(info_text, text_pos)
            return
        
        # Store current state
        self.network_state = network_state
        self.connections = connections
        
        # Draw connections first (so they appear behind neurons)
        for source_id, targets in connections.items():
            if source_id in self.neuron_positions:
                start_pos = self.neuron_positions[source_id]
                for target_id, weight in targets:
                    if target_id in self.neuron_positions:
                        end_pos = self.neuron_positions[target_id]
                        self.draw_connection(surface, start_pos, end_pos, weight)
        
        # Draw neurons with their current activation states
        for neuron_id, pos in self.neuron_positions.items():
            if neuron_id in network_state:
                neuron_type = self.get_neuron_type(neuron_id)
                self.draw_neuron(surface, pos, neuron_id, neuron_type, 
                               network_state.get(neuron_id, 0.0))
        
        # Draw tooltip for hovered neuron
        if self.hovered_neuron:
            self.draw_tooltip(surface, self.hovered_neuron)

class UI:
    """Main UI manager"""
    def __init__(self, screen_width: int, screen_height: int):
        # Calculate panel dimensions
        panel_spacing = 10
        evolution_width = 250
        connectome_width = 300
        simulation_width = (screen_width - evolution_width - connectome_width - 
                          panel_spacing * 4)
        
        # Create panels
        self.evolution_panel = EvolutionPanel(
            pygame.Rect(panel_spacing, panel_spacing, 
                       evolution_width, screen_height - panel_spacing * 2)
        )
        
        self.simulation_panel = SimulationPanel(
            pygame.Rect(evolution_width + panel_spacing * 2, panel_spacing,
                       simulation_width, screen_height - panel_spacing * 2)
        )
        
        self.connectome_panel = ConnectomePanel(
            pygame.Rect(evolution_width + simulation_width + panel_spacing * 3, 
                       panel_spacing, connectome_width, screen_height - panel_spacing * 2)
        )
        
    def draw(self, surface: pygame.Surface):
        """Draw all UI elements"""
        surface.fill(DARK_GRAY)  # Background
        self.evolution_panel.draw(surface)
        self.simulation_panel.draw(surface)
        self.connectome_panel.draw(surface)
        
    def handle_event(self, event: pygame.event.Event) -> Dict[str, float]:
        """Handle UI events and return any changes"""
        # Update panel activation on mouse down
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Deactivate all panels first
            self.evolution_panel.active = False
            self.connectome_panel.active = False
            
            # Then activate the panel under the mouse
            if self.evolution_panel.rect.collidepoint(event.pos):
                self.evolution_panel.active = True
            elif self.connectome_panel.rect.collidepoint(event.pos):
                self.connectome_panel.active = True
        
        # Handle events for active panels
        if self.evolution_panel.active:
            changes = self.evolution_panel.handle_event(event)
            if changes:
                return changes
        
        if self.connectome_panel.active:
            self.connectome_panel.handle_event(event)
        
        return {}
