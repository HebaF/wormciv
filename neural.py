import numpy as np
from typing import Dict, List, Optional, Tuple

class Neuron:
    """Base class for neurons in the simplified C. elegans neural network."""
    def __init__(self, neuron_id: str, neuron_type: str):
        self.id = neuron_id
        self.type = neuron_type  # 'sensory', 'inter', or 'motor'
        self.activation = 0.0
        self.connections: List[Tuple[str, float]] = []  # [(target_id, weight)]
        self.bias = 0.0
        
    def connect(self, target_id: str, weight: float = 1.0):
        """Add a connection to another neuron with specified weight."""
        self.connections.append((target_id, weight))
        
    def update(self, inputs: float) -> float:
        """Update neuron activation using a simple sigmoid function."""
        self.activation = 1 / (1 + np.exp(-(inputs + self.bias)))
        return self.activation

class SensoryNeuron(Neuron):
    """Neurons that process environmental inputs (food, temperature)."""
    def __init__(self, neuron_id: str, sense_type: str):
        super().__init__(neuron_id, "sensory")
        self.sense_type = sense_type  # e.g., 'chemical', 'thermal'
        
    def process_stimulus(self, stimulus_value: float) -> float:
        """Process environmental stimulus and update activation."""
        # Normalize stimulus to 0-1 range and update
        normalized = max(0.0, min(1.0, stimulus_value))
        return self.update(normalized)

class InterNeuron(Neuron):
    """Interneurons that form the processing layer with learning capabilities."""
    def __init__(self, neuron_id: str):
        super().__init__(neuron_id, "inter")
        # Enhanced memory system
        self.short_term_memory = 0.0
        self.working_memory = 0.0  # New working memory buffer
        self.long_term_memory = 0.0
        self.episodic_memory: List[Tuple[float, float]] = []  # [(stimulus, outcome)]
        
        # Learning parameters
        self.learning_rate = 0.1
        self.stm_decay = 0.2
        self.wm_decay = 0.1  # Working memory decay
        self.ltm_decay = 0.02
        self.experience_count = 0
        
        # Plasticity parameters
        self.hebbian_strength = 0.0
        self.synaptic_tags: Dict[str, float] = {}  # Synaptic tagging for LTP/LTD
        self.consolidation_threshold = 0.7
        
    def update(self, inputs: float) -> float:
        """Update with enhanced memory and learning mechanisms."""
        # Update working memory with current context
        self.working_memory = (self.working_memory * (1 - self.wm_decay) + 
                             inputs * self.wm_decay)
        
        # Update short-term memory with working memory integration
        self.short_term_memory = (self.short_term_memory * (1 - self.stm_decay) + 
                                (inputs * 0.7 + self.working_memory * 0.3) * self.stm_decay)
        
        # Update episodic memory
        if len(self.episodic_memory) >= 10:  # Keep last 10 episodes
            self.episodic_memory.pop(0)
        self.episodic_memory.append((inputs, self.activation))
        
        # Pattern recognition for long-term memory formation
        if self._detect_pattern():
            self.experience_count += 1
            if self.experience_count > 3:  # Reduced threshold for faster learning
                # Update long-term memory with pattern consolidation
                self.long_term_memory = (self.long_term_memory * (1 - self.ltm_decay) + 
                                       self.short_term_memory * self.ltm_decay)
                # Update Hebbian learning strength
                self.hebbian_strength = min(1.0, self.hebbian_strength + 0.1)
        else:
            self.experience_count = max(0, self.experience_count - 1)
            self.hebbian_strength = max(0.0, self.hebbian_strength - 0.05)
        
        # Combine all memory systems with Hebbian influence
        combined_input = (
            inputs * 0.4 +
            self.working_memory * 0.1 +
            self.short_term_memory * 0.2 +
            self.long_term_memory * (0.3 + self.hebbian_strength * 0.2)
        )
        
        return super().update(combined_input)
        
    def _detect_pattern(self) -> bool:
        """Detect recurring patterns in episodic memory."""
        if len(self.episodic_memory) < 3:
            return False
            
        recent_inputs = [ep[0] for ep in self.episodic_memory[-3:]]
        pattern_strength = sum(abs(recent_inputs[i] - recent_inputs[i-1]) 
                             for i in range(1, len(recent_inputs)))
        return pattern_strength < 0.5  # Pattern detected if inputs are similar

    def adapt_weights(self, reward: float, learning_rate_modifier: float = 1.0):
        """Adapt connection weights based on reward signal with synaptic tagging."""
        effective_rate = self.learning_rate * learning_rate_modifier
        
        # Update synaptic tags based on recent activity
        for target_id, _ in self.connections:
            if target_id not in self.synaptic_tags:
                self.synaptic_tags[target_id] = 0.0
            
            # Tag strength increases with correlated activity
            tag_change = self.activation * reward * 0.3
            self.synaptic_tags[target_id] = min(1.0, 
                self.synaptic_tags[target_id] + tag_change)
        
        # Adapt weights based on tags and reward
        for i in range(len(self.connections)):
            target_id, weight = self.connections[i]
            tag_strength = self.synaptic_tags.get(target_id, 0.0)
            
            # Weight change influenced by tags and Hebbian learning
            weight_change = (effective_rate * reward * self.activation * 
                           (1.0 + tag_strength + self.hebbian_strength * 0.5))
            
            # Apply weight change with homeostatic constraints
            new_weight = max(0.1, min(2.0, weight + weight_change))
            self.connections[i] = (target_id, new_weight)
            
            # Tag decay
            self.synaptic_tags[target_id] *= 0.95

class MotorNeuron(Neuron):
    """Neurons that control movement outputs."""
    def __init__(self, neuron_id: str, movement_type: str):
        super().__init__(neuron_id, "motor")
        self.movement_type = movement_type  # e.g., 'forward', 'turn'
        
    def get_movement_command(self) -> Tuple[str, float]:
        """Convert activation to movement command."""
        return (self.movement_type, self.activation)

class NeuralNetwork:
    """Enhanced C. elegans neural network with learning capabilities."""
    def __init__(self):
        self.neurons: Dict[str, Neuron] = {}
        self.learning_history: List[float] = []
        self.plasticity_modifier = 1.0
        self.social_memory: Dict[str, List[float]] = {}  # Social learning buffer
        self.global_activity_state = 0.0
        
        # Initialize neural circuits
        self._init_chemotaxis_circuit()
        self._init_thermotaxis_circuit()
        self._init_learning_circuit()
        
    def _init_chemotaxis_circuit(self):
        """Initialize the food sensing circuit."""
        # Sensory neurons for chemical detection
        self.neurons["ASEL"] = SensoryNeuron("ASEL", "chemical")
        self.neurons["ASER"] = SensoryNeuron("ASER", "chemical")
        
        # Interneurons for processing
        self.neurons["AIY"] = InterNeuron("AIY")
        self.neurons["AIZ"] = InterNeuron("AIZ")
        
        # Motor neurons for movement
        self.neurons["SMBD"] = MotorNeuron("SMBD", "forward")
        self.neurons["SMBV"] = MotorNeuron("SMBV", "turn")
        
        # Connect neurons (simplified circuit)
        self.neurons["ASEL"].connect("AIY", 0.8)
        self.neurons["ASER"].connect("AIZ", 0.8)
        self.neurons["AIY"].connect("SMBD", 0.6)
        self.neurons["AIZ"].connect("SMBV", 0.6)
        
    def _init_thermotaxis_circuit(self):
        """Initialize the temperature sensing circuit."""
        # Sensory neurons for temperature
        self.neurons["AFD"] = SensoryNeuron("AFD", "thermal")
        
        # Interneurons
        self.neurons["RIA"] = InterNeuron("RIA")
        
        # Connect to existing motor neurons
        self.neurons["AFD"].connect("RIA", 0.7)
        self.neurons["RIA"].connect("SMBD", 0.5)
        self.neurons["RIA"].connect("SMBV", 0.5)
        
    def _init_learning_circuit(self):
        """Initialize additional neurons for learning circuit."""
        # Add learning-specific interneurons
        self.neurons["AIM"] = InterNeuron("AIM")  # Associative learning interneuron
        self.neurons["RIM"] = InterNeuron("RIM")  # Reinforcement learning interneuron
        
        # Connect learning circuit to existing circuits
        self.neurons["AIM"].connect("RIA", 0.5)
        self.neurons["RIM"].connect("AIY", 0.5)
        self.neurons["RIM"].connect("AIZ", 0.5)
        
    def process_sensory_input(self, inputs: Dict[str, float], reward: Optional[float] = None, 
                            social_inputs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Process environmental and social inputs through the network."""
        # Update global activity state
        self.global_activity_state = sum(self.get_network_state().values()) / len(self.neurons)
        
        # Process social inputs if available
        if social_inputs:
            for neuron_id, social_signal in social_inputs.items():
                if neuron_id not in self.social_memory:
                    self.social_memory[neuron_id] = []
                self.social_memory[neuron_id].append(social_signal)
                if len(self.social_memory[neuron_id]) > 5:  # Keep recent social history
                    self.social_memory[neuron_id].pop(0)
        
        # Update sensory neurons with social modulation
        for neuron_id, stimulus in inputs.items():
            if neuron_id in self.neurons and isinstance(self.neurons[neuron_id], SensoryNeuron):
                # Modulate stimulus with social information if available
                social_mod = 0.0
                if neuron_id in self.social_memory and self.social_memory[neuron_id]:
                    social_mod = sum(self.social_memory[neuron_id]) / len(self.social_memory[neuron_id])
                modulated_stimulus = stimulus * (1.0 + social_mod * 0.2)  # 20% social influence
                self.neurons[neuron_id].process_stimulus(modulated_stimulus)
        
        # Calculate network state for learning
        prev_state = self.get_network_state()
        
        # Update interneurons with enhanced learning
        for neuron in self.neurons.values():
            if isinstance(neuron, InterNeuron):
                # Calculate total input with social modulation
                total_input = sum(self.neurons[src].activation * weight 
                                for src, weight in neuron.connections)
                
                # Add social influence to total input
                if social_inputs:
                    social_factor = sum(social_inputs.values()) / len(social_inputs)
                    total_input *= (1.0 + social_factor * 0.1)  # 10% social modulation
                
                neuron.update(total_input)
                
                # Apply learning with social reinforcement
                if reward is not None:
                    social_bonus = 0.1 * self.global_activity_state if social_inputs else 0.0
                    effective_reward = reward * (1.0 + social_bonus)
                    neuron.adapt_weights(effective_reward, self.plasticity_modifier)
        
        # Update motor neurons and collect outputs
        outputs = {}
        for neuron in self.neurons.values():
            if isinstance(neuron, MotorNeuron):
                total_input = sum(self.neurons[src].activation * weight 
                                for src, weight in neuron.connections)
                neuron.update(total_input)
                movement_type, strength = neuron.get_movement_command()
                outputs[movement_type] = strength
        
        # Update learning history
        if reward is not None:
            self.learning_history.append(reward)
            if len(self.learning_history) > 100:  # Keep last 100 rewards
                self.learning_history.pop(0)
        
        return outputs
        
    def set_plasticity(self, modifier: float):
        """Set the plasticity modifier for learning rate."""
        self.plasticity_modifier = max(0.1, min(2.0, modifier))
        
    def get_learning_performance(self) -> float:
        """Calculate recent learning performance."""
        if not self.learning_history:
            return 0.0
        return sum(self.learning_history[-20:]) / min(20, len(self.learning_history))

    def get_network_state(self) -> Dict[str, float]:
        """Return current activation states of all neurons."""
        return {nid: neuron.activation for nid, neuron in self.neurons.items()}

    def get_connections(self) -> Dict[str, List[Tuple[str, float]]]:
        """Return all neural connections in the network."""
        return {nid: neuron.connections for nid, neuron in self.neurons.items()}

    def update_connection(self, source_id: str, target_id: str, new_weight: float):
        """Update the weight of a connection between two neurons."""
        if source_id in self.neurons:
            neuron = self.neurons[source_id]
            for i, (target, _) in enumerate(neuron.connections):
                if target == target_id:
                    neuron.connections[i] = (target_id, new_weight)
                    break
