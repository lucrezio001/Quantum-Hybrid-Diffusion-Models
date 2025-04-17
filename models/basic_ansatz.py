import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# Equivalent of BasicEntanglerLayers i discovered late that it was implemented in pennylane so i build it from scratch

class fqconv_ansatz(nn.Module):

    def basic_angle_encoding(inputs):
        qml.AngleEmbedding(inputs, wires=range(4), rotation='X')
        
    def Basic_ansatz(weights):
        qml.RX(weights[0], 0)
        qml.RX(weights[1], 1)
        qml.RX(weights[2], 2)
        qml.RX(weights[3], 3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
        
        
        self.n_qubits = 4
        self.dev = qml.device("default.qubit", wires = self.n_qubits) # GPU "lightning.gpu"

        @qml.qnode(self.dev, interface="torch")
        def basic_ansatz_circuit(inputs, weights):
            basic_angle_encoding(inputs)
            Basic_ansatz(weights)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        # HQCONV layer
        self.basic_ansatz_shapes = {"weights": (4)}
        self.basic_ansatz_layer = qml.qnn.TorchLayer(basic_ansatz_circuit, self.basic_ansatz_shapes)
    