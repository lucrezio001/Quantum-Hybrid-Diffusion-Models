import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

def angle_encoding(inputs):
    qml.AngleEmbedding(inputs, wires=range(12), rotation='X')

def fqconv_ansatz(weights):

    # C & D blocks
    # C0 block
    qml.CRZ(weights[0], wires=[3, 11])
    qml.CRX(weights[1], wires=[2, 10])
    qml.CRZ(weights[2], wires=[1, 9])
    qml.CRX(weights[3], wires=[0, 8])
    
    # D0 block
    qml.CRX(weights[4], wires=[3, 11])
    qml.CRZ(weights[5], wires=[2, 10])
    qml.CRX(weights[6], wires=[1, 9])
    qml.CRZ(weights[7], wires=[0, 8])
    
    # C1 block
    qml.CRZ(weights[8], wires=[11, 7])
    qml.CRX(weights[9], wires=[10, 6])
    qml.CRZ(weights[10], wires=[9, 5])
    qml.CRX(weights[11], wires=[8, 4])
    
    # D1 block
    qml.CRX(weights[12], wires=[11, 7])
    qml.CRZ(weights[13], wires=[10, 6])
    qml.CRX(weights[14], wires=[9, 5])
    qml.CRZ(weights[15], wires=[8, 4])
    
    # C2 block
    qml.CRZ(weights[16], wires=[7, 3])
    qml.CRX(weights[17], wires=[6, 2])
    qml.CRZ(weights[18], wires=[5, 1])
    qml.CRX(weights[19], wires=[4, 0])
    
    # D2 block
    qml.CRX(weights[20], wires=[7, 3])
    qml.CRZ(weights[21], wires=[6, 2])
    qml.CRX(weights[22], wires=[5, 1])
    qml.CRZ(weights[23], wires=[4, 0])

# Dispositivo quantistico con 1 qubit
n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def fqconv_ansatz_circuit(inputs, weights):
    angle_encoding(inputs)
    fqconv_ansatz(weights)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


class fqconv_ansatz_layer(nn.Module):
    def __init__(self, n_qubits=12, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Parametri quantistici trainabili
        weight_shape = (n_layers, n_qubits)
        self.fqconv_weights = nn.Parameter(torch.randn(self.n_qubits * self.n_layers, requires_grad = True) * np.pi)
        print(self.fqconv_weights.shape)
        
    def forward(self, x):

        self.batch = x.shape[0]
        self.channel_shape = x.shape[1]
        self.channel = torch.split(x, 3, dim=1)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        for i in range(self.channel_shape//3):
            for j in range(self.batch):
                if i == 0 and j == 0:
                    outputs = torch.stack(fqconv_ansatz_circuit(torch.flatten(self.channel[i][j]), self.fqconv_weights), dim=0)
                else:
                    outputs = torch.cat((outputs,torch.stack(fqconv_ansatz_circuit(torch.flatten(self.channel[i][j]), self.fqconv_weights), dim=0)),0)
                    
        outputs = torch.reshape(outputs,[self.batch, self.channel_shape, 2, 2])
        outputs = torch.as_tensor(outputs, dtype=torch.float32)
        return outputs