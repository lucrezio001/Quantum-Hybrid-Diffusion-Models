import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

def angle_encoding(inputs):
    qml.AngleEmbedding(inputs, wires=range(12), rotation='X')

def hqconv_ansatz(weights):
    #A blocks
    # A2 block
    qml.CRZ(weights[0], wires=[0, 3])
    qml.CRX(weights[1], wires=[0, 3])
    qml.CRZ(weights[2], wires=[3, 2])
    qml.CRX(weights[3], wires=[3, 2])
    qml.CRZ(weights[4], wires=[2, 1])
    qml.CRX(weights[5], wires=[2, 1])
    qml.CRZ(weights[6], wires=[1, 0])
    qml.CRX(weights[7], wires=[1, 0])
    # A1 block
    qml.CRZ(weights[8], wires=[4, 7])
    qml.CRX(weights[9], wires=[4, 7])
    qml.CRZ(weights[10], wires=[7, 6])
    qml.CRX(weights[11], wires=[7, 6])
    qml.CRZ(weights[12], wires=[6, 5])
    qml.CRX(weights[13], wires=[6, 5])
    qml.CRZ(weights[14], wires=[5, 4])
    qml.CRX(weights[15], wires=[5, 4])
    # A0 block
    qml.CRZ(weights[16], wires=[8, 11])
    qml.CRX(weights[17], wires=[8, 11])
    qml.CRZ(weights[18], wires=[11, 10])
    qml.CRX(weights[19], wires=[11, 10])
    qml.CRZ(weights[20], wires=[10, 9])
    qml.CRX(weights[21], wires=[10, 9])
    qml.CRZ(weights[22], wires=[9, 8])
    qml.CRX(weights[23], wires=[9, 8])

    # B Blocks
    # B0
    qml.CRZ(weights[24], wires=[4, 8])
    qml.CRX(weights[25], wires=[4, 8])
        
    # B1
    qml.CRZ(weights[26], wires=[0, 4])
    qml.CRX(weights[27], wires=[0, 4])
        
# Dispositivo quantistico con 1 qubit
n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def hqconv_ansatz_circuit(inputs, weights):
    angle_encoding(inputs)
    hqconv_ansatz(weights)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


class hqconv_ansatz_layer(nn.Module):
    def __init__(self, n_qubits=12, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Parametri quantistici trainabili
        weight_shape = (n_layers, n_qubits)
        self.fqconv_weights = nn.Parameter(torch.randn(self.n_qubits * self.n_layers + 4, requires_grad = True) * np.pi)
        print(self.fqconv_weights.shape)
        
    def forward(self, x):

        self.batch = x.shape[0]
        self.channel_shape = x.shape[1]
        self.channel = torch.split(x, 3, dim=1)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        for i in range(self.channel_shape//3):
            for j in range(self.batch):
                if i == 0 and j == 0:
                    outputs = torch.stack(hqconv_ansatz_circuit(torch.flatten(self.channel[i][j]), self.fqconv_weights), dim=0)
                else:
                    outputs = torch.cat((outputs,torch.stack(hqconv_ansatz_circuit(torch.flatten(self.channel[i][j]), self.fqconv_weights), dim=0)),0)
                    
        outputs = torch.reshape(outputs,[self.batch, self.channel_shape, 2, 2])
        outputs = torch.as_tensor(outputs, dtype=torch.float32)
        return outputs