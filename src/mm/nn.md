graph TD
    subgraph Actor["Actor Network (Policy)"]
        direction TB
        A_Input[("Input: Observation (11)")]
        A_H1["Hidden Layer 1 <br/>(Linear 256 + ReLU)"]
        A_H2["Hidden Layer 2 <br/>(Linear 256 + ReLU)"]
        
        subgraph Heads["Output Heads"]
            A_Mean["Mean μ (2)"]
            A_Std["Log Std σ (2)"]
        end
        
        A_Sample(Gaussian Sampling)
        A_Tanh(Tanh Activation)
        A_Out[("Output: Action (2) <br/>Throttle, Steering")]
        
        A_Input --> A_H1
        A_H1 --> A_H2
        A_H2 --> A_Mean
        A_H2 --> A_Std
        A_Mean --> A_Sample
        A_Std --> A_Sample
        A_Sample --> A_Tanh
        A_Tanh --> A_Out
    end

    subgraph Critic["Critic Network (Q-Function)"]
        direction TB
        C_InputObs[("Input: Observation (11)")]
        C_InputAct[("Input: Action (2)")]
        C_Concat{Concatenate (13)}
        C_H1["Hidden Layer 1 <br/>(Linear 256 + ReLU)"]
        C_H2["Hidden Layer 2 <br/>(Linear 256 + ReLU)"]
        C_Out[("Output: Q-Value (1) <br/>Expected Return")]

        C_InputObs --> C_Concat
        C_InputAct --> C_Concat
        C_Concat --> C_H1
        C_H1 --> C_H2
        C_H2 --> C_Out
    end
