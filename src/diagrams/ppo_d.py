from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.compute import Rack

with Diagram("PPO Architecture (ppo.py)", show=False, direction="LR"):
    env = Rack("Atari Environment")
    
    with Cluster("PPO Agent"):
        with Cluster("Actor-Critic Model"):
            actor = Python("Actor (Policy)")
            critic = Python("Critic (Value)")
        clip = Python("PPO Clip Logic\n(Surrogate Loss)")

    env >> Edge(label="State") >> actor
    env >> Edge(label="Reward") >> critic
    actor >> clip
    critic >> clip >> Edge(label="Policy Update") >> actor
    actor >> Edge(label="Action (Stochastic)", color="red") >> env