from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.compute import Rack

with Diagram("A2C Architecture (a2c.py)", show=False, direction="LR"):
    env = Rack("Atari Environment")
    
    with Cluster("A2C Agent"):
        actor = Python("Actor")
        critic = Python("Critic")
        advantage = Python("Advantage Calculation\n(TD Error)")

    env >> Edge(label="State") >> actor
    env >> Edge(label="State") >> critic
    critic >> advantage >> Edge(label="Update") >> actor
    actor >> Edge(label="Action", color="red") >> env