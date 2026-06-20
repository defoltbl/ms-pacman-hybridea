from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage

with Diagram("BC Architecture (bc.py)", show=False, direction="LR"):
    dataset = Storage("Expert Dataset\n(Demonstrations)")
    model = Python("CNN Policy")
    loss = Python("Cross-Entropy Loss")

    dataset >> Edge(label="Expert State") >> model
    model >> Edge(label="Predicted Action") >> loss
    dataset >> Edge(label="Target Action") >> loss
    loss >> Edge(label="Optimization") >> model