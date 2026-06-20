from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.device import Tablet  # For GPU/M4 representation
from diagrams.generic.storage import Storage # Using Storage instead of Folder
from diagrams.generic.compute import Rack   # For the game environment
from diagrams.onprem.client import User     # For the CLI interface

# Graph appearance settings
graph_attr = {
    "fontsize": "15",
    "bgcolor": "white"
}

with Diagram("Software Architecture HYBRIDEA on Mac M4", show=False, direction="LR", graph_attr=graph_attr):
    
    with Cluster("External Environment"):
        env = Rack("Ms. Pac-Man\n(Gymnasium/ALE)")

    with Cluster("Mac mini M4 (Computational Hub)"):
        with Cluster("Hardware Acceleration"):
            gpu = Tablet("10-Core GPU\n(Metal Performance)")
            ane = Tablet("16-Core Neural Engine\n(ANE)")
        
        with Cluster("Core Model Logic"):
            logic = Python("HYBRIDEA Engine\n(CNN + PPO)")
            cli = User("CLI Interface\n(train, refine, eval)")

    with Cluster("Data & Persistence"):
        db = Storage("Checkpoints (.pt)\nTraining Metrics")

    # Connections (Data Flow)
    env >> Edge(label="Raw Pixels", color="blue") >> logic
    logic >> Edge(label="MPS Ops") >> gpu
    logic >> Edge(label="ANE Inference") >> ane
    logic >> Edge(label="Load/Save", style="dashed") >> db
    cli >> Edge(color="darkgreen") >> logic
    logic >> Edge(label="Actions", color="red") >> env