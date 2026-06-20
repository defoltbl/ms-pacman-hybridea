from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage

with Diagram("Monte Carlo Architecture (monte_carlo.py)", show=False, direction="LR"):
    with Cluster("Episode Cycle"):
        env = Python("Environment")
        traj = Storage("Trajectory Buffer\n(Full Episode)")
    
    model = Python("Policy Network")
    update = Python("Return Calculation\n(G_t)")

    env >> Edge(label="Actions/Rewards") >> traj
    traj >> Edge(label="End of Episode") >> update
    update >> Edge(label="Gradient Ascent") >> model
    model >> Edge(label="Pi(a|s)") >> env