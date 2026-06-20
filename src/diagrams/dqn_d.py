from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack

with Diagram("DQN Architecture (dqn.py)", show=False, direction="LR"):
    env = Rack("Atari Environment")
    
    with Cluster("DQN Agent"):
        logic = Python("DQN Logic")
        with Cluster("Neural Networks"):
            p_net = Python("Policy Net")
            t_net = Python("Target Net")
        replay = Storage("Experience Replay\n(Buffer)")

    env >> Edge(label="s, r, done") >> logic
    logic >> replay
    replay >> Edge(label="Batch Sample") >> p_net
    p_net >> Edge(label="Update Q-values") >> t_net
    p_net >> Edge(label="Action (Epsilon-greedy)", color="red") >> env