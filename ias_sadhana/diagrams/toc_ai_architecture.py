from diagrams import Cluster, Diagram, Edge
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage
from diagrams.onprem.compute import Server
from diagrams.onprem.monitoring import Prometheus
from diagrams.programming.flowchart import Decision

graph_attr = {
    "splines": "ortho",
    "nodesep": "1.0",
    "ranksep": "1.5",
    "pad": "0.5",
}

node_attr = {
    "shape": "box",
    "style": "rounded,filled",
    "fillcolor": "white",
    "fontsize": "11",
}

edge_attr = {"fontsize": "10"}

with Diagram(
    "",
    filename="toc_ai_architecture_diagram",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):

    # Column 1: Task Generation
    task_gen = Rack("Task\nGenerator")

    # Column 2: Dispatching
    with Cluster("Task Dispatching", graph_attr={"rank": "same"}):
        central_buffer = Storage("Central Buffer")
        dispatcher = Decision("Central\nDispatcher")

    # Column 3: Server Pool
    with Cluster("Server Pool"):
        server1 = Server("Server 1\nCPU | Network")
        server2 = Server("Server 2\nCPU | Network")
        server3 = Server("Server 3\nCPU | Network")
        servers = [server1, server2, server3]

    # Column 4: Monitoring
    constraint_detector = Prometheus("AI Constraint\nDetector")

    # Column 5: Scaling
    autoscaler = Decision("Autoscaler")

    # Column 6: Logging
    logger = Rack("Logger")

    # Main flow (left to right)
    task_gen >> Edge(label="generate tasks") >> central_buffer
    central_buffer >> dispatcher
    dispatcher >> Edge(label="dispatch tasks") >> server1
    dispatcher >> server2
    dispatcher >> server3

    # Resource reporting (servers to constraint detector)
    (
        server1
        >> Edge(label="report resource\nusage", style="dashed")
        >> constraint_detector
    )
    server2 >> Edge(style="dashed") >> constraint_detector
    server3 >> Edge(style="dashed") >> constraint_detector

    # Constraint info to autoscaler
    constraint_detector >> Edge(label="constraint info") >> autoscaler

    # Logging
    autoscaler >> Edge(label="log scaling\nevents") >> logger

    # Feedback loops (dotted lines going backward)
    # Use xlabel to position labels outside the edge path
    (
        constraint_detector
        >> Edge(xlabel="query constraint", style="dotted", dir="back")
        >> dispatcher
    )
    autoscaler >> Edge(xlabel="scale up/down", style="dotted", dir="back") >> server1
    dispatcher >> Edge(xlabel="log stats", style="dotted") >> logger
    logger >> Edge(xlabel="feedback", style="dotted", dir="back") >> autoscaler
