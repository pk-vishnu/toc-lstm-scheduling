from diagrams import Cluster, Diagram, Edge
from diagrams.generic.compute import Rack
from diagrams.onprem.compute import Server
from diagrams.programming.flowchart import Decision

with Diagram("", filename="rr_scheduler_diagram", show=False, direction="LR"):
    # Left side components
    task_arrival = Rack("Task Arrival\nProcess")
    dispatcher = Decision("Round Robin\nDispatcher")

    # Server cluster
    with Cluster("Server Cluster"):
        server1 = Server("Server 1\n\nCPU\nNetwork")
        server2 = Server("Server 2\n\nCPU\nNetwork")
        server3 = Server("Server 3\n\nCPU\nNetwork")

        servers = [server1, server2, server3]

    # Right side component
    logger = Rack("Periodic\nLogger")

    # Connections
    task_arrival >> Edge(label="") >> dispatcher
    dispatcher >> Edge(label="assign task") >> server1
    dispatcher >> Edge(label="assign task") >> server2
    dispatcher >> Edge(label="assign task") >> server3

    # Connect all servers to logger
    server1 >> Edge(label="update metrics") >> logger
    server2 >> Edge(label="update metrics") >> logger
    server3 >> Edge(label="update metrics") >> logger
