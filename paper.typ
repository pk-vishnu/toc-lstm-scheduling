#set page(
  paper: "us-letter",
  margin: (x:2.8cm, y:2.6cm),
  header: align(right)[
    TOC and LSTM Based Resource Scheduling
  ], 
  numbering: "1"
)
#set text(
  font: "New Computer Modern",
  size: 12pt
)
#set par(
  justify: true,
  leading: 0.52em
)
#set heading(
  numbering:"1.1" 
)
#align(center, text(17pt)[
  *TOC and LSTM Based Resource Scheduling*
])
#grid(
  columns: (1fr, 1fr,1fr),
  align(center)[
    Nipun S Nair \
    VIT Vellore \
  ],
  align(center)[
    Anamika \
    VIT Vellore \
  ],
  align(center)[
    Vishnu P K  \
    VIT Vellore \
  ],
)


= Literature Review
#lorem(100)


= Methodology
This section describes the procedures, tools, and methods used to conduct the research.

== Training LSTM Model to predict bottlenecks
=== Synthetic Data Generation
We simulate a realistic time-series dataset representing server resource usage over time. The goal is to generate data that mimics real-world system behavior to train an LSTM model for bottleneck prediction.

#figure(
  caption: [Simulation Parameters],
  kind: table, 
  [
    #table(
      columns: (auto,auto),
      align: left,
      stroke: 0.4pt,
      [*Parameter*], [*Description*],
      [NUM_SERVERS], [Number of simulated servers],
      [SIM_TIME], [Total simulated duration (in time units)],
      [TASK_INTERVAL], [Mean inter-arrival time for new tasks],
      [CPU_CAPACITY], [Max CPU Usage (100%)],
      [NET_CAPACITY], [Max network bandwidth (100%)],
    ) 
  ]
)<simulation_parameters>

==== Poisson-Based Task Arrival
Task arrival times are based on a Poisson process, simulated using the exponential distribution:

$ "Interarrival Time" ~ "Exponential"(lambda = 1 / "TASK_INTERVAL") $

This introduces realistic irregularity in task arrivals—mimicking user requests, job submissions, or packet arrivals.

==== Time-Varying Resource Usage Patterns
We simulate periodic system load using sinusoidal functions to represent diurnal load patterns, cyclic CPU spikes, and network traffic fluctuations.

*CPU Usage*
$ "cpu_base"(t) = 50 + 40 * sin((2pi t) / "PERIOD") $

*Network In/Out*
$ "net_base"(t, s) = 30 + 25 * sin((2pi (t + s * 10)) / "PERIOD") $

Where:
- $t$: current timestamp
- $s$: server index (adds phase shift between servers)
- `PERIOD`: defines workload cycle length (e.g. peaks every 100 time units)

==== Gaussian Noise for Realism
We inject Gaussian noise to simulate sensor or monitoring variability. This produces "wobble" on top of clean trends, similar to real monitoring data.

$ "cpu" = "clip"("cpu_base" + cal(N)(0, 10), 0, 100) $

$ "net_in", "net_out" = "clip"("net_base" + cal(N)(0, 5), 0, 100) $

==== Scheduled Bottlenecks
At specific intervals (e.g., for $t = 250$ to $260$ on Server 1), we introduce high-load conditions to ensure predictable bottlenecks for model learning and testing.
- CPU, Net In, and Net Out are all set to $>= 90%$.

==== Bottleneck Label Definition
A binary bottleneck label is assigned based on an 80% utilization threshold:

$ "bottleneck" = cases(
  1, "if CPU" >= 80 " or Net In/Out" >= 80,
  0, "otherwise"
) $

#figure(
  image("./TrainingLSTM/SyntheticData.png"),
  caption: "Synthetic Data - Compute and Network Resources"
)<syntheticData>

=== Training Model
We train the LSTM model using the scikit learn library with a train test split of 80:20.

=== Evaluation
The model was evaluated on a test set of 293 samples, achieving an overall accuracy of 89%. The detailed performance metrics from the classification report are presented below.
#figure(
  caption: [Classification Report],
  kind: table,
  [
    #table(
      columns: 5,
      align: (center, center, center, center, center),
      stroke: 0.4pt,
      [], [*Precision*], [*Recall*], [*F1-Score*], [*Support*],
      [*No Bottleneck (0)*], [0.94], [0.91], [0.93], [235],
      [*Bottleneck (1)*], [0.69], [0.78], [0.73], [58],
      [], [], [], [], [],
      [*Accuracy*], [], [], [0.89], [293],
      [*Macro Avg*], [0.82], [0.85], [0.83], [293],
      [*Weighted Avg*], [0.89], [0.89], [0.89], [293],
    ) 
  ]
)<classification_report>

#figure(
  caption: [Confusion matrix of model predictions.],
  kind: table,
   [
    #table(
      columns: (auto, auto, auto),
      align: center,
      stroke: 0.4pt,
      [], [*Predicted: No Bottleneck*], [*Predicted: Bottleneck*],
      [*Actual: No Bottleneck*], [215], [20],
      [*Actual: Bottleneck*], [13], [45],
    )
  ],
) <confusion_matrix>


== Algorithm 1 (Base) - Round Robin Simulation

