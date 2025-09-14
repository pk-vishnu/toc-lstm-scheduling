#set page(
  paper: "us-letter",
  margin: (x:2.8cm, y:2.6cm),
  header: align(right)[
    TOC and LSTM Based Resource Scheduling
  ], 
  numbering: "1",
  // columns: 2
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
#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 2em,
)[
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

  #align(center)[
    #set par(justify: false)
    *Abstract* \
    #lorem(80)
  ]
]

= Literature Review
Chang et al. @Chang2017ApplyingTO applied a TOC-Based approach to address memory allocation in cloud storage, which uses market information to build a rolling forecast. This demonstrates that TOC principles can be extended to resource scheduling problems in dynamic environments.

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
To identify potential bottlenecks, we trained a Long-Short-Term Memory neural network, as it is well suited for learning patterns and dependencies in time series data.

To capture temporal patterns, the continuous time series data for each server was transformed into overlapping sequences of windows of size 20 timesteps, with each window serving as an input sequence and the following timestep's bottleneck status as the label (1/0). This information allows the model to learn the patterns leading up to a bottleneck.

The dataset was then split into training and test sets and a MinMax scaler was fit only on the training data and applied to both the sets. Finally we trained a recurrent neural network with an LSTM layer followed by dense layers optimized using binary cross-entropy loss, to classify whether a bottleneck would occur in the next timestep, given the previous 20 timesteps.

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
The core principle of Round Robin is to achieve fairness and prevent starvation by distributing tasks in a simple, cyclical sequence by employing a stateless, turn-based approach. It maintains a pointer to the last server that received a task and assigns the next incoming task to the subsequent server in the sequence.

Our *resource aware* algorithm performs two crucial checks to make sure there will be no queue overloads or system failure due to capacity overloads.
1.  *Queue Capacity Check:* The server's task queue must not be full.
2.  *Resource Availability Check:* The server must have sufficient free CPU and network capacity to begin processing the specific task at that moment.

If the designated server fails these checks, the algorithm continues its cyclical search until an adequate server is found. If no server in the system can accept the task, it is rejected.

=== Algorithm Steps

Let $S$ be the set of $N$ servers, indexed as $S = \{s_0, s_1, ..., s_(N-1)\}$. Let $I_"last"$ be the index of the server that received the previous task. The scheduling process for each new incoming task, $J_"new"$, is executed as follows:

+ *Initialization:* The algorithm identifies the starting index for its search, $I_"start"$, based on the last server to receive a task:
  $
    I_"start" = (I_"last" + 1) mod N
  $

+ *Cyclical Search:* The algorithm iterates through all active servers $s_k$ in a circular order for $N$ steps, starting from the server at index $I_"start"$.

+ *Eligibility Check:* For each candidate server $s_k$, the algorithm determines its eligibility by evaluating two boolean conditions. The task, $J_"new"$, arrives with a set of resource demands $D = \{D_"cpu", D_"net_in", D_"net_out"\}$. The state of server $s_k$ at time $t$ is defined by its queue length $Q_(s_k)(t)$ and its available capacities for each resource $r in {"cpu", "net_in", "net_out"}$, denoted $C_(s_k, "avail")^r(t)$.

  1.  *The Queue Capacity Condition*, $"Accept"_Q(s_k)$, must be true:
      $
        "Accept"_Q(s_k) = [ Q_(s_k)(t) < Q_"max" ]
      $
      where $[.]$ is the Iverson bracket.

  2.  *The Resource Availability Condition*, $"Accept"_R(s_k, J_"new")$, must also be true:
      $
        "Accept"_R(s_k, J_"new") = [D_"cpu" <= C_(s_k, "avail")^"cpu"(t)] and [D_"net_in" <= C_(s_k, "avail")^"net_in"(t)] and [D_"net_out" <= C_(s_k, "avail")^"net_out"(t)]
      $

+ *Assignment or Rejection:* The first server $s_k$ in the sequence for which both conditions are met is selected as the target server, $S_"target"$.
  $
    S_"target" = s_k quad "where" quad "Accept"_Q(s_k) and "Accept"_R(s_k, J_"new")
  $
  Upon successful assignment, the index $I_"last"$ is updated to $k$, and the search terminates. If the algorithm completes a full cycle and no server satisfies both conditions, the task $J_"new"$ is rejected, constituting an SLA violation.

=== Architecture
#figure(
  image("/TrainingLSTM/RR_architecture.png"),
  caption: "Architecture of Round Robin Scheduler"
)<RRarchitecture>

== Algorithm 2 - Theory of Constraints Simulation
  This algorithm provides a practical implementation of TOC's *Drum-Buffer-Rope (DBR)* methodology for a parallel, dynamic server environment.

- *The Drum:* The system's identified constraint, whose processing rate dictates the optimal rate at which new work should be introduced.
- *The Buffer:* A small, managed queue of tasks placed before each resource. The buffer in front of the constraint is the most critical, as it must ensure the constraint is never idle due to a lack of work.
- *The Rope:* A signaling mechanism that links the constraint's buffer back to the system's entry point. It authorizes the release of new work into the system only when the constraint has the capacity to process it, thereby synchronizing the entire system to the pace of its slowest part.

The algorithm is composed of four primary components that map to the Five Focusing Steps of TOC: Constraint Identification, Flow Control (Dispatcher), Task Assignment, and System Scaling.

===  Algorithm Steps

Let $S$ be the set of all servers. At any time $t$, the set of active servers is denoted by $S_"active"(t) subset.eq S$. Each server $s in S$ has a maximum CPU capacity $C_(s,"cpu")$ and network capacity $C_(s,"net")$.
==== Constraint Identification (Identify)

The first step is to dynamically and continuously identify the system's primary constraint. Instantaneous resource utilization is often volatile; therefore, a smoothing function is required to identify the most persistently loaded resource. We employ an *Exponentially Weighted Moving Average (EWMA)* for this purpose.

Let $U_(s,r)(t)$ be the instantaneous utilization of a resource $r in {"cpu", "net"}$ on a server $s$ at time $t$. The utilization is a normalized value where $0 <= U <= 1$.

The smoothed utilization, $U.bar_(s,r)(t)$, is calculated recursively:
$
  U.bar_(s,r)(t) = (alpha dot U_(s,r)(t)) + ((1 - alpha) dot U.bar_(s,r)(t - Delta t))
$
Where:
- $alpha$ is the smoothing factor ($0 < alpha < 1$). A lower $alpha$ results in a smoother, less volatile trendline.
- $Delta t$ is the time interval between measurements.

The system constraint at time $t$, denoted $C(t)$, is the specific resource $(s, r)$ with the highest smoothed utilization across all active servers.
$
  C(t) = op("argmax")_(s in S_"active"(t), r in {"cpu", "net"}) { U.bar_(s,r)(t) }
$
This identification process runs at a fixed interval, `CONSTRAINT_CHECK_INTERVAL`, to adapt to changing system loads.

==== Flow Control via Dispatcher (Exploit & Subordinate)

The core of the DBR implementation is a centralized *Dispatcher* that acts as the "Rope." It manages a central priority queue of incoming tasks, $B_"central"$, and only releases work into the system based on the state of the identified constraint, $C(t)$.

Let $C_s(t)$ be the server component of the constraint $C(t)$. Let $Q_s(t)$ be the length of the local buffer (queue) of server $s$ at time $t$, and let $Q_"max"$ be the maximum configured size of this buffer.

The *Rope Condition*, $"Release"(t)$, is a boolean function that determines if a new task should be released from $B_"central"$:
$
  "Release"(t) = [Q_(C_s(t))(t) < Q_"max"]
$
This condition ensures that a new task is only introduced into the system when the constraint's buffer has capacity. This prevents the accumulation of excessive Work-in-Process (WIP) and paces the entire system to its bottleneck. If $"Release"(t)$ is false, no tasks are dispatched, and they remain in the managed, prioritized central buffer.

==== Task Assignment (Subordinate)

When the Rope Condition $"Release"(t)$ is met, the highest-priority task, $J_"next"$, is selected from the central buffer:
$
  J_"next" = op("argmin")_(j in B_"central") { P(j) }
$
where $P(j)$ is the priority value of task $j$ (lower is higher).

This task must then be assigned to an active server. This is a subordinate decision, designed to efficiently utilize non-constraint resources without disturbing the system's flow. The target server, $S_"target"$, is selected from the set of available servers, $S_"avail"(t)$, by finding the server with the smallest local buffer.

The set of available servers is defined as:
$
  S_"avail"(t) = { s in S_"active"(t) | Q_s(t) < Q_"max" }
$
The target server is then chosen by:
$
  S_"target" = op("argmin")_(s in S_"avail"(t)) { Q_s(t) }
$
This ensures the released task is routed to the most idle part of the system, minimizing its local wait time and keeping non-constraint resources productive.

==== System Scaling (Elevate)

The final component is the autoscaler, which implements the "Elevate the Constraint" step. It modifies the size of the active server set, $abs(S_"active"(t))$.

Let $theta_"up"$ be the scale-up threshold and $theta_"down"$ be the scale-down threshold. Let $N(t) = abs(S_"active"(t))$ be the number of active servers.

*Scale-Up Condition:* The decision to scale up is based solely on the status of the constraint. If the smoothed utilization of the constraint resource exceeds the threshold, a new server is activated.
$
  "ScaleUp"(t) = [U.bar_(C(t))(t) > theta_"up"] and [N(t) < N_"max"]
$
This ensures that capacity is added precisely where it is needed to relieve the system's bottleneck.

*Scale-Down Condition:* The decision to scale down is based on overall system idleness. Let $U.bar_"sys"(t)$ be the average CPU utilization across all active servers. To prevent premature scaling during the initial warm-up phase, a time condition $T_"warmup"$ is included.
$
  "ScaleDown"(t) = [t > T_"warmup"] and [U.bar_"sys"(t) < theta_"down"] and [N(t) > N_"min"]
$
This allows the system to conserve resources when the overall demand is low, without being triggered by the intentionally low utilization of non-constraint servers during periods of high load.

=== Architecture
#figure(
  image("/TrainingLSTM/TOC_architecture.png"),
  caption: "Architecture of Theory of Constraints Scheduler"
)<TOCarchitecture>
== Algorithm 3 - Theory of Constraints with LSTM Bottleneck Prediction

The final algorithm represents the synthesis of the preceding methodologies, integrating the predictive capabilities of the trained Long-Short Term Memory (LSTM) model into the Theory of Constraints (TOC) framework. This creates a proactive, intelligent scheduling system that anticipates bottlenecks rather than merely reacting to them. The architecture evolves from the reactive DBR model of Algorithm 2 into a more sophisticated system where scheduling and scaling decisions are informed by the predicted future state of the server cluster.

A critical refinement in this model is a shift in the core TOC logic. Instead of pacing the system based on the constraint's buffer (Drum-Buffer-Rope), this algorithm employs a *constraint avoidance* strategy. The *Dispatcher* now treats the predicted bottleneck as a "hot zone" and actively routes tasks to other, healthier servers, using the constraint server only as a last resort. This maintains system fluidity by preventing the predicted bottleneck from becoming overloaded in the first place.

=== Algorithm Steps

The algorithm's components are refactored to incorporate the predictive model. The *ConstraintDetector* is now AI-driven, and its output directly influences the *Dispatcher* and the *Autoscaler*.

==== Predictive Constraint Identification (The AI "Drum")

The reactive, utilization-based constraint identification is replaced by a predictive process that leverages the trained LSTM model. This component's goal is to forecast which server is most likely to become a bottleneck in the near future.

Let $s_i$ be an active server. At time $t$, its state over the last $W$ timesteps (the `WINDOW_SIZE`) is represented by a feature matrix $X_(s_i)(t) in RR^(W times F)$, where $F$ is the number of features (CPU, Queue Length, Network In, Network Out).

1.  *Feature Scaling:* The raw feature matrix $X_(s_i)(t)$ is normalized using the pre-trained scaler function, $g_"scaler"$:
    $
      hat(X)_s_i (t) = g_"scaler" (X_(s_i)(t))
    $

2.  *Inference:* The normalized feature matrix is fed into the trained LSTM model, $f_"LSTM"$, which outputs a bottleneck probability score, $P_"bottleneck"$.
    $
      P_"bottleneck"(s_i, t) = f_"LSTM"(hat(X)_(s_i)(t))
    $

The system's predicted constraint at time $t$, $C_p (t)$, is the server with the highest probability score.
  $
    C_p (t) = op("argmax")(s_i in bb(S)_("active")(t)) { P_"bottleneck"(s_i, t)}
  $

==== Task Dispatching via Constraint Avoidance (Subordinate)

The `Dispatcher` logic is inverted from the DBR model. Instead of subordinating to the constraint's pace, it actively works around the predicted constraint to prevent pile-ups.

For an incoming task $J_"new"$, the target server, $bb(S)_"target"$, is selected as follows:

1.  *Define the Candidate Pool:* First, a set of eligible, non-constraint servers, $bb(S)_("eligible")(t)$, is created.
    $
      bb(S)_"eligible"(t) = {s in bb(S)_"active"(t) | s != C_p (t) and Q_s (t) < Q_"max" }
    $

2.  *Primary Assignment:* If the eligible pool is not empty, the task is assigned to the server with the minimum queue length within that pool.
    $
      bb(S)_"target" = op("argmin")_(s in bb(S)_"eligible" (t)) { Q_s (t) }
    $

3.  *Fallback Assignment:* If, and only if, the eligible pool is empty (meaning all non-constraint servers are at maximum capacity), the system will attempt to assign the task to the constraint server, $C_p (t)$, provided it has queue space. This prevents a total system stall when under extreme load.

==== Hybrid System Scaling (Elevate)

The autoscaler is enhanced with a hybrid proactive/reactive policy to make it both intelligent and resilient.

Let $P_c (t)$ be the bottleneck score of the predicted constraint server, $C_p (t)$. Let $U_c (t)$ be the *current* maximum resource utilization (CPU or Network) of that same server. The scale-up decision is triggered by either a proactive or a reactive condition.

1.  *Proactive Trigger (AI-Driven):* Scale up if the AI's confidence in an impending bottleneck is high.
    $
      "Trigger"_"proactive" = [ P_c (t) > theta_"up_prob" ]
    $

2.  *Reactive Trigger (Failsafe):* Scale up if the predicted constraint is *already* in a danger zone, even if the AI's score is not high. This protects against sudden, unpredicted load spikes.
    $
      "Trigger"_"reactive" = [ U_c (t) > theta_"danger" ]
    $

The final scale-up condition is a logical OR of these two triggers:
$
  "ScaleUp"(t) = ("Trigger"_"proactive" or "Trigger"_"reactive") and [N(t) < N_"max"]
$
where $N(t)$ denotes the number of active servers at time $t$, and $N_"max"$ is the maximum configured server limit for the system.

The scale-down logic remains unchanged from Algorithm 2, providing stability by only removing resources when the entire system is demonstrably idle.


#bibliography("refs.bib", title: "References")
