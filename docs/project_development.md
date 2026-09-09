# Project Development Journey

## Why I Started This Project

I began this project as a **self-directed summer project immediately after completing my first year of Biomedical Engineering**. My aim was to use the summer to build practical experience in medical robotics beyond what I had covered formally at university.

At the start, I did **not** already know how to build the final system. I had first-year engineering mathematics, introductory programming experience, and a general interest in medical robotics, but I had not previously built a ROS 2 system or worked in depth with motion planning, active perception, uncertainty-aware robotics, statistical experimental design, or autonomous safety supervision.

The project therefore developed as a learning process rather than from a complete plan made at the beginning. I started with one simple engineering question:

> **Can I make a simulated surgical instrument find a collision-free path to a target?**

Each time the system exposed a limitation, I researched the relevant area, learned the mathematics or software concepts needed to address it, implemented a small version, tested it, and then integrated it into the larger project.

A typical cycle was:

**Identify limitation â†’ Research established approaches â†’ Learn the required theory â†’ Implement â†’ Test â†’ Inspect failures â†’ Refine â†’ Integrate â†’ Validate**

This process is what gradually turned a small planning experiment into an uncertainty-aware surgical-navigation research framework with ROS 2/Gazebo integration and autonomous runtime safety supervision.

---

## What I Knew at the Beginning

My starting point was not postgraduate robotics expertise.

Before the project, I was comfortable with the kinds of mathematical and programming ideas expected from an early engineering student, including basic algebra, vectors, matrices, functions, numerical calculations, and introductory Python programming.

I had not yet formally studied many of the topics that later became important to the project. These included:

- configuration-space motion planning;
- Rapidly-exploring Random Trees (RRT);
- active perception;
- covariance-based uncertainty representation;
- uncertainty calibration;
- bootstrap confidence intervals and permutation tests;
- chance/risk-aware planning;
- ROS 2 nodes, topics, messages and actions;
- Gazebo and ros2_control;
- autonomous runtime supervision.

I did not try to learn all of these topics before beginning. They appeared because each new stage created a problem that the previous stage could not solve.

---

## How I Found the Right Algorithms, Mathematics and Platforms

I generally started from the **engineering problem**, not from a list of fashionable technologies.

When I encountered a problem I did not know how to solve, I first identified the technical field it belonged to, then looked for established approaches in research papers, textbooks, official documentation, educational material and implementation examples.

For important technical decisions, I tried to answer:

1. **What problem does this method solve?**
2. **What assumptions does it make?**
3. **What does each variable or parameter mean?**
4. **Why is it appropriate for this project?**
5. **What alternatives exist?**
6. **How can I test whether I have implemented it correctly?**

I used AI-assisted development tools as one part of this learning workflow for activities such as concept explanation, debugging, code review and exploring implementation approaches. I treated suggestions as things to inspect and test rather than as authoritative answers. The main development story of the project is the sequence of engineering questions, decisions, experiments, failures and refinements described below.

---

# Development Progression

## 1. Geometry, Collision Checking and Motion Planning

The first version of the project needed only a simplified surgical workspace, an instrument model, protected anatomical structures, a start configuration and a target.

To make this work, I first needed to understand:

- coordinate systems;
- basic 3-D geometry;
- distances and safety margins;
- joint configurations;
- collision checking;
- configuration-space motion planning.

When I researched collision-free robot motion planning, I encountered **sampling-based planning** and **Rapidly-exploring Random Trees (RRT)**.

RRT was a reasonable starting point because my problem involved searching a continuous joint/configuration space while rejecting configurations or edges that collided with protected structures. I implemented a collision-aware joint-space planner with deterministic seeds so that experiments could be repeated.

The first meaningful system was therefore:

**Geometry â†’ Collision Checking â†’ RRT Planning**

### Mathematics I learned or used

At this stage I relied mainly on:

- vectors and Euclidean distance;
- joint-space representations;
- coordinate transforms;
- geometric collision tests;
- interpolation between robot configurations.

The mathematics was not added for appearance. It was introduced because the planner needed a numerical way to represent where the robot and anatomy were and whether motion between configurations was safe.

---

## 2. Realising That Perfect Anatomy Was Unrealistic

The first planner could use exact simulated anatomical positions. This was useful for development but unrealistic as a model of a medical-robotics problem.

That created the next question:

> **What happens if the robot does not know exactly where the anatomy is?**

I separated the simulator's **hidden ground truth** from the anatomical estimate available to the navigation algorithm.

The simulator could still use the hidden truth to evaluate whether the system was actually safe, but deployable decision-making code was restricted to estimated information.

Instead of representing anatomy only as:

**position**

I began representing it as:

**estimated position + uncertainty**

This became a core design principle:

> **The navigation algorithm should not be allowed to use hidden simulator truth to make decisions.**

### Mathematics I learned or used

This led me to study:

- standard deviation;
- covariance;
- Gaussian-style uncertainty representations;
- how uncertainty can change geometric safety reasoning.

The key idea was that an estimate should communicate both **where the system thinks something is** and **how confident that estimate is**.

---

## 3. Discovering Active Perception

Once the estimate could be uncertain, I noticed another limitation: the robot was simply accepting the current observation.

That led to:

> **If the current camera view produces poor information, can the system deliberately choose a better viewpoint?**

Researching this question led me to **active perception**: the idea that an autonomous system can change how or where it senses in order to improve the information available for its task.

I implemented candidate viewpoints and a way of scoring them.

The perception process became:

**Observe â†’ Evaluate Candidate Views â†’ Select View â†’ Improve Estimate**

This was the first time the robot was not only reacting to sensed information, but making a decision about **how to sense**.

---

## 4. Moving From Generic to Task-Aware Perception

Reducing uncertainty everywhere is not necessarily useful for navigation. Uncertainty near the intended trajectory may matter much more than uncertainty in a region that the instrument will never approach.

That led to the central research question:

> **Can viewpoint selection prioritise information according to its relevance to the navigation task?**

I developed a task-aware scoring approach that considered factors such as:

- the planned trajectory;
- safety-critical regions;
- anatomical uncertainty;
- viewpoint quality;
- task relevance;
- viewpoint movement.

This became the **Task-Aware Active Perception** component.

The important distinction was:

**Generic active perception:** choose a view because it improves sensing in general.

**Task-aware active perception:** choose a view because it improves the information most relevant to the planned task.

I did not treat active perception itself as an original invention. My work was in understanding the concept, implementing and adapting it to the project, integrating it with navigation, and evaluating whether task-aware scoring produced useful behaviour.

---

## 5. Moving From Demonstrations to Experiments

A visually convincing simulation does not prove that one method is better than another.

That led to:

> **How do I test the method properly rather than selecting one example where it looks good?**

I started running repeated trials and recording machine-readable outputs.

I measured quantities including:

- planning success;
- safe-navigation rate;
- collisions;
- safety-margin violations;
- minimum clearance;
- path cost;
- localisation error;
- uncertainty.

This required me to learn more statistics than I had previously used.

### Statistics I learned or used

I investigated:

- paired comparisons;
- confidence intervals;
- bootstrap resampling;
- permutation/randomisation testing;
- effect measures;
- repeated-trial experimental design.

This changed the way I thought about the project. My aim became:

> **Design an experiment capable of showing whether the method works or does not work.**

rather than:

> **Find evidence that the method I prefer is better.**

---

## 6. Discovering Experimental Fairness Problems

As the experiments became larger, I realised that even a repeated comparison could be misleading if the strategies were not given comparable conditions.

For example, one viewpoint-selection strategy could appear better simply because it was allowed to move the camera farther.

This led me to investigate and implement:

- movement-budget matching;
- score normalisation;
- controlled candidate viewpoints;
- development, validation and held-out scenario splits;
- deterministic random seeds;
- frozen experimental configurations.

This became the Phase 1 fair-comparison framework.

The experiment was deliberately separated into development and held-out evaluation so that final scenarios were not continually retuned after observing the primary result.

This was one of the most important methodological lessons from the project: **fairness of the experimental design can matter as much as complexity of the algorithm.**

---

## 7. Retaining a Negative Held-Out Result

The held-out evaluation did **not** establish that the full Task-Aware strategy was superior to the movement-budget-matched Generic Active strategy.

For the primary held-out comparison:

- Task-Aware safe-navigation rate: **61.33%**
- movement-budget-matched Generic safe-navigation rate: **71.33%**
- difference: **-10 percentage points**
- 95% bootstrap confidence interval: **[-17, -3] percentage points**
- permutation p-value: **0.012390**

The result therefore favoured the matched Generic strategy on this endpoint.

I retained this result instead of changing the experiment after seeing it.

That result changed how I understood research. A negative result does not mean that the entire project failed. It can reveal that:

- the original hypothesis was too strong;
- a proposed mechanism may only help under certain conditions;
- the comparison may expose trade-offs;
- more detailed ablation or robustness experiments are needed.

This motivated further work on mechanism ablation, uncertainty stress testing, efficiency analysis and robustness.

---

## 8. Developing More Realistic Perception Components

As the project expanded, I investigated how uncertainty could arise from more realistic perception processes rather than only being assigned as a parameter.

This included work on:

- stereo perception uncertainty;
- segmentation;
- registration;
- tracking;
- uncertainty calibration;
- illumination degradation.

### Why U-Net-style segmentation appeared

When I researched biomedical image segmentation, I encountered **U-Net**, an established encoder-decoder architecture designed for biomedical image segmentation.

I implemented a small U-Net-style model as part of the perception experiments.

The project does not claim that U-Net itself is original. The value of this part was learning how a learned perception component could be tested and connected to uncertainty-aware navigation.

### Why calibration mattered

It is not enough for a system to output an uncertainty value. The value should have a meaningful relationship with observed error.

This led me to uncertainty-calibration experiments comparing predicted uncertainty with empirical behaviour.

---

## 9. Making Planning Respond to Uncertainty

A further limitation remained: even if perception reported uncertainty, the planner still needed to respond to it.

That led to:

> **Can greater uncertainty make the navigation system more conservative?**

I investigated risk-aware and chance-constrained planning concepts.

The basic engineering intuition was:

**effective protected region = physical structure + safety margin + uncertainty allowance**

This allowed uncertain anatomy to influence path feasibility and clearance rather than being treated as a separate diagnostic number.

The implementation is simulation-based and does not claim clinical probability-of-harm guarantees. The aim was to investigate how uncertainty information could affect navigation decisions.

---

## 10. Moving From One-Shot Planning to Closed-Loop Behaviour

Planning once and assuming nothing changes is not a strong model of autonomous execution.

This led to:

**Observe â†’ Plan â†’ Execute â†’ Observe Again â†’ Reassess â†’ Replan if Needed**

I added closed-loop experiments involving:

- changing uncertainty;
- tracking error;
- predicted clearance;
- replanning;
- safety interventions.

The project therefore moved from an offline planning experiment toward a runtime autonomy problem.

---

## 11. Developing Autonomous Safety Supervision

Once the system could execute a trajectory, I needed to consider what should happen if conditions became unsafe during execution.

The question became:

> **How should the system respond when continuing is no longer the safest action?**

I developed runtime monitoring and supervisory logic capable of behaviours such as:

**CONTINUE â†’ REACQUIRE â†’ RECOVER/REPLAN â†’ STOP**

I also used fault-injection-style experiments to deliberately create unsafe or degraded conditions.

This was important because safety logic should be tested when things go wrong, not only when navigation succeeds.

The thresholds used in the simulation are engineering/research parameters for the experimental system. They are **not clinically validated surgical safety thresholds**.

---

## 12. Why I Moved From PyBullet to ROS 2 and Gazebo

The software stack also developed in response to changing requirements.

### Why Python and PyBullet came first

Python allowed rapid implementation of numerical algorithms, experiments and tests.

PyBullet was useful during early algorithm development because I needed a relatively lightweight environment for repeated motion-planning and collision experiments without physical hardware.

### Why ROS 2 appeared later

As the project became less about isolated algorithms and more about communication between perception, planning, execution and safety components, I investigated robotics middleware and encountered **ROS 2**.

I then had to learn:

- nodes;
- topics;
- publishers/subscribers;
- messages;
- actions;
- launch files;
- parameters;
- robot descriptions.

ROS 2 was appropriate because it allowed the research components to be represented as separate runtime processes with explicit interfaces.

### Why Gazebo and ros2_control appeared

Once I wanted to test an integrated ROS runtime with robot state, controllers and execution rather than only algorithm-level simulation, I added Gazebo and ros2_control integration.

The final runtime therefore separates:

**Research algorithms â†’ ROS interfaces â†’ Runtime nodes â†’ Robot/controller simulation**

ROS 2, Gazebo and ros2_control are established infrastructure, not original contributions. My work was in learning how to use them, creating the project-specific interfaces and nodes, integrating the research logic, and validating the complete runtime behaviour.

---

## 13. What I Implemented Versus What I Used

This distinction is important.

### Established methods and technologies used

The project uses established ideas and tools including:

- RRT motion planning;
- U-Net-style segmentation;
- bootstrap statistics;
- permutation testing;
- active perception concepts;
- chance/risk-aware planning concepts;
- Python and scientific libraries;
- PyBullet;
- ROS 2;
- Gazebo;
- ros2_control;
- pytest.

I do **not** claim that these underlying methods or platforms are my inventions.

### My project contribution

My work focused on:

- learning and implementing the required algorithms;
- developing the uncertainty-aware navigation framework;
- separating hidden truth from deployable estimates;
- developing task-aware viewpoint scoring for the project;
- connecting perception uncertainty to planning and safety;
- designing controlled strategy comparisons;
- building held-out and reproducible evaluation pipelines;
- creating ablations, robustness and stress experiments;
- integrating the research code into a ROS 2 runtime;
- developing project-specific execution, monitoring and autonomous-supervision logic;
- creating custom ROS interfaces;
- validating behaviour through automated tests and experiments;
- documenting the architecture, requirements, evidence and limitations.

The project is therefore strongest as an **engineering and research-integration project**, rather than as a claim that every individual algorithm is novel.

---

## 14. Problems and Failures That Changed the Project

The project did not progress in a straight line.

Several difficulties exposed weaknesses and caused later stages to be redesigned.

### Unfair strategy comparison

Early active-perception comparisons raised the issue that strategies could have different movement budgets.

This led to movement-budget matching and a more rigorous Phase 1 protocol.

### Negative held-out result

The held-out evaluation did not support the expected superiority of the Task-Aware strategy.

Instead of retuning the final experiment, I preserved the result and used it to motivate further analysis.

### Runtime uncertainty behaviour

During ROS integration, I identified that a simulated perception uncertainty value could be cached rather than updated continuously during execution.

I changed the runtime so the uncertainty parameter could be read live on each publish cycle, allowing uncertainty changes to influence autonomous behaviour during the demonstration.

### Planning infeasibility

Not every requested target was valid or safely reachable. Some targets were correctly rejected by the planner.

This reinforced the distinction between a planner failing and a planner correctly refusing an unsafe or infeasible request.

### Software/environment problems

I encountered dependency, environment and ROS setup problems while working across Windows, WSL, Conda, ROS 2 and simulation tools.

These were part of the learning process and required checking which interpreter/environment was actually running rather than assuming a test failure was always a code failure.

### Legacy architecture

An earlier ROS workspace eventually became obsolete as the final ROS structure matured. It was later removed from the repository after confirming that the current implementation and tests no longer depended on it.

These problems are useful evidence of how the system evolved. They also explain why the final repository structure and experimental methodology differ from the earliest versions.

---

## 15. Verification and Reproducibility

As the codebase grew, I increasingly relied on automated verification rather than assuming that changes had not broken previous behaviour.

The final full research-core regression reached:

**1031 passing tests**

The final ROS 2 workspace validation reported:

**29 tests, 0 errors, 0 failures, 2 skipped**

The project also contains:

- deterministic seeds;
- machine-readable experiment outputs;
- a frozen experimental protocol;
- held-out scenario evaluation;
- architecture documentation;
- coordinate-frame documentation;
- requirements;
- a verification plan;
- a traceability matrix;
- a final evidence package;
- a reproducibility manifest.

The test counts should not be interpreted as clinical validation or proof that the research hypothesis is correct. They show that a large amount of expected software behaviour is automatically checked and that regression testing became part of the development process.

---

## 16. How the Codebase Fits Together

The research code is organised around four main areas:

```text
src/
â”œâ”€â”€ geometry/      # spatial representations, transforms and registration foundations
â”œâ”€â”€ perception/    # observations, camera models, uncertainty and active perception
â”œâ”€â”€ robotics/      # robot model, collision checking, planning, execution and safety logic
â””â”€â”€ simulation/    # experiments, benchmarks, statistics and evidence generation
```

The main conceptual flow is:

```text
SIMULATED WORLD / HIDDEN TRUTH
            â†“
        PERCEPTION
   estimate + uncertainty
            â†“
    ACTIVE PERCEPTION
 choose useful viewpoint
            â†“
         PLANNING
 collision/risk-aware RRT
            â†“
        EXECUTION
            â†“
   RUNTIME MONITORING
 clearance / tracking / uncertainty
            â†“
  AUTONOMOUS SUPERVISOR
   continue / recover / stop
            â†“
         RESULTS
 trials / statistics / evidence
```

The ROS runtime adds another software layer:

```text
ros2/src/
â”œâ”€â”€ surgical_navigation_interfaces/
â”œâ”€â”€ surgical_navigation_ros/
â”œâ”€â”€ surgical_navigation_description/
â””â”€â”€ surgical_navigation_bringup/
```

The custom interfaces define how project-specific data and navigation goals are communicated. The ROS nodes bridge perception, execution and safety logic. The robot-description and bringup packages support the simulated runtime.

---

## 17. What I Learned About Engineering Research

The most important change during the summer was not simply learning more libraries.

I learned to ask increasingly difficult questions about my own system:

- Is the comparison fair?
- Am I accidentally using simulator truth?
- Does the uncertainty value mean what I think it means?
- Does a visually good demonstration generalise?
- What happens in held-out scenarios?
- What if the result contradicts my hypothesis?
- What happens when the system becomes uncertain during execution?
- Can the robot recover?
- Can it refuse an unsafe request?
- Can another person understand and reproduce the evidence?

The project moved from:

> **Can I make the robot find a path?**

to:

> **How can a simulated surgical-navigation system reason about uncertain anatomy, actively improve its perception, plan with uncertainty, monitor execution, and autonomously respond when safety deteriorates?**

This progression was more important to me than simply increasing the size of the codebase.

---

## 18. What the Project Does Not Claim

The project remains a **simulation-based research and engineering framework**.

It does not claim:

- clinical validation;
- patient testing;
- certified medical-device safety;
- physical surgical-robot validation;
- that the simulated safety thresholds are clinically established;
- that ROS 2/Gazebo infrastructure was developed by me;
- that RRT, U-Net, active perception, bootstrap statistics or chance constraints are original algorithms created by me;
- that the Task-Aware strategy is universally superior.

The held-out evidence explicitly demonstrates why the last point matters.

The next major steps for a more mature research system would include physical hardware, real surgical/medical perception data, stronger state estimation and control, and independent external validation.

---

## 19. How I Would Explain the Project in One Minute

I started the summer after first year with a simple goal: make a simulated surgical instrument plan a collision-free path. Once that worked, I realised the planner was unrealistically using perfect anatomy, so I separated hidden truth from estimated anatomy and added uncertainty. That led me to active perception, because I wanted the system to improve poor observations rather than just accept them. I then developed a task-aware viewpoint strategy and, importantly, learned that proving it required fair experiments rather than one good demo. I introduced movement-budget matching, held-out evaluation and statistical testing, and kept the negative primary result when the task-aware strategy did not outperform the matched generic strategy. From there I investigated risk-aware planning, closed-loop execution and autonomous safety, and finally translated the research code into a ROS 2/Gazebo runtime. The final project is therefore as much a record of how I learned to research, test and challenge an engineering system as it is a robotics implementation.

---

# Learning and Technical References

These are examples of established sources relevant to methods and platforms used in the project. They make the academic and technical foundations explicit; they are not presented as sources of original project contributions.

1. S. M. LaValle, **â€œRapidly-Exploring Random Trees: A New Tool for Path Planning,â€** Technical Report, 1998.
   https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf

2. R. Bajcsy, **â€œActive Perception,â€** *Proceedings of the IEEE*, 76(8), 966â€“1005, 1988.
   https://doi.org/10.1109/5.5968

3. R. Bajcsy, Y. Aloimonos and J. K. Tsotsos, **â€œRevisiting Active Perception,â€** *Autonomous Robots*, 42, 177â€“196, 2018.
   https://doi.org/10.1007/s10514-017-9615-3

4. O. Ronneberger, P. Fischer and T. Brox, **â€œU-Net: Convolutional Networks for Biomedical Image Segmentation,â€** MICCAI, 2015.
   https://doi.org/10.1007/978-3-319-24574-4_28

5. B. Efron and R. J. Tibshirani, **_An Introduction to the Bootstrap_**, Chapman & Hall, 1993.

6. S. Dai, S. Schaffert, A. Jasour, A. Hofmann and B. Williams, **â€œChance Constrained Motion Planning for High-Dimensional Robots,â€** 2018.
   https://arxiv.org/abs/1811.03073

7. **ROS 2 Documentation â€” Interfaces: Topics, Services and Actions.**
   https://docs.ros.org/en/ros2_documentation/rolling/Concepts/Basic/Interfaces-Topics-Services-Actions.html

8. **ros2_control / gz_ros2_control Documentation.**
   https://control.ros.org/

9. **Gazebo Documentation.**
   https://gazebosim.org/docs/

This is a starting record rather than a claim that these were the only sources consulted. A future version could expand the bibliography and link individual references directly to design decisions.

---

## Related Repository Documentation

For technical details rather than the development story, see:

- [`architecture.md`](architecture.md)
- [`coordinate_frames.md`](coordinate_frames.md)
- [`experimental_protocol.md`](experimental_protocol.md)
- [`requirements.md`](requirements.md)
- [`traceability_matrix.md`](traceability_matrix.md)
- [`verification_plan.md`](verification_plan.md)
- [`final_ros2_runtime_evidence.md`](final_ros2_runtime_evidence.md)
- [`../results/README.md`](../results/README.md)
- [`../results/final_evidence/final_results_report.md`](../results/final_evidence/final_results_report.md)
