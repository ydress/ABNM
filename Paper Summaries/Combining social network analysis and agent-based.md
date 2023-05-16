# Combining social network analysis and agent-based modeling to explore dynamics of human interaction: A review

Author: Meike Will, Jürgen Groeneveld, Karin Frank, Birgit Müller
URL: https://doi.org/10.18174/sesmo.2020a16325

# Achievements

1. General overview about what has been done to combine SNA and ABM
2. “Guidelines that provide a basis for a comprehensive and clearly structured model set-up which supports the application of a systematic and quantitative analysis of social networks in agent-based models”
3. Ideas for future work:
    1. “Not only focus on the network as channels for transfer of material or non-material resources, but also design models where the network provides social integration”
    2. “Carefully determine the appropriate approach for the integration of social networks in ABM”
    3. “**Pay attention to structurally explicit analysis of the model”**

# Quotes

*”On a micro-level, agents act interdependently according to prescribed rules and adjust their behaviour to the current state of themselves, of other agents and of the environment
(Bonabeau, 2002; Railsback & Grimm, 2012).
On the macro-level, emergent patterns and dynamics arise from the aggregated individual behaviours and the interactions between the agents (Kiesling et al., 2012).”*

# Guidelines for Model Setup

## Network Definition

### (a) Nodes

**Level of aggregation**: *What is represented by an agent (c.f. “Collectives” in ODD design concepts)?*

The chosen subdivision that represents interacting partners has crucial influence on the network (Levin, 1992). Subdivisions depend on the level of decision making or action and the required level of accuracy but are limited by computational power (number of interacting agents grows fast if low level of aggregation is chosen). Possible subdivisions are:

- **Individual agents**: used in situations where the personal context is relevant (e.g. epidemics, opinion formation)
- **Households**: aggregated behaviour of family members or relevant decisions made by household head (e.g. land-use context: farmers, energy consumption: data availability on household level)
- **Firms**: similar to households but no relation to family (e.g. marketing: product diffusion can be either on individual or on firm level)
- **Higher level of aggregation** possible (e.g. regions, countries)

**Typology of agents**: *Which entities are grouped together and treated in a similar manner?*

Within the levels of aggregation, agents are grouped according to their attributes to allow generalizations of individual actors (Arneth et al., 2014). This includes classification on the same organisational level with same (e.g. green vs. conventional farmers, early vs. late adopters) and different functions (e.g. buyers vs. sellers) or across hierarchical organisational scales (e.g. land users vs. government).

### (b) Links

**Reciprocity**: *Are the links directed or undirected?*

Some problems need reciprocal links, some can deal with both but are probably more realistic with either directed or undirected links (opinion diffusion sometimes modelled in directed networks, sometimes in undirected), some need directed links (e.g. material transfer often only in one direction).

**Weight**: *Do the links include weighted relationships and preferences?*

Link strength allows including weighted relationships and preferences among neighbours. Link strength can be discrete (e.g. strong/weak) or continuous (assigning relative or absolute weights to links) and can be determined by the number of common contacts or emotional intensity such as trust or similarity of opinions.

### (c) Initialization

**Initial condition**: *Which links are present as initial conditions?*

The network formation can **start from scratch** with no links between the nodes established at the beginning of the simulation or with links set up according to a **specified topology**.

**Network topology**: *How are initial links motivated?*

If links are set up according to a specified topology, initial network topologies can be calibrated with **empirical data** or with **idealized topologies** (e.g. random, small-world or scale-free).

## Dynamics of the network

**Link formation**: *Why are links formed between agents?*

The formation of links between agents can be based on **agent properties** (e.g. spatial proximity or similarity), **probability**, **utility maximization**, etc.

**Network size**: *Does the number of nodes in the network vary during the simulation?*

Network size can be **static** if the network consists of the same nodes over the whole simulation or **dynamic** if the nodes vanish or appear during the simulation (e.g. due to extinction and reproduction processes or migration).

## Dynamics on the network

**Condition for interaction**: *When do agents interact?*

In some contexts, interaction takes place **independent of the network**. Thus, no condition on the interaction is needed (e.g. if influence of social norms is always present). Alternatively, a **threshold** (e.g. number of neighbours, fraction of neighbours, distance (spatially or between opinions), properties of neighbours etc.) has to be reached before the agents interact (Granovetter, 1978).

**Interaction direction and partners**: *Who do agents interact with?*

The **interaction direction** has a large effect on the dynamics on the network as it influences the direction of causality and therefore the relevance of the positions of the agents. The acting agent can either be influenced by the agents in its network (“in”, unidirectional) or influence other agents it has links to (“out”, unidirectional). Additionally, both agents can change their status based on the interaction (“both”, bidirectional). The acting agent can either pick one (“pairwise”), several (“selected”) or all other agents of its network as **interaction partners**.

![Untitled](Combining%20social%20network%20analysis%20and%20agent-based%20%203c810c022b9747888d479d7e5647ba7e/Untitled.png)

**Agent state transition**: *How do agents change their status?*

Change of agent state is influenced by processes like e.g. **imitation or averaging** or based on **probability**, **distance**, **utility**, etc. The representation of agent state (i.e. behaviour, opinion, health condition etc.) is possible either as **continuous traits** (e.g. opinions) or **distinct nominal categories** (e.g. product adoption levels, epidemics). Change of agent state is possible either in **one way** only (e.g. adoption of an innovation: once somebody has adopted a product he will never come back to the non-adopted state; opinion dynamics: models of assimilative social influence (Flache et al., 2017)) or in **two or more ways** (e.g. opinion dynamics: models with repulsive influence, opinions can be influenced positively or negatively (Flache et al., 2017; Jager & Amblard, 2005); epidemiology: agents can get infected but also recover from a disease).
