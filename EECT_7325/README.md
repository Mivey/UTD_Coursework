# EECT 7325: Advanced VLSI Design

**Instructor:** Dr. Carl Sechen

## Course Objectives
Building upon the foundations of EECT 6325, this advanced course focused on deep-submicron circuit design, analyzing the physical and electrical challenges of modern process nodes (7nm). The curriculum emphasized custom layout techniques, logical effort sizing, and delay optimization in the absence of automated Place & Route (P&R) tools.

## Featured Project: 12-bit Carry Select Adder (7nm)
**Process Node:** 7nm (Predictive Technology Model)
**Tools:** Cadence Virtuoso (Schematic & Layout XL), Mentor Calibre

Due to licensing constraints on automated P&R tools (Innovus), this project required a **full-custom manual layout** approach. This constraint provided a unique opportunity to understand the physical routing complexities and density challenges at the 7nm node.

### Key Technical Achievements
* **Arithmetic Logic Architecture:** Designed and implemented a **12-bit Carry Select Adder (CSA)**, selected for its balance of speed and area compared to Ripple Carry adders.
* **Hierarchical Layout Methodology:** Decomposed the 12-bit architecture into modular 1-bit standard cells (as seen in `carry_sel.v`) to manage complexity. This "bottom-up" approach ensured robust connectivity and Design Rule Check (DRC) compliance before top-level integration.
* **Physical Verification:** Validated layout connectivity against the schematic using **[Assura / PVS] LVS (Layout Versus Schematic)**, ensuring atomic-level correctness for every bit-slice.
* **Technology Scaling Analysis:** Observed and analyzed the dramatic density scaling and parasitic implications of 7nm FinFET technology compared to legacy 65nm planar nodes.