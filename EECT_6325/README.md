# EECT 6325: VLSI Design

**Instructor:** Dr. Carl Sechen

## Course Objectives
This course served as a rigorous foundation in CMOS integrated circuit design, bridging the gap between transistor physics and digital system architecture. The curriculum focused on the "Logical Effort" method for delay optimization, static/dynamic logic families, and the physical realities of the manufacturing process.

## Featured Project: Standard Cell Library Design & Characterization
**Process Node:** 65nm

**Tools:** Cadence Virtuoso, Cadence Innovus (P&R/Timing), Mentor Calibre (DRC/LVS)

This comprehensive project simulated the full ASIC design lifecycle, moving from transistor-level schematic capture to physical layout and verification.

### Key Technical Achievements
* **Full-Custom Cell Design:** Designed and characterized a custom standard cell library comprising over **24 unique logic gates** (NAND, NOR, XOR, AOI/OAI), optimizing transistor sizing for symmetric rise/fall times.
* **Design Automation:** Developed custom **Bash scripts** to automate the repetitive tasks of layout generation, netlist extraction, and verification, significantly reducing the iteration time for library characterization.
* **Physical Verification:** Utilized **Mentor Calibre** to perform Design Rule Checks (DRC) and Layout Versus Schematic (LVS) verification, ensuring the library was free of geometric violations and electrically accurate.
* **ASIC Implementation Flow:** Gained hands-on experience with the complete digital backend flow, including floorplanning, placement, and routing using **Cadence Innovus**, validating timing constraints through Static Timing Analysis (STA).