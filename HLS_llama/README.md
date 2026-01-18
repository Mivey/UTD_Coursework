# Design Space Exploration of LLMs in HLS for Heterogeneous SoC Applications (Working Title)

**Thesis Advisor:** Dr. Benjamin Schaefer

## Thesis Objectives
The primary goal of this thesis is to explore possible heterogeneous architectures that can be used to implement an llm on a lower cost versal part. My work is based in part on Andrej Karpathy's 'llama2.c' (insert link to repo) with the idea being to implement in full the forward pass of a decoder based transformer. 
This snapshot serves more as a sample of one of the branches I am working on, and highlights the Head Major architecture I implemented which elminated cache thrashing and significantly increasd DDR resource utiliztion.
Future work on this subject will be primarily centered around the implementation of AI-E for matrix multiplication, and stratiges around resource reuse.