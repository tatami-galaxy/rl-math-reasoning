#### Background

Language models post-trained for reasoning tasks (math, code) are trained to produce long "chains of thoughts" (COT) before producing the final output [cite].  The consensus behind why this enhances model performance is that the number of serial computations a transformer based LM can perform is bounded by constant model depth. COT allows a constant-depth transformer to tackle problems that require sequential computation, by unrolling the computation across time-steps [cite]. This is also supported by evidence that model performance grows with increasing COT length and test-time compute. 


#### Research Questions

There are several questions that we might have about these reasoning traces. Broadly they maybe classified into two categories: 

- Effiency : Assuming the model primarily uses COT as a mechanism to perform sequential computation, how efficient is this computation? That is, how efficently does the model utilize its COT token budget? Given a prompt and a generated COT, is there a more concise COT that allows the same model to perform the same computation?

- Semantics : Are the traces semantically meaningful to the end user? There is a growing body of research that suggests that often these traces contain decorative segments from the perspective of the user. That is, they are not tied to the computation that they semantically indicate [cite]. 

We are primarily interested in these questions in the context of different training setups. That is, how different training methods affect efficiency and semantics of COT. Given some desirable set of COT behaviors at test-time with respect to efficiency and semantics, we are also interested in asking whether we can come up with training methodologies that can enable such behaviors.


Training on traces from different model

RL-zero training