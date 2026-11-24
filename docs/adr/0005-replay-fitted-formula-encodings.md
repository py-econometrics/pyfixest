# Replay fitted formula encodings on new data

`predict` and the DID2S covariance evaluate data through the Formulaic model
specification saved when the model was fitted. They never parse the formula
again. Parsing again learns the categorical levels and reference levels from
the new data, forgets which rows the fit dropped, and evaluates a fixed-effect
interaction such as `a^b` as Python's XOR.

## Consequences

- A prediction row with an unseen level gets a missing prediction. It is never
  encoded as the reference level.
