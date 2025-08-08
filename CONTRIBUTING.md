## Contributing Guardrails

If a guardrail is not available, fork this repo to add it and then issue a pull request. Please use the following steps:

### Create a new Class that Inherits from Guardrail

We have an abstract `Guardrail` class that has the minimum api required to create a new guardrail.

Create a new file in src/any_guardrail/guardrails for your guardrail, and create a class that inherits all the abstract
methods from Guardrail.

See an existing guardrail for implementation hints.

#### Add your provider to GuardrailName

Create a new enum value for your guardrail. The string should follow lower_case while the enum key should be UPPER_CASE.

### Create PR

From there, you should be all set! Send a PR to our main repo, so we can review and add your guardrail.
