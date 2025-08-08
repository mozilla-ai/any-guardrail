## Contributing Guardrails

If a guardrail is not available, fork this repo to add it and then issue a pull request. Please use the following steps:

### Step 1: Create a `Guardrail` class

We have an abstract `Guardrail` class that has the minimum api required to create a new guardrail. To do so, implement the following:

```python
class YourGuardrail(Guardrail):
    def __init__(self, ...):
        super().__init__(model_id)
        self.guardrail = _model_instantiation(model_id, ...)

    def safety_review(...):
        # Your implementation for reviewing text

    def _model_instantiation(...):
        # Your implementation for instantiating a model
```

For more detailed examples, we recommend looking through the `guardrails` directory.

### Step 2: Add your model to the `model_registry.py`

Now that you have created `YourGuardrail`, you need add a model identifier to help the `AnyGuardrail` identify your guardrail. It will look something like this:

```python

model_registry= {
    "already/implemented/guardrail": SomeGuardrail,
    ...
    "your/guardrail/identifier": YourGuardrail
}
```

From there, you should be all set! Send a PR to our main repo, so we can review and add your guardrail.
