from tasks import generate


output = generate.delay(
    "12345",
    {
        "prompt": "deer shaped lamp",
        "ddim_steps": 30,
        "ddim_eta": 0,
        "n_iter": 1,
        "W": 256,
        "H": 256,
        "n_samples": 2,
        "scale": 5,
        "plms": True,
    },
)

# Get task result
print(output.get())
