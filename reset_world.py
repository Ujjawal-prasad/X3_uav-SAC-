import subprocess

cmd = [
    "gz", "service",
    "-s", "world/quadcopter/control",
    "--reqtype", "gz.msgs.WorldControl",
    "--reptype", "gz.msgs.Boolean",
    "--req", "reset: {all: true}"
]


try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("Command executed successfully!")
    print("Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error running command:")
    print(e.stderr)
