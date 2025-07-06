# check_imports.py - Place this file in your docs/ directory temporarily

import os
import sys

# This path should exactly match the sys.path.insert line in your conf.py
# It's relative to the location of this script (docs/)
sys.path.insert(0, os.path.abspath("../src"))


print(f"Current working directory: {os.getcwd()}")
print(f"Python path (sys.path): {sys.path}")
print("-" * 30)

# --- Test importing your top-level package ---
try:
    import connectoviz

    print("Successfully imported 'connectoviz' package!")
    if hasattr(connectoviz, "__version__"):
        print(f"  connectoviz.__version__: {connectoviz.__version__}")
except Exception as e:
    print(f"Failed to import 'connectoviz': {e}")
    print(
        "  Hint: Check if 'connectoviz' package (containing __init__.py) is directly under the 'src' directory."
    )
    print(
        "  Hint: Ensure 'src/connectoviz/__init__.py' exists and is valid Python code."
    )

print("-" * 30)

# --- Test importing a specific module you use in api.rst ---
try:
    import connectoviz.plotting.circular_graph

    print("Successfully imported 'connectoviz.plotting.circular_graph'!")
    # Optionally, test a function from it
    # from connectoviz.plotting.circular_graph import plot_circular_connectome
    # print(f"  Found plot_circular_connectome: {plot_circular_connectome}")
except Exception as e:
    print(f"Failed to import 'connectoviz.plotting.circular_graph': {e}")
    print("  Hint: Check for missing __init__.py in 'plotting' or 'circular_graph.py'.")
    print(
        "  Hint: Check for syntax errors or missing dependencies within 'circular_graph.py' itself."
    )

print("-" * 30)

# --- Test another specific module ---
try:
    import connectoviz.core.connectome

    print("Successfully imported 'connectoviz.core.connectome'!")
except Exception as e:
    print(f"Failed to import 'connectoviz.core.connectome': {e}")
    print("  Hint: Check for missing __init__.py in 'core' or 'connectome.py'.")
    print(
        "  Hint: Check for syntax errors or missing dependencies within 'connectome.py' itself."
    )

print("-" * 30)

print("\n--- Import test complete ---")

# You can uncomment this line if you want the script to exit immediately after running
# sys.exit(0)


# Add this to your check_imports.py
print("-" * 30)
try:
    import connectoviz.visualization.circular_plot_builder

    print("Successfully imported 'connectoviz.visualization.circular_plot_builder'!")
except Exception as e:
    print(f"Failed to import 'connectoviz.visualization.circular_plot_builder': {e}")
    print(
        "  Hint: Check for missing __init__.py in 'visualization' or 'circular_plot_builder.py'."
    )
    print(
        "  Hint: Check for syntax errors or missing dependencies within 'circular_plot_builder.py' itself."
    )
print("-" * 30)
