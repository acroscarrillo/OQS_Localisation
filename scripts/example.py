# The following code is only needed to run the script through the run
# without debugging command, or to run using the cell syntax # %%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# The preceding code is only needed to run the script through the run
# without debugging command, or to run using the cell syntax # %%

# The below code can be run on its own line by line interactively using
# shift enter, as this runs in working directory (where test.py is located). 
# Unfortunately although this isn't a particularly nice experience as it
# doesn't handle multi-line code blocks well, and doesn't move to the next
# line automatically when you run a line.
from src import *

print(delta(1, 2))
print(delta(2, 2))