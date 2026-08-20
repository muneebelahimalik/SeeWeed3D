"""One file per dataset, so several can exist without editing one another.

`make_dataset.py` holds a single CONFIG, which is right when there is one
dataset and wrong the moment there are three. Editing it back and forth between
an onion build, a weed build and a mixed build is how a stale DATASET_DIR
survives into a training run, and how the same key ended up in that CONFIG
twice.

Each module here is a thin override: it imports the shared CONFIG, replaces the
handful of keys that differ, and calls the same `main`. Everything not named
stays the shared default, so a fix to the defaults reaches every dataset.
"""
