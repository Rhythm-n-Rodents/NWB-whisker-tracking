import os
from pynwb import load_namespaces, get_class

# Set path of the namespace.yaml file to the expected install location
ndx_whisk_specpath = os.path.join(
    os.path.dirname(__file__),
    'spec',
    'ndx-whisk.namespace.yaml'
)

# If the extension has not been installed yet but we are running directly from
# the git repo
if not os.path.exists(ndx_whisk_specpath):
    ndx_whisk_specpath = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..',
        'spec',
        'ndx-whisk.namespace.yaml'
    ))

# Load the namespace
load_namespaces(ndx_whisk_specpath)

# Define the class using get_class to make them accessible at the package level
WhiskerMeasurementTable = get_class('WhiskerMeasurementTable', 'ndx-whisk')