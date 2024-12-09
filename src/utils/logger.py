# coding: utf-8
# @email: enoche.chow@gmail.com

"""
###############################
"""

import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
    """
    # Absolute path for log directory (adjusted for Colab environment)
    LOGROOT = '/content/log/'
    os.makedirs(LOGROOT, exist_ok=True)  # Create the log directory if it doesn't exist

    # Construct the log filename
    logfilename = '{}-{}-{}.log'.format(
        config.final_config_dict.get('model', 'unknown'),
        config.final_config_dict.get('dataset', 'unknown'),
        get_local_time()
    )
    logfilepath = os.path.join(LOGROOT, logfilename)

    # Debugging: Print the log file path
    print(f"Log file will be saved at: {logfilepath}")

    # Define file and stream formatters
    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = u"%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)

    # Determine logging level based on 'state'
    state = config.final_config_dict.get('state', 'info').lower()
    level = {
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'critical': logging.CRITICAL
    }.get(state, logging.INFO)  # Default to INFO

    # File handler
    try:
        fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
        fh.setLevel(level)
        fh.setFormatter(fileformatter)
    except Exception as e:
        print(f"Error initializing FileHandler: {e}")
        fh = None

    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    # Configure logging
    handlers = [sh]
    if fh:  # Add FileHandler only if initialized successfully
        handlers.append(fh)

    logging.basicConfig(
        level=level,
        handlers=handlers
    )
