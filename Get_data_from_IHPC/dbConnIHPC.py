""" all functions needed for data transmission between IHPC """
import time
import logging
import base64
import requests
import json
import pandas as pd
from requests.auth import HTTPBasicAuth

# API_key = "sutd-nse_api:dj6M9RAxynrjw9aWztzprfh5AKHssgVj4qKXiKSfHRyGKeoX92wmwmEJKpHMIB5"
API_key = "dj6M9RAxynrjw9aWztzprfh5AKHssgVj4qKXiKSfHRyGKeoX92wmwmEJKpHMIB5"
protal_username = 'sutd-nse_api'


def getDataIHPC(url, nid, start_time, end_time, table=None):
    """Retrieve raw hardware data for device nid for the specified time
    frame and specified table if specified.  Return a pandas data frame of
    measurements or None if no data was returned.

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'nid': nid, 'start': start_time, 'end': end_time, 'ts': int(time.time())}
    if table:
        payload['table'] = table

    req = \
        requests.get("%s/getdatafull" % url, params=payload, auth=HTTPBasicAuth(protal_username, API_key), verify=False)
    logging.debug("getdata url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getData returned http status %d" % req.status_code)
    resp = req.json()
    # resp["success"] has type bool
    if resp["success"]:
        return pd.DataFrame(resp["data"])
    else:
        logging.warning("getData for " + str(nid) + " for end: " + str(end_time) + " returned %s" % resp["error"])
        return None


def getStatus(url, nid, date):
    """Get the processed/unprocessed status for device nid on the
    specified date (date format 'YYYY-MM-dd') using the specified API
    url. Return True (processed) or False (not processed yet).

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'nid': nid, 'date': date, 'ts': int(time.time())}

    header = {"Content-Type": "application/json", 'Authorization': 'Basic %s' % base64.b64encode(API_key)}
    req = requests.get("%s/getanalysestatus" % url, params=payload, headers=header)
    logging.debug("getStatus url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getStatus returned http status %d" % req.status_code)
    resp = req.json()
    if not resp["success"]:
        raise Exception("getStatus returned error message %s" % resp["error"])
    stat = resp["status"]
    logging.debug("getStatus has value: %s" % str(stat))
    return stat == 1


def setStatus(url, nid, date, status):
    """Set the processed/unprocessed status for device nid on the
    specified date (date format 'YYYY-MM-dd') using the specified API
    url. Return boolean to indicate if setting the status was
    successful.

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'nid': nid, 'date': date, 'status': status}

    header = {"Content-Type": "application/json", 'Authorization': 'Basic %s' % base64.b64encode(API_key)}
    req = requests.get("%s/updateanalysestatus" % url, params=payload, headers=header)
    logging.debug("setStatus url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("setStatus returned http status %d" % req.status_code)
    resp = req.json()
    return resp["success"]


def saveModeIHPC(url, nid, timestamps, modes):
    """Save predicted travel modes for device nid. Modes is a list of
    predicted modes, timestamps is the list of timestamps of the
    measurements. The order of the predicted modes and timestamps must
    be the same. Return True if successful and False otherwise.

    """
    header = {"Accept": "application/json", "Content-Type": "application/json",
              'Authorization': 'Basic %s' % base64.b64encode(API_key)}
    if modes is None or timestamps is None or len(modes) != len(timestamps):
        raise ValueError("modes and timestamps must be of the same length and not None.")
    to_dict = lambda x: {"timestamp": long(x[0]), "cmode": int(x[1])}
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'nid': nid, 'cmodes': map(to_dict, izip(timestamps, modes))}
    req = requests.post("%s/importcmode" % url, data=json.dumps(payload), headers=header)
    if req.status_code != requests.codes.ok:
        logging.warning("saveMode returned http status %d" % req.status_code)
        return False
    resp = req.json()
    # resp["success"] has type bool
    stat = resp["success"]
    logging.debug("saveMode has value: %s" % str(stat))
    return stat


def saveTrips(url, nid, date, payload):
    """Save trips for device nid on the specified date (date format
    'YYYY-MM-dd') using the specified API url. Trips is a list of
    identified trips, each trip is dictionary containing the start
    time, end time, overall travel mode. Return True if successful and
    False otherwise.

    1. The analytics result contains maximum following 12 keys:
        nid: int
        date: string
        am_mode : list
        am_distance : list
        pm_mode : list
        pm_distance : list
        travel_co2 : float
        outdoor_time : float
        home_loc : tuple (lat:lon)
        school_loc : tuple (lat:lon)
        poi_lat : list
        poi_lon : list
    2. besides nid & date, all other keys could be missing the the json file.
    3. If the keys are present, their values will replace previous values with same key.

    """
    if payload == None:
        logging.warning("When saving trips to backend, payload must not be None.")
        return False

    payload['nid'] = nid
    payload['date'] = date
    #    logging.warning(payload)

    logging.warning("Saving analytic summary of " + str(nid) + " on " + str(date))

    header = {"Accept": "application/json", "Content-Type": "application/json",
              'Authorization': 'Basic %s' % base64.b64encode(API_key)}

    req = requests.post("%s/importanalysedsummary" % url, data=json.dumps(payload), headers=header)

    #    logging.warning("json dump to the API: "+json.dumps(payload))

    if req.status_code != requests.codes.ok:
        logging.warning("saveTrips returned http status %d" % req.status_code)
        return False
    resp = req.json()
    # resp["success"] has type bool
    stat = resp["success"]
    #    logging.warning("saveTrips has return value: %s" % str(stat))
    if not stat:
        logging.warning("saveTrips returned error message %s" % resp["error"])
    return stat
