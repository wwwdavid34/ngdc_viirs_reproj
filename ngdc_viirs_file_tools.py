#!/usr/bin/env python

import os
import numpy as np
from datetime import datetime as dt
import jdcal
import h5py
import glob
from inspect import currentframe, getframeinfo


def ngdc_viirs_get_fname_adjacent(fname, search_dirs=None, previous=False, ext=None, time_tol=None):

    # Returns next adjacent in time filename matching product id in given
    # filename. If not adjacent file is found, returns null string. Can
    # also be sued to find previous in time filename using keyowrd PREVIOUS.
    # Will search same directory as input filename, can override using
    # SEARCH_DIR keyword. Will look for file with same extension as input
    # filename, can override using EXT keyword.

    # default time tolerance for time between files is 2 seconds
    if time_tol is None:
        time_tol = 2./60/60/24

    fname_metadata = ngdc_viirs_get_fname_metadata(fname)
    fname_start_time = fname_metadata.data_start_date.julian + \
        fname_metadata.data_start_time.julian
    fname_end_time = fname_metadata.data_start_date.julian + \
        fname_metadata.data_stop_time.julian
    if fname_end_time < fname_start_time:
        fname_end_time += 1

    if search_dirs is None:
        # search 2 layers up for pool - pool/aggregate/h5
        search_dirs = os.path.dirname(os.path.dirname(fname))
    if ext is None:
        ext = os.path.splitext(fname)[1]
        if ext == '':
            ext = '.h5'
    # convert search_dirs to list object
    if not isinstance(search_dirs, list):
        search_dirs = [search_dirs]
    search_glob = [os.path.join(i, '-'.join(fname_metadata.data_product_ids)) + '*' + ext for i in search_dirs]
    # search each search_glob, add result to files list if any
    files = []
    for sg in search_glob:
        f = glob.glob(sg)
        if len(f) > 0:
            files.extend(f)
    #print('glob_path', os.path.join(search_dirs, '-'.join(fname_metadata.data_product_ids) + '*' + ext))
    #files = glob.glob(os.path.join(search_dirs, '-'.join(fname_metadata.data_product_ids) + '*' + ext))
    
    # nf = len(files)
    # idx = np.where(np.array([os.path.basename(f) for f in files]) != os.path.basename(fname))
    #print('idx:', idx)
    #nf = len(idx)
    #if nf == 0:
    #    return ''

    #print('files', files)
    #files = np.array(files)[idx]
    files = [i for i in files if os.path.basename(i) != os.path.basename(fname)]
    nf = len(files)
    if nf == 0:
        return ''

    match_fnames_start_time = ['.'] * nf
    match_fnames_end_time = ['.'] * nf
    for i in range(0, nf):
        match_fname_metadata = ngdc_viirs_get_fname_metadata(files[i])
        match_fnames_start_time[i] = match_fname_metadata.data_start_date.julian + \
            match_fname_metadata.data_start_time.julian
        match_fnames_end_time[i] = match_fname_metadata.data_start_date.julian + \
            match_fname_metadata.data_stop_time.julian
        if match_fnames_end_time[i] < match_fnames_start_time[i]:
            match_fnames_end_time[i] += 1

    if previous:
        t1 = fname_start_time
        t2 = match_fnames_end_time
    else:
        t1 = match_fnames_start_time
        t2 = fname_end_time

    diff = np.abs(t1-t2)
    min_time_dist = min(diff)
    min_idx = np.where(diff == min_time_dist)
    if min_time_dist < time_tol:
        return np.array(files)[min_idx][0]

    return ''


def ngdc_viirs_get_fname_direction(h5_fname, direction_granules=None):

    # Get number of granules in this file
    # 'AggregateNumberGranule' is the attribute. It is found in the
    # h5 file structure as Data_Products/?/*Aggr

    # get the list of name hierachy in h5 file
    h5f = h5py.File(h5_fname)
    dataset_info = ngdc_h5_index_dataset(h5f)

    dataset_name = [i for i in dataset_info if i.endswith('Aggr')]
    if len(dataset_name) == 0:
        print('WARNING: Unable to find asc/des attribute in file %s. '
              'Returning "u" for unknown.' % os.path.basename(h5_fname))
        return 'u'

    attr_name = 'AggregateNumberGranules'
    if isinstance(dataset_name, list):
        dataset_name = dataset_name[0]
    n_granules = h5f[dataset_name].attrs[attr_name]
    while isinstance(n_granules, np.ndarray):
        n_granules = n_granules[0]
    # Get asc/des tag for each granule
    lend = len(dataset_name)
    data_prefix = dataset_name[0:lend-4] + 'Gran_'
    attr_name = 'Ascending/Descending_Indicator'

    direction_granules = ['0.'] * n_granules
    for i in range(0, n_granules):
        attr = h5f[data_prefix+str(i)].attrs[attr_name]
        direction_granules[i] = attr

    h5f.close()

    # 0->ascending, 1->descending
    n_des = sum(direction_granules)
    pct_des = float(n_des)/float(n_granules)

    direction = 'a'
    if pct_des >= .5:
        direction = 'd'

    return direction


def ngdc_viirs_get_fname_granule_status(h5_fname, granule_status=None):

    # Get number of granules in this file
    # 'AggregateNumberGranules' is the attribute. It is found in the
    # h5 file structure as Data_Products/?/*Aggr
    h5f = h5py.File(h5_fname)
    dataset_info = ngdc_h5_index_dataset(h5f)

    dataset_name = [i for i in dataset_info if i.endswith('Aggr')]
    if len(dataset_name) == 0:
        print('WARNING: Unable to find N_Granule_Status attribute in file %s' % h5_fname)
        return None
    else:
        dataset_name = dataset_name[0]

    attr_name = 'AggregateNumberGranules'
    n_granules = h5f[dataset_name].attrs[attr_name][0][0]

    # Get asc/dex tag for each granule
    lend = len(dataset_name)
    data_prefix = dataset_name[0:lend-4] + 'Gran_'
    attr_name = 'N_Granule_Status'

    granule_status = [None] * n_granules
    for i in range(0, n_granules):
        attr = h5f[data_prefix+str(i)].attrs[attr_name]
        granule_status[i] = attr[0][0]

    h5f.close()
    # N/A -> granule contains data
    # granule_status = np.array(granule_status)
    idx = np.where(np.array(granule_status) == b'N/A')

    return len(idx)


def ngdc_h5_index_dataset(tlg_id, tlg_name='/'):

    dataset_info = []
    tlg_id[tlg_name].visititems(lambda name, obj: dataset_info.append(name))

    return dataset_info


def ngdc_viirs_read_h5(h5_file, d_name):

    # example of a d_name 'All_Data/VIIRS-DNB-SDR_All/Radiance
    # return numpy array

    if not os.path.exists(h5_file):
        print('File not exist: ', h5_file)
        return None
    h5f = h5py.File(h5_file)
    try:
        return h5f[d_name]
    except KeyError:
        print('Key %s not found in %s' % (d_name, h5_file))
        return None


def ngdc_viirs_time_text_convert(time_text):
    hh = int(time_text[1:3])
    mm = int(time_text[3:5])
    ss = int(time_text[5:7])
    st = int(time_text[-1])

    time_jul = dt(2012, 1, 1, hh, mm, ss, int(st*1e5)) - dt(2012, 1, 1, 0, 0, 0)
    time_jul = (time_jul.seconds + time_jul.microseconds/1.0e6)/86400.

    return time_jul


def ngdc_viirs_date_text_convert(date_text):
    yyyy = date_text[1:5]
    mm = date_text[5:7]
    dd = date_text[7:9]
    date_jul = np.sum(jdcal.gcal2jd(yyyy, mm, dd))-0.5

    return date_jul


def ngdc_viirs_datetime_text_convert(datetime_text):

    '''
    Return julian date/time value for given datetime_text
    datetime_text should be given in 21 chars is format ?yyyymmddhhmmssuuuuuu
    where uuuuu = microseconds
    :param datetime_text:
    :return:
    '''

    yyyy = int(datetime_text[1:5])
    mm = int(datetime_text[5:7])
    dd = int(datetime_text[7:9])
    hh = int(datetime_text[9:11])
    mn = int(datetime_text[11:13])
    ss = int(datetime_text[13:15])
    uuuuuu = int(datetime_text[15:21])

    jd = np.sum(jdcal.gcal2jd(yyyy, mm, dd))-0.5
    jt = dt(2012, 1, 1, hh, mn, ss, uuuuuu) - dt(2012, 1, 1, 0, 0, 0)
    jt = (jt.seconds + jt.microseconds/1.0e6)/86400.
    date_jul = jd + jt

    return date_jul


def ngdc_viirs_h5_attr_exist(loc_id, attr_name):

    #num_attrs = len(list(loc_id.attrs))
    fi = getframeinfo(currentframe())
    for this_attr_name in list(loc_id.attrs):
        # this_attr_name = i
        try:
            # if this_attr_name == np.array(attr_name).astype(this_attr_name.dtype):
            if this_attr_name == str(attr_name):
                return True
        except TypeError:
            return False
    return False


def ngdc_viirs_get_geo_prefix(h5_datafile, no_tc_swap=False):

    h5f = h5py.File(h5_datafile)

    geo_prefix_lut = {'GMODO': 'GMTCO',
                      'GDNBO': 'GDTCN',
                      'GIMGO': 'GITCO',
                      'GMTCO': 'GMTCO',
                      'GITCO': 'GITCO'}

    h5_prefix_lut = {'SVI04': 'GITCO',
                     'SVI05': 'GITCO',
                     'SVDNB': 'GDNBO',
                     'SVM10': 'GMTCO'}

    geo_prefix = ''

    if ngdc_viirs_h5_attr_exist(h5f, 'N_GEO_Ref'):
        geo_fname = h5f.attrs['N_GEO_Ref'][0][0]
        geo_name_metadata = ngdc_viirs_get_fname_metadata(geo_fname)
        geo_prefix = geo_name_metadata.data_product_ids
        if no_tc_swap:
            try:
                geo_prefix = geo_prefix_lut[geo_prefix]
            except KeyError:
                print('KeyError')
                pass

    else:
        # hdf5 file DOES NOT have N_GEO_Ref tag
        # This is a lookup table for common types - add as needed.
        h5_fname_metadata = ngdc_viirs_get_fname_metadata(h5_datafile)
        h5_prefix = h5_fname_metadata.data_product_ids
        try:
            if isinstance(h5_prefix, list) and len(h5_prefix) > 1:
                print('More than one h5_prefix, use first one.')
                h5_prefix = h5_prefix[0]
            geo_prefix = h5_prefix_lut[h5_prefix]
        except KeyError:
            print('KeyError2')
            pass
    #else:
    #    print('No N_GEO_Ref attr.')

    h5f.close()

    return geo_prefix


class TimeObj(object):
    def __init__(self):
        self.text = ''
        self.julian = 0.


class OrbitObj(object):
    def __init__(self):
        self.text = ''
        self.val = 0.


class ViirsFnameMetadataObj(object):

    def __init__(self):
        self.data_product_ids = ''
        self.spacecraft_id = ''
        self.data_start_date = TimeObj()
        self.data_start_time = TimeObj()
        self.data_stop_time = TimeObj()
        self.orbit_number = OrbitObj()
        self.creation_date = TimeObj()
        self.origin = ''
        self.domain_description = ''


def ngdc_viirs_get_fname_metadata(viirs_fname):

    '''
    Return dictionary with metadata info retrieved from VIIRS filename
    Naming convention chosen to match NPP naming convention described
    in section 3.4.1 of NPOESS CDFCB V1, Rev F.
    :param viirs_fname:
    :return:
    '''

    field_separator = '_'
    product_separator = '-'
    # extention_separator = '.'

    # First remove any path and .extensions from filename. It is assumed that
    # there are no '.' characters in the filename metadata
    if isinstance(viirs_fname, type(b'')):
        viirs_fname = viirs_fname.decode('ascii')
    viirs_basename = os.path.basename(os.path.splitext(viirs_fname)[0])

    # Extract metadata into string array by fields
    viirs_parts = str(viirs_basename).split(field_separator)

    # Extract data product ids, count number in this filename - will be used
    # to define filename metadata structure
    data_product_ids = str(viirs_parts[0]).split(product_separator)
    n_data_product_ids = len(data_product_ids)
    if n_data_product_ids == 1:
        data_product_ids = data_product_ids[0]

    # Define filename metadata structure
    viirs_fname_metadata = ViirsFnameMetadataObj()

    # Assign values to filename metadata structure. String values are always
    # kept and numeric values are assigned where appropriate.
    viirs_fname_metadata.data_product_ids = data_product_ids
    viirs_fname_metadata.spacecraft_id = viirs_parts[1]
    viirs_fname_metadata.data_start_date.text = viirs_parts[2]
    viirs_fname_metadata.data_start_time.text = viirs_parts[3]
    viirs_fname_metadata.data_stop_time.text = viirs_parts[4]
    viirs_fname_metadata.orbit_number.text = viirs_parts[5]
    viirs_fname_metadata.creation_date.text = viirs_parts[6]
    viirs_fname_metadata.origin = viirs_parts[7]
    viirs_fname_metadata.domain_description = viirs_parts[7]

    # Assign numeric orbit number
    viirs_fname_metadata.orbit_number.val = \
            int(viirs_fname_metadata.orbit_number.text[1:6])

    # Assign numeric date/time values
    viirs_fname_metadata.data_start_time.julian = \
        ngdc_viirs_time_text_convert(viirs_fname_metadata.data_start_time.text)
    viirs_fname_metadata.data_stop_time.julian = \
        ngdc_viirs_time_text_convert(viirs_fname_metadata.data_stop_time.text)
    viirs_fname_metadata.data_start_date.julian = \
        ngdc_viirs_date_text_convert(viirs_fname_metadata.data_start_date.text)
    viirs_fname_metadata.creation_date.julian = \
        ngdc_viirs_datetime_text_convert(viirs_fname_metadata.creation_date.text)

    # viirs_fname_metadata = {
    #     'data_product_ids': data_product_ids,
    #     'spacecraft_id': '',
    #     'data_start_date': {'text': '', 'julian': 0.0},
    #     'data_start_time': {'text': '', 'julian': 0.0},
    #     'data_stop_time': {'text': '', 'julian': 0.0},
    #     'orbit_number': {'text': '', 'val': 0},
    #     'creation_date': {'text': '', 'julian': 0},
    #     'origin': '',
    #     'domain_description': ''
    # }
    # nt = len(list(viirs_fname_metadata.keys()))
    #
    # # Assign values to filename metadata structure. String values are always
    # # kept and numeric values are assigned where appropriate.
    # viirs_fname_metadata['data_product_ids'] = data_product_ids
    # viirs_fname_metadata['spacecraft_id'] = viirs_parts[1]
    # viirs_fname_metadata['data_start_date']['text'] = viirs_parts[2]
    # viirs_fname_metadata['data_start_time']['text'] = viirs_parts[3]
    # viirs_fname_metadata['data_stop_time']['text'] = viirs_parts[4]
    # viirs_fname_metadata['orbit_number']['text'] = viirs_parts[5]
    # viirs_fname_metadata['creation_date']['text'] = viirs_parts[6]
    # viirs_fname_metadata['origin'] = viirs_parts[7]
    # viirs_fname_metadata['domain_description'] = viirs_parts[7]
    #
    # # Assign numeric orbit number
    # viirs_fname_metadata['orbit_number']['val'] = \
    #     int(viirs_fname_metadata['orbit_number']['text'])
    #
    # # Assign numeric date/time values
    # viirs_fname_metadata['data_start_time']['julian'] = \
    #     ngdc_viirs_time_text_convert(viirs_fname_metadata['data_start_time']['text'])
    # viirs_fname_metadata['data_stop_time']['julian'] = \
    #     ngdc_viirs_time_text_convert(viirs_fname_metadata['data_stop_time']['text'])
    # viirs_fname_metadata['data_start_date']['julian'] = \
    #     ngdc_viirs_date_text_convert(viirs_fname_metadata['data_start_date']['text'])
    # viirs_fname_metadata['creation_date']['julian'] = \
    #     ngdc_viirs_datetime_text_convert(viirs_fname_metadata['creation_date']['text'])

    return viirs_fname_metadata
