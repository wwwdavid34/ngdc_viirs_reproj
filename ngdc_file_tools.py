#!/usr/bin/env

import os
import numpy as np
from datetime import datetime as dt
import gzip
from congrid import congrid
import struct


# Define file_info structure
class FileInfoObj(object):
    # Original IDL code below
    # file_info = {'fname': '', 'fname_hdr': '', 'compress': False}
    def __init__(self):
        self.fname = ''
        self.fname_hdr = ''
        self.compress = False


def ngdc_get_fnames(filename, newfile=False):

    '''
    The ngdc_get_fnames function returns filename information about a requested
    file. If file does not exist on disk, the null string is returned. To get
    information about a new file (i.e. one the calling code will create), set the
    newfile ekyword.
    :param filename:
    :param newfile:
    :return:
    '''

    # Fill initial state of file_info structure
    file_info = FileInfoObj()
    file_info.fname = filename
    if os.path.split(filename)[1] == '.gz':
        file_info.fname_hdr=os.path.splitext(filename)[0]+'.hdr'
        file_info.compress=True
    else:
        file_info.fname_hdr = filename+'.hdr'
        file_info.compress=False

    # Verify existence of files unless newfile keyword is set
    if not newfile:
        file_exists = os.path.exists(file_info.fname)

        if not file_exists and file_info.compress is False:
            # see if there's a gzipped version
            fname = filename + '.gz'
            file_exists = os.path.exists(fname)
            if file_exists:
                file_info.fname = fname
                file_info.compress = True

        # If file is still not found, reset appropriate info structure tags to
        # contain null values
        if not file_exists:
            file_info.fname = ''
            file_info.compress = False

        # Look for hdr file, set to null string if not found
        file_exists = os.path.exists(file_info.fname_hdr)
        if not file_exists:
            file_info.fname_hdr = ''

    return file_info


def dmsp_get_fnames(filename, newfile=False):

    return ngdc_get_fnames(filename, newfile=newfile)


def dmsp_file_size_check(file_name):

    '''
    The DMSP_FILE_SIZE_CHECK function checks the disk file size against the file
    size computed by the parameters in the ENVI header. This function works on
    uncompressed files and gzip compressed files.
    :param file_name:
    :return:
    '''

    # Error check input parameter
    # file_info = ngdc_get_fnames(file_name)
    # hdr_info = envi_hdr(file_info.fname_hdr)
    # print(hdr_info)
    # type = hdr_info['data type']
    # if type != 8:
    # print(type(file_name))
    # print(type(FileInfoObj()))
    if type(file_name) != type(FileInfoObj()):
        # Assume routine was given a filename - get structure
        file_info = dmsp_get_fnames(file_name)
        fname = file_name
    else:
        file_info = file_name
        fname = file_info.fname

    if file_info.fname == '' or file_info.fname_hdr == '':
        print('ERROR: Processing filename %s' % fname)
        print('ERROR: Unable to find input filename and/or its ENVI header.')
        return -1

    # Get uncompressed file size information
    file_size = None
    if file_info.compress:
        file_size = _get_uncompressed_gz_size(fname)

    if not file_info.compress:
        file_size = os.path.getsize(fname)

    # Get dimensions & data type from ENVI header
    hdr_info = envi_hdr(file_info.fname_hdr)
    ns = hdr_info['samples']
    nl = hdr_info['lines']
    dt = hdr_info['data type']
    correct_file_size = ns * nl * lut_bytes_per_pixel(dt)

    return file_size == correct_file_size


def lut_bytes_per_pixel(dt):

    # Returns the size of one element of data given the IDL data type.
    # example: long integers have IDL data type 3, lut[3]=4. A
    # return value of 0 indicates a variable value (i.e. strings)

    lut = [0, 1, 2, 4, 4, 8, 0, 0, 0, 0, 0, 0, 2, 4, 8, 8]

    return lut[dt]


def _get_uncompressed_gz_size(filename):
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def ngdc_envi_file_integrity_check(fname):

    # Make sure ENVI file has a .hdr, and is the correct size for the number
    # of bands/lines/smaples/datatype.

    # Return 1 (true) if ENVI file is good, otherwise 0.

    fname_info = ngdc_get_fnames(fname)
    check_exist = os.path.exists(fname_info.fname) * os.path.exists(fname_info.fname_hdr)
    if not check_exist:
        return check_exist

    check_size = dmsp_file_size_check(fname_info)
    return check_size


def ngdc_get_image_from_file(file, resample_pct=None):

    fstruct = ngdc_get_fnames(file)

    if fstruct.fname == '':
        print('ERROR: FILE NOT FOUND %s' % file)
        return None
    if fstruct.fname_hdr == '':
        print('ERROR: FILE HEADER NOT FOUND FOR %s' % file)
        return None
    hdr_meta = ngdc_read_envi_hdr(fstruct.fname_hdr)
    ns = hdr_meta['n_samples']
    nl = hdr_meta['n_lines']
    dt = hdr_meta['data type']

    if resample_pct is not None:
        pct = resample_pct
    else:
        pct = 1
    ns_out = np.floor(pct*ns)
    nl_out = np.floor(pct*nl)
    dtype = dt_lookup(dt)

    if fstruct.compress:
        f = gzip.GzipFile(file)
        fc = f.read()
        image = np.frombuffer(fc, dtype=dtype)
        f.close()
    else:
        image = np.fromfile(file, dtype=dtype)
    image = np.reshape(image, (nl, ns))

    if pct != 1:
        return congrid(image, [nl_out, ns_out])
    else:
        return image


def ngdc_read_envi_hdr(filename):

    return envi_hdr(filename)


def envi_hdr(filename):

    '''
    The ENVI_HDR routine reads in a ENVI data file header and returns values in dictionary.
    @ author Kimberly Baugh
    :param filename:
    :return:
    '''
    print('hdr filename:', filename)
    if not os.path.exists(filename):
        raise FileNotFoundError('File not exist.')

    # Read in ASCII header
    f = open(filename, 'r')

    try:
        a_envi_file = f.readline().strip().startswith('ENVI')
    except UnicodeDecodeError:
        f.close()
        print('File %s is not a text file but a binary.' % filename)
        raise
    else:
        if not a_envi_file:
            raise RuntimeError('This is not an ENVI header file.')

    lines = f.readlines()
    f.close()

    dict_hdr = {}
    # dict = hdr_obj()
    have_nonlowercase_param = False
    support_nonlowercase_params = False  # settings.envi_support_nonlowercase_params
    try:
        while lines:
            line = lines.pop(0)
            if line.find('=') == -1:
                continue
            if line[0] == ';':
                continue

            (key, sep, val) = line.partition('=')
            key = key.strip()
            if not key.islower():
                have_nonlowercase_param = True
                if not support_nonlowercase_params:
                    key = key.lower()
            val = val.strip()
            if val and val[0] == '{':
                strr = val.strip()
                while strr[-1] != '}':
                    line = lines.pop(0)
                    if line[0] == ';':
                        continue

                    strr += '\n' + line.strip()
                if key == 'description':
                    dict_hdr[key] = strr.strip('{}').strip()
                    # dict.key = str.strip('{}').strip()
                else:
                    vals = strr[1:-1].split(',')
                    for j in range(len(vals)):
                        vals[j] = vals[j].strip()
                    dict_hdr[key] = vals
            else:
                dict_hdr[key] = val
                # dict.key = val
        if have_nonlowercase_param and not support_nonlowercase_params:
            msg = 'Parameters with non-lowercase names encountered ' \
                  'and converted to lowercase. To retain source file ' \
                  'parameter name capitalization, set ' \
                  'spectral.setttings.envi_support_nonlowercase_params to ' \
                  'True.'
            # warnings.warn(msg)
            print('Warning:', msg)
            print('Header parameter names converted to lower case.')
        # return dict <- Do some more conversion before return dict
    except:
        raise RuntimeError('ENVI header file parsing error.')

    cast_list = [('lines', np.int),
                 ('samples', np.int),
                 ('bands', np.int),
                 ('interleave', np.str),
                 #('geo info', np.str),
                 ('map info', np.str),
                 ('y start', np.int),
                 ('x start', np.int),
                 ('data type', np.int),
                 ('byte order', np.str),
                 ('good vis lines', np.str),
                 ('range of lines', np.str),
                 ('start date UTC', np.int),
                 ('start time UTC', np.float),
                 ('seconds per scan line', np.float),
                 ('lunar illuminance', np.float)]

    print(cast_list)
    for item in cast_list:
        if item[0] in list(dict_hdr.keys()):
            dict_hdr[item[0]] = np.array(dict_hdr[item[0]]).astype(item[1])
        else:
            dict_hdr[item[0]] = None
    print('map_info:', dict_hdr['map info'])
    print(type(dict_hdr['map info']))
    # print(','.split(dict_hdr['map info']))
    # start_lon = np.array(','.split(dict_hdr['map info'])[3]).astype(np.float)
    # start_lat = np.array(','.split(dict_hdr['map info'])[4]).astype(np.float)
    # gridsz = np.array(','.split(dict_hdr['map info'])[5]).astype(np.float)
    start_lon = np.array(dict_hdr['map info'][3]).astype(np.float)
    start_lat = np.array(dict_hdr['map info'][4]).astype(np.float)
    gridsz = np.array(dict_hdr['map info'][5]).astype(np.float)

    dict_hdr['start_lon'] = start_lon
    dict_hdr['start_lat'] = start_lat
    dict_hdr['gridsz'] = gridsz
    dict_hdr['n_lines'] = dict_hdr['lines']
    dict_hdr['n_samples'] = dict_hdr['samples']
    dict_hdr['geo info'] = dict_hdr['map info']

    dict_hdr['data_type'] = dict_hdr['data type']

    # When other projections become in use, put in a check for
    # geographic and only do ppd (pixels per degree)
    # calculation for geographic map projections
    ppd = np.round(1./gridsz)
    end_lat = start_lat - (dict_hdr['n_lines'] - 1)/ppd
    end_lon = start_lon + (dict_hdr['n_samples'] - 1)/ppd
    dict_hdr['map_limits'] = [start_lat, start_lon, end_lat, end_lon]

    return dict_hdr


def ngdc_get_temp_fname(prefix=None):

    if prefix is None:
        prefix = 'junk'

    suffix = '.tmp'
    fname = None

    result = 1
    # counter = 0
    while result:
        dd = (dt.now() - dt(1970, 1, 1))
        systime_seconds = dd.days * 86400. + dd.seconds + dd.microseconds/1.0e6
        number = (str(np.round(systime_seconds, 3)))[-7:]
        fname = prefix + number + suffix
        result = os.path.exists(fname)

    return fname


def dt_lookup(dtype_in, pos=2):

    '''
    This method map IDL datatype code to numpy datatype and vice versa
    for numpy type lookup, only take numpy.dtype.string
    :param dtype_in:
    :return:
    '''
    # Name, IDL code, numpy type, struct format code, nbyte
    dt_list = [('!NULL', 0, None, None),
               ('Byte', 1, np.dtype('uint8'), 'B', 1),
               ('Int', 2, np.dtype('int16'), 'h', 2),
               ('Long', 3, np.dtype('int32'), 'l', 4),
               ('Float', 4, np.dtype('float32'), 'f', 4),
               ('Double', 5, np.dtype('float64'), 'd', 8),  #float64, float
               ('Complex', 6, np.dtype('complex64'), 'ff', 8),
               ('String', 7, np.dtype('str'), 'c', 8),  #str, unicode
               ('Structure', 8, None, None),
               ('Double Complex', 9, np.dtype('complex128'), 'dd', 16),
               ('Pointer', 10, None, None),
               ('Objref', 11, None, None),
               ('Uint', 12, np.dtype('uint16'), 'I', 2),
               ('Ulong', 13, np.dtype('uint32'), 'L', 4),
               ('Long64', 14, np.dtype('int64'), 'q', 8),
               ('Ulong64', 15, np.dtype('uint64'), 'Q', 8)]
    npdt_list = [i for i in dt_list if isinstance(i[2], np.dtype)]

    # if input is text or numpy dtype object
    if isinstance(dtype_in, str):
        try:
            # a numpy data type name
            dtype = np.dtype(dtype_in)

            dt_find = [(d[0], d[1]) for d in npdt_list if isinstance(dtype, d[2])]

            if len(dt_find) != 1:
                return None
            else:
                return dt_find[0]

        except TypeError:
            if dtype_in not in [d[0] for d in dt_list]:
                raise TypeError('Not a valid IDL or Numpy datatype name.')
            # a IDL data type name
            dtype = dtype_in
            dt_find = [d[pos] for d in dt_list if d[0] == dtype]
            if len(dt_find) != 1:
                return None
            else:
                return dt_find[0]

    # if input is integer
    elif isinstance(dtype_in, int):
        dtype = dtype_in
        if dtype > 15:
            raise TypeError('IDL type code 0-15')
        # a IDL data type code
        dt_find = [d[pos] for d in dt_list if d[1] == dtype]
        if len(dt_find) != 1:
            return None
        else:
            return dt_find[0]

    # if input is a numpy datatype object
    elif isinstance(dtype_in, np.dtype):
        dtype = dtype_in

        dt_find = [(d[0], d[1]) for d in npdt_list if dtype == d[2]]

        if len(dt_find) != 1:
            return None
        else:
            return dt_find[0]


def ngdc_make_envi_hdr(infile, n_samples, n_lines, n_bands=None, header_offset=None,
#data.previous_filename = /eog/reference/python/h5_file/npp_d20180701_t0034511_e0040314_b34584/SVDNB_npp_d20180701_t0034511_e0040314_b34584_c20180701064031122643_nobc_ops.h5
                       data_type=None, byte_order=None, maxlat=None, minlon=None,
                       gridsz=None, map_info=None, description=None, band_names=None,
                       interleave=None, class_names=None, class_lookup=None):

    '''
    Note on keywords - ALL are OPTIONAL
    data_type - follow IDL convention for data types:
     byte = 1
     short = 2
     unsigned short = 12
     long = 3
     unsigned long = 13
     float = 4
     double = 5
     complex = 6
     double complex = 9

    :param infile:
    :param n_samples:
    :param n_lines:
    :param n_bands:
    :param header_offset:
    :param data_type:
    :param byte_order:
    :param maxlat:
    :param minlon:
    :param gridsz:
    :param map_info:
    :param description:
    :param band_names:
    :param interleave:
    :param class_names:
    :param class_lookup:
    :return:
    '''

    # Argument and keyword check
    if infile is None and n_samples is None and n_lines is None:
        print('USAGE: ngdc_make_envi_hdr(infile, n_samples, n_lines)')
        return

    if (minlon is None) + (maxlat is None) == 1:
        print('ERROR: Keywords minlon and maxlat must both be defined together.')
        return

    if minlon is not None and map_info is not None:
        print('ERROR: Keywords minlon and map_info are mutually exclusive.')
        return

    # The 2 classification keyword must be set together
    class_keyword_count = (class_names is not None) + (class_lookup is not None)
    if class_keyword_count == 1:
        print('ERROR: Keywords class_names and class_lookup must be defined together.')
        return

    # Error check classification keywords & format as strings
    n_classes = ''
    if class_keyword_count == 2:
        n_classes = len(class_names)
        n_class_lookup = len(class_lookup)
        if n_class_lookup != n_classes * 3:
            print('ERROR: Keyword class_lookup must contain one RGB color triple for each'
                  'element in class_names.')
            print('Number of RGB triples = ', n_class_lookup/3)
            print('Number of classes = ', n_classes)
            return

        # class_lookup =reform(class_lookup, n_class_lookup) #IDL
        class_lookup_dims = np.array(class_lookup).shape
        class_lookup = np.array(class_lookup).reshape(class_lookup_dims[class_lookup_dims > 1])
        class_names = str(class_names)
        n_classes = str(n_classes)

    # forcal all inputs to string & set defaults
    n_samples = str(n_samples)
    n_lines = str(n_lines)

    if n_bands is not None:
        n_bands = str(n_bands)
    else:
        n_bands = '1'
    # nb = float(n_bands)

    if interleave is None:
        interleave = 'bsq'

    if header_offset is not None:
        header_offset = str(header_offset)
    else:
        header_offset = '0'

    if data_type is not None:
        data_type = str(dt_lookup(data_type)[1])  # (IDL type name, IDL type code)
    else:
        data_type = '1'

    if byte_order is not None:
        byte_order = str(byte_order)
    else:
        byte_order = '0'

    if gridsz is not None:
        gridsz = str(gridsz)
    else:
        gridsz = str(1./120.)

    if class_names is not None:
        file_type = 'ENVI Classification'
    else:
        file_type = 'ENVI Standard'

    # Create ENVI header filename. If input filename has a .hdr extension
    # already, use that, otherwise add .hdr to the end.
    envi_hdr = infile + '.hdr'
    if infile.endswith('.hdr'):
        envi_hdr = infile

    # Write into the ENVI header
    with open(envi_hdr, 'w') as f:
        f.write('ENVI\n')
        if description is not None:
            f.write('description = {' + str(description) + '\n')
            f.write('}\n')
        f.write('samples = ' + n_samples + '\n')
        f.write('lines = ' + n_lines + '\n')
        f.write('bands = ' + n_bands + '\n')
        f.write('header offset = ' + header_offset + '\n')
        f.write('file type = ' + file_type + '\n')
        f.write('data type = ' + data_type + '\n')
        f.write('interleave = ' + interleave + '\n')
        f.write('sensor type = Unknown\n')
        f.write('byte order = ' + byte_order + '\n')
        if map_info is not None:
            f.write('map info = ' + str(map_info) + '\n')
        if minlon is not None:
            lon = str(minlon)
            lat = str(maxlat)
            f.write('map_info = {Geographic Lat/Lon, 1.5, 1.5,, ' +
                    lon + ',' + lat + ',' + gridsz + ', ' + gridsz + '}\n')
        if band_names is not None:
            # band_names = ' ' + str(band_names)
            # Create format statement for band_names in ENVI hdr
            band_names = ','.join([bn.strip() for bn in band_names])
            f.write('band names = {' + band_names + '}\n')
        if class_names is not None:
            f.write('classes = ' + n_classes + '\n')
            class_lookup = ', '.join([cl.strip() for cl in class_lookup])
            f.write('class lookup = {\n' + class_lookup + '}\n')
            class_names = ',\n'.join([cn.strip() for cn in class_names])
            f.write('class_names = {\n' + class_names + '}\n')


##
# Not using object, but dictionary
##
# def ngdc_get_fnames(filename, newfile=False):
#
#     '''
#     The NGDC_GET_FNAMES function returns filename information about a requested file. If
#     file does not exist on disk, the null string is returned. To get information about a
#     new file (i.e. one the calling code will create), set the newfile keyword.
#     '''
#
#     # Define file_info structure
#
#     file_info={
#         "fname":'',
#         "fname_hdr":'',
#         "compress":0
#     }
#
#     file_info['fname'] = filename
#     if os.path.splitext(filename)[-1] == '.gz':
#         l = len(filename)
#         file_info['fname_hdr'] = os.path.splitext(filename)[0]+'.hdr'
#         file_info['compress'] = True
#     else:
#         file_info['fname_hdr'] = filename+'.hdr'
#         file_info['compress'] = False
#
#     # Verify existence of files unless newfile keyword is set
#
#     if not newfile:
#         file_exists = os.path.exists(file_info['fname'])
#
#         if not file_exists and not file_info['compress']:
#             # see if there's a gzipped version
#             fname = filename + '.gz'
#             file_exists = os.path.exists(fname)
#             if file_exists:
#                 file_info['fname'] = fname
#                 file_info['compress'] = True
#         if not file_exists and file_info['compress']:
#             # see if there's an unzipped version
#             l = len(filename)
#             fname = os.path.splitext(filename)[0]
#             file_exists = os.path.exists(fname)
#             if file_exists:
#                 file_info['fname'] = fname
#                 file_info['compress'] = False
#
#         # if file is still not found, reset appropriate info structure tags to
#         # contain null values
#         if not file_exists:
#             file_info['fname'] = ''
#             file_info['compress'] = False
#
#         # Look for the hdr file, set to null string if not found
#         file_exists = os.path.exists(file_info['fname_hdr'])
#         if not file_exists:
#             file_info['fname_hdr'] = ''
#
#     return file_info
