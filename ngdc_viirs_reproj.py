#!/usr/bin/env python

import os, sys
import glob
import ngdc_file_tools as tools
import numpy as np
import ngdc_viirs_file_tools as vtools
import gzip
import subprocess
import shutil
from spectral import *


def ngdc_viirs_reproject_create_cfg(h5_datafile, cfg_fname=None, adj_search_dirs=None,
                                    gp_id=None, dp_id=None, data_group=None, data_name=None,
                                    geo_group=None, latitude_limits=None,
                                    geo_fname=None, create_lines_samples=False, out_file_prefix=None,
                                    out_ls_prefix=None, add_direction_to_out_file_prefix=False,
                                    out_dir=None, grid_sz=None, outlimits=False,
                                    spawn_reproj=False, gzip_reproj=False, no_overwrite=False):

    # adj_search_dir: list of dirs to search for h5 files with the same product id

    # See if h5 file has any "real data" in it. Some VIIRS data
    # products are valid only in the daytime (e.g. SVI0[1-3]), but h5 files
    # are still created and are blank-filled
    n_granule_status = vtools.ngdc_viirs_get_fname_granule_status(h5_datafile)
    if n_granule_status == 0:
        print('WARNING: NGDC_VIIRS_REPROJECT_CREATE_CFG: No valid data in h5 file %s. Not continuing...' % h5_datafile)
        return

    h5_dirname = os.path.dirname(h5_datafile)
    if out_dir is None:
    #     out_dir = ''
    # if out_dir == '':
        out_dir = h5_dirname
    h5_datafile_metadata = vtools.ngdc_viirs_get_fname_metadata(h5_datafile)

    if dp_id is None:
        dp_id = h5_datafile_metadata.data_product_ids
        if isinstance(dp_id, list):
            dp_id = dp_id[0]

    if geo_fname is None:
        # Get matching geo filename from file system
        if gp_id is None:
            gp_id = vtools.ngdc_viirs_get_geo_prefix(h5_datafile)
        geo_glob = '_'.join([gp_id,
                             h5_datafile_metadata.spacecraft_id,
                             h5_datafile_metadata.data_start_date.text,
                             h5_datafile_metadata.data_start_time.text,
                             h5_datafile_metadata.data_stop_time.text,
                             h5_datafile_metadata.orbit_number.text
                             ]) + '*.h5'
        print('h5_dirname', h5_dirname)
        geo_fname = glob.glob(os.path.join(h5_dirname, geo_glob))
        print('Geoname: %s' % os.path.join(h5_dirname, geo_glob))
        n_geo = len(geo_fname)

        '''
        Special case for reprojecting DNB files:
            Use GDTCN file for geolocation IF it is present. Otherwise
              use GDNBO file.
            Rational: Staring in May 2014, GDNBO files contain terrain 
            corrected layers Latitude_TC and Longitude_TC and ellipsoid
            geolocation layers Latitude and Longitude. This is unlike the
            other VIIRS datasets that have separate geolocation files for terrain
            corrected Latitude/Longitude layers. Reprojection code was
            modified to use the *_TC layers if they are there.
        '''
        if (n_geo == 0) and (gp_id == "GDTCN"):
            # NGDC Terrain correction file not found, try to use GDNBO file
            gp_id = 'GDNBO'
            geo_glob = '_'.join([gp_id,
                                 h5_datafile_metadata.spacecraft_id,
                                 h5_datafile_metadata.data_start_date.text,
                                 h5_datafile_metadata.data_start_time.text,
                                 h5_datafile_metadata.data_stop_time.text,
                                 h5_datafile_metadata.orbit_number.text
                                 ]) + '*.h5'

            print('Finding Geo Files: %s' % h5_dirname+geo_glob)
            geo_fname = glob.glob(os.path.join(h5_dirname+geo_glob))
            n_geo = len(geo_fname)

        if n_geo == 0:
            print('ERROR: NGDC_VIIRS_REPROJECT_CREATE_CFG: '
                  'Unable to find matching geo file for %s' % h5_datafile)
            return

        # take last if more than one match
        geo_fname = geo_fname[-1]

    # Set default output grid size
    if grid_sz is None:
        grid_sz = '0.00416667'

    # Set default latitude limits for output grid
    if latitude_limits is None:
        latitude_limits = [-65.0, 75.0]

    # Use default data group based on input data filename
    if data_group is None:
        data_group = ngdc_viirs_reproject_create_cfg_default_data_group(dp_id)
    if geo_group is None:
        geo_group = ngdc_viirs_reproject_create_cfg_default_data_group(gp_id)

    # Use default data name based on input data filename
    if data_name is None:
        data_name = ngdc_viirs_reproject_create_cfg_default_data_name(dp_id)

    refactor_dataset = ngdc_viirs_reproject_create_cfg_default_factors(data_name)

    # Create output filenames
    # h5_datafile_basename = os.path.splitext(os.path.basename(h5_datafile))[0]
    direction = '.'+vtools.ngdc_viirs_get_fname_direction(h5_datafile)

    if out_file_prefix is None:
        out_file_prefix = os.path.join(out_dir, os.path.splitext(os.path.basename(h5_datafile))[0])
    else:
        out_file_prefix = os.path.join(out_dir, os.path.basename(out_file_prefix))

    if add_direction_to_out_file_prefix:
        out_file_prefix = out_file_prefix + direction

    ext = ngdc_viirs_reproject_create_cfg_default_reproj_ext(data_name)
    if direction == '.d' and dp_id == 'SVDNB':
        ext = ext+'e9'
    out_fname = out_file_prefix + '.' + ext

    if out_ls_prefix is None:
        geo_basename = os.path.splitext(os.path.basename(geo_fname))[0]
        lines_fname = os.path.join(out_dir, geo_basename+'.lines')
        samples_fname = os.path.join(out_dir, geo_basename+'.samples')
    else:
        lines_fname = out_ls_prefix+'.lines'
        samples_fname = out_ls_prefix+'.samples'

    # Exit if no_overwrite keyword is set and output file checks out as valid
    if tools.ngdc_envi_file_integrity_check(out_fname):
        if no_overwrite:
            print('INFO: Output file %s already exists and'
                  'no_overwrite keyword is set. Returning...' % out_fname)
            return

    # Set apply_cal_coefficients flag if data needs to have a scale/offset
    # applied, or a user-supplied multiplier (like we do for nighttime DNB)
    # "Most of the time" this is when the refactor_dataset is supplied, but
    # des need to be turned off for M3,M4,M5,M7,M13, which are radiance
    # datasets, but are stored as radiance instead of scaled integers
    if dp_id == 'SVM07' or dp_id == 'SVM13' or dp_id == 'SVM03' or \
       dp_id == 'SVM04' or dp_id == 'SVM05' or refactor_dataset == '':
        apply_cal_coeffs = False
    else:
        apply_cal_coeffs = True

    # Multiply DNB radiance by 1E9 so they're not so small at night
    # TODO: Switch to using a multiplier for nighttime - righ now using orbital direction as a proxy
    data_multiplier = None
    if dp_id == 'SVDNB':
        refactor_dataset = ''
        if direction == '.d':
            data_multiplier = '1E9'
            apply_cal_coeffs = True

    # Create default config filename if not given
    if cfg_fname is None:
        cfg_fname = out_fname+'.cfg'

    # Set to use additional bow-tie deletion mask when reprojecting if
    # input is M-band data using GMTCO or GMODO reprojection files
    if gp_id == 'GMTCO' or gp_id == 'GMODO':
        use_addl_bowtie_deletions = True
    else:
        use_addl_bowtie_deletions = False

    # Get next/previous file from file system
    next_data_fname = vtools.ngdc_viirs_get_fname_adjacent(h5_datafile,
                                                           search_dirs=adj_search_dirs)
    prev_data_fname = vtools.ngdc_viirs_get_fname_adjacent(h5_datafile,
                                                           search_dirs=adj_search_dirs,
                                                           previous=True)
    if next_data_fname != '':
        next_geo_fname = vtools.ngdc_viirs_get_fname_adjacent(geo_fname,
                                                              search_dirs=adj_search_dirs)
    else:
        next_geo_fname = ''

    if prev_data_fname != '':
        prev_geo_fname = vtools.ngdc_viirs_get_fname_adjacent(geo_fname,
                                                              search_dirs=adj_search_dirs,
                                                              previous=True)
    else:
        prev_geo_fname = ''

    # TODO: Make more of these optional input keywords
    fw = open(cfg_fname, 'w')
    fw.write('#debug.num_scanLines = 200\n')
    fw.write('\n# Fit grid to extent of input file\n')
    fw.write('grid.fit = %s\n' % ('true', 'false')[outlimits])
    fw.write('\n# Grid spacing in degrees\n')
    fw.write('grid.spacing_x = %s\n' % str(grid_sz))
    fw.write('grid.spacing_y = %s\n' % str(grid_sz))
    fw.write('\n# Grid boundaries in degrees [-180, 180]\n')
    if outlimits:
        fw.write('grid.min_lat = %s\n' % str(outlimits[2]))
        fw.write('grid.max_lat = %s\n' % str(outlimits[0]))
        fw.write('grid.min_lon = %s\n' % str(outlimits[1]))
        fw.write('grid.max_lon = %s\n' % str(outlimits[3]))
    else:
        fw.write('#grid.min_lat = %s\n' % '29')
        fw.write('#grid.max_lat = %s\n' % '55')
        fw.write('#grid.min_lon = %s\n' % '49')
        fw.write('#grid.max_lon = %s\n' % '94')
    fw.write('\n# Limit output grid ROI\n')
    fw.write('limit.min_lat = %s\n' % str(latitude_limits[0]))
    fw.write('limit.max_lat = %s\n' % str(latitude_limits[1]))
    fw.write('#limit.min_lon = -180\n')
    fw.write('#limit.max_lon = 180\n')
    fw.write('\n# Data file(s)\n')
    fw.write('data.filename = %s\n' % h5_datafile)
    if next_data_fname != '' and next_geo_fname != '':
        fw.write('data.next_filename = %s\n' % next_data_fname)
    if prev_data_fname != '' and prev_geo_fname != '':
        fw.write('data.previous_filename = %s\n' % prev_data_fname)
    fw.write('data.group = %s\n' % data_group)
    fw.write('data.dataset = %s\n' % data_name)
    fw.write('data.refactor = %s\n' % ('false', 'true')[apply_cal_coeffs])
    if refactor_dataset != '':
        fw.write('data.refactordataset = %s\n' % refactor_dataset)
    if data_multiplier:
        fw.write('data.multiplier = %s\n' % data_multiplier)
    fw.write('\n# Geolocation file(s)\n')
    fw.write('geo.filename = %s\n' % geo_fname)
    if next_geo_fname != '' and next_data_fname != '':
        fw.write('geo.next_filename = %s\n' % next_geo_fname)
    if prev_geo_fname != '' and prev_data_fname != '':
        fw.write('geo.previous_filename = %s\n' % prev_geo_fname)
    fw.write('geo.group = %s\n' % geo_group)
    fw.write('\n# Missing values\n')
    fw.write('miss.useExternalFilter = %s\n' % str(use_addl_bowtie_deletions).lower())
    fw.write('miss.filterName = EDR2\n')
    fw.write('\n# Output ENVI file\n')
    fw.write('output.envi = true\n')
    fw.write('output.envi.filename = %s\n' % out_fname)
    fw.write('\n# Output PNG file\n')
    fw.write('output.png = false\n')
    fw.write('output.png.filename = fname.png\n')
    fw.write('grid.fill_value = 0\n')
    fw.write('\n# Lookup tables\n')
    fw.write('output.lookup = %s\n' % ('false', 'true')[create_lines_samples])
    fw.write('output.lookup.scanlines.filename = %s\n' % lines_fname)
    fw.write('output.lookup.pixels.filename = %s\n' % samples_fname)
    fw.close()

    # run VIIRS reprojection software
    if spawn_reproj:
        #path = '/eog/reference/java/viirs_reprojection'
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'java/viirs_reprojection')
        reproj_cmd = ['java', '-Xmx16384M', '-cp', 
            ''.join([
                #'-Xmx16384M', '-cp',
                #'java -Xmx1500M -cp ',
                path, '/lib/jhdf5/jhdf5.jar:',
                path, '/viirs_reprojection.jar:',
                path, '/lib/jcoord-1.1-b.jar'
            ]),
            ''.join([
                '-Djava.library.path=', path, '/lib/jhdf5/linux'
            ]),
            'viirs.ViirsProcessor', cfg_fname]
        if 'win' in sys.platform:
            reproj_cmd = reproj_cmd.replace('/', os.sep)
        print(' '.join(reproj_cmd))
        subprocess.run(reproj_cmd)
        if gzip_reproj:
            if not os.path.exists(out_fname):
                raise RuntimeError('ERROR: %s not found.' % out_fname)
                
            with open(out_fname, 'rb') as f_in:
                with gzip.open(out_fname+'.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # subprocess.run('gzip -vf '+out_fname)
            if create_lines_samples:
                with open(lines_fname, 'rb') as f_in:
                    with gzip.open(lines_fname+'.gz', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                with open(samples_fname, 'rb') as f_in:
                    with gzip.open(samples_fname+'.gz', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                # subprocess.run('gzip -vf '+lines_fname)
                # subprocess.run('gzip -vf '+samples_fname)


def ngdc_viirs_reproject_create_cfg_default_reproj_ext(data_name):

    data_name_lut = {
        'Radiance': 'rad',
        'SkinSST': 'sst_skin',
        'BulkSST': 'sst_bulk',
        'LandSurfaceTemperature': 'lst',
        'VegetationFraction': 'vfrac',
        'SurfaceType': 'stype',
        'TOA_NDVI': 'ndvi',
        'TOC_EVI': 'evi',
        'Chlorophyll_a': 'chla',
        'IOP_a_412nm': 'iopa_412',
        'IOP_a_445nm': 'iopa_445',
        'IOP_a_488nm': 'iopa_488',
        'IOP_a_555nm': 'iopa_555',
        'IOP_a_672nm': 'iopa_672',
        'IOP_s_412nm': 'iops_412',
        'IOP_s_445nm': 'iops_445',
        'IOP_s_488nm': 'iops_488',
        'IOP_s_555nm': 'iops_555',
        'IOP_s_672nm': 'iops_672',
        'nLw_412nm': 'nlw_412',
        'nLw_445nm': 'nlw_445',
        'nLw_488nm': 'nlw_488',
        'nLw_555nm': 'nlw_555',
        'nLw_672nm': 'nlw_672',
        'cot': 'cot',
        'ctp': 'ctp',
        'ctt': 'ctt',
        'cth': 'cth',
        'cbh': 'cbh',
        'cloudType': 'cloudtype',
        'cloudLayer': 'cloudlayer',
        'eps': 'eps',
        'faot550': 'faot550',
        'angexp': 'angexp',
        'RainRate': 'rain_rate',
        'Albedo': 'alb',
        'AerosolOpticalDepth_at_550nm': 'aot550',
        'ColumnAmountO3': 'cto3'
    }

    try:
        return data_name_lut[data_name]
    except KeyError:
        return str.lower(data_name)


def ngdc_viirs_reproject_create_cfg_default_factors(data_name):

    factors_lut = {
        'Radiance': 'RadianceFactors',
        'BrightnessTemperature': 'BrightnessTemperatureFactors',
        'SkinSST': 'SkinSSTFactors',
        'BulkSST': 'BulkSSTFactors',
        'LandSurfaceTemperature': 'LSTFactors',
        'VegetationFraction': 'VegetationFractionFactors',
        'TOA_NDVI': 'TOA_NDVI_Factors',
        'TOC_EVI': 'TOC_EVI_Factors',
        'Albedo': 'AlbedoFactors',
        'AerosolOpticalDepth_at_550nm': 'AerosolOpticalDepthFactors',
    }

    try:
        return factors_lut[data_name]
    except KeyError:
        #return str.lower(data_name)
        return ''


def ngdc_viirs_reproject_create_cfg_default_data_name(data_product_id):

    data_product_id_lut = {
        'SVDNB': 'Radiance',
        'SVM01': 'Radiance',
        'SVM02': 'Radiance',
        'SVM03': 'Radiance',
        'SVM04': 'Radiance',
        'SVM05': 'Radiance',
        'SVM06': 'Radiance',
        'SVM07': 'Radiance',
        'SVM08': 'Radiance',
        'SVM09': 'Radiance',
        'SVM10': 'Radiance',
        'SVM11': 'Radiance',
        'SVM12': 'Radiance',
        'SVM13': 'Radiance',
        'SVM14': 'Radiance',
        'SVM15': 'Radiance',
        'SVM16': 'Radiance',
        'SVI01': 'Radiance',
        'SVI02': 'Radiance',
        'SVI03': 'Radiance',
        'SVI04': 'Radiance',
        'SVI05': 'Radiance',
        'VSSTO': 'SkinSST',
        'VLSTO': 'LandSurfaceTemperature',
        'VOCCO': 'Chlorophyll_a',
        'VSTYO': 'SurfaceType',    # other layer is VegetationFraction
        'VIVIO': 'TOA_NDVI',       # other layer is TOC_EVI
        'IVCOP': 'cot',
        'IVICC': 'cloudType',      # other layer is cloudLayer
        'IVCTP': 'ctp',            # other layers are ctt, cth
        'IVCBH': 'cbh',
        'IVAOT': 'faot550',
        'IICMO': 'QF1_VIIRSCMIP',
        'VISAO': 'Albedo',
        'DEPTMS': 'Rain Rate',
        'VAOOO': 'AerosolOpticalDepth_at_550nm',
        'OOCTO': 'ColumnAmountO3'
    }
    if isinstance(data_product_id, list):
        if len(data_product_id) > 1:
            print('Multiple data_product_id, use first one.')
        data_product_id = list(data_product_id)[0]
    try:
        return data_product_id_lut[data_product_id]
    except KeyError:
        return ''


def ngdc_viirs_reproject_create_cfg_default_data_group(data_product_id):

    data_product_id_lut = {
        'SVDNB': '/All_Data/VIIRS-DNB-SDR_All',
        'SVM01': '/All_Data/VIIRS-M1-SDR_All',
        'SVM02': '/All_Data/VIIRS-M2-SDR_All',
        'SVM03': '/All_Data/VIIRS-M3-SDR_All',
        'SVM04': '/All_Data/VIIRS-M4-SDR_All',
        'SVM05': '/All_Data/VIIRS-M5-SDR_All',
        'SVM06': '/All_Data/VIIRS-M6-SDR_All',
        'SVM07': '/All_Data/VIIRS-M7-SDR_All',
        'SVM08': '/All_Data/VIIRS-M8-SDR_All',
        'SVM09': '/All_Data/VIIRS-M9-SDR_All',
        'SVM10': '/All_Data/VIIRS-M10-SDR_All',
        'SVM11': '/All_Data/VIIRS-M11-SDR_All',
        'SVM12': '/All_Data/VIIRS-M12-SDR_All',
        'SVM13': '/All_Data/VIIRS-M13-SDR_All',
        'SVM14': '/All_Data/VIIRS-M14-SDR_All',
        'SVM15': '/All_Data/VIIRS-M15-SDR_All',
        'SVM16': '/All_Data/VIIRS-M16-SDR_All',
        'SVI01': '/All_Data/VIIRS-I1-SDR_All',
        'SVI02': '/All_Data/VIIRS-I2-SDR_All',
        'SVI03': '/All_Data/VIIRS-I3-SDR_All',
        'SVI04': '/All_Data/VIIRS-I4-SDR_All',
        'SVI05': '/All_Data/VIIRS-I5-SDR_All',
        'VSSTO': '/All_Data/VIIRS-SST-EDR_All',
        'VLSTO': '/All_Data/VIIRS-LST-EDR_All',
        'VSTYO': '/All_Data/VIIRS-ST-EDR_All',
        'VIVIO': '/All_Data/VIIRS-VI-EDR_All',
        'VOCCO': '/All_Data/VIIRS-OCC-EDR_All',
        'IVCOP': '/All_Data/VIIRS-Cd-Opt-Prop-IP_All',
        'IVICC': '/All_Data/VIIRS-Cd-Layer-Type-IP_All',
        'IVCTP': '/All_Data/VIIRS-Cd-Top-Parm-IP_All',
        'IVCBH': '/All_Data/VIIRS-CB-Ht-IP_All',
        'IVAOT': '/All_Data/VIIRS-Aeros-Opt-Thick-IP_All',
        'IICMO': '/All_Data/VIIRS-CM-IP_All',
        'VISAO': '/All_Data/VIIRS-SA-EDR_All',
        'DEPTMS': '/All_Data/EDR_TMS_ORB_All',
        'GMODO': '/All_Data/VIIRS-MOD-GEO_All',
        'GMTCO': '/All_Data/VIIRS-MOD-GEO-TC_All',
        'GDNBO': '/All_Data/VIIRS-DNB-GEO_All',
        'GDTCN': '/All_Data/VIIRS-DNB-GEO-TC_All',
        'GIMGO': '/All_Data/VIIRS-IMG-GEO_All',
        'GITCO': '/All_Data/VIIRS-IMG-GEO-TC_All',
        'VAOOO': '/All_Data/VIIRS-Aeros-EDR_ALL',
        'GAERO': '/All_Data/VIIRS-Aeros-EDR-GEO_All',
        'OOCTO': '/All_Data/OMPS-TC-EDR_All',
        'GOTCO': '/All_Data/OMPS-TC-GEO_All'
    }
    if isinstance(data_product_id, list):
        if len(data_product_id) > 1:
            print('Multiple data_product_id, use first one.')
        data_product_id = list(data_product_id)[0]
    try:
        return data_product_id_lut[data_product_id]
    except KeyError:
        return ''


def ngdc_viirs_reproj_to_dnb(h5_file, dnb_h5_file, out_file,
                             data_name=None, granule_prefix=None,
                             gp_id=None, out_file_ext=None, preserve_reproj=False):

    '''
    :param h5_file:
    :param dnb_h5_file:
    :param out_file:
    :param data_name:
    :param granule_prefix:
    :param gp_id:
    :param out_file_ext:
    :param preserve_reproj:
    :return:
    '''

    # Reproject first - this determines spatial extent
    if preserve_reproj:
        out_file_prefix = os.path.splitext(h5_file)[0]
    else:
        out_file_prefix = tools.ngdc_get_temp_fname(prefix='tempreproj')

    if out_file_ext is None:
        out_file_ext = 'rad'

    # Only spawn reprojection if needed
    # Note: HACK - should check for file (compressed or raw) AND ENVI hdr
    if not os.path.exists(out_file_prefix + '.' + out_file_ext + '.hdr'):
        h5_dirname = os.path.dirname(h5_file)
        base_dir = os.path.dirname(h5_dirname)
        adj_search_dirs = [i for i in glob.glob(base_dir+'npp_d*') if os.path.isdir(i)]
        ngdc_viirs_reproject_create_cfg(h5_file, out_file_prefix=out_file_prefix,
                                        grid_sz=1./240., spawn_reproj=True,
                                        gp_id=gp_id, latitude_limits=[-90, 90],
                                        adj_search_dirs=adj_search_dirs)
    # Get spatial extent of reprojection
    hdr = tools.envi_hdr(out_file_prefix + '.' + out_file_ext + '.hdr')
    max_lat = hdr['start_lat']
    min_lon = hdr['start_lon']
    gridsz = hdr['gridsz']
    nl = hdr['nl']
    ns = hdr['ns']
    map_limits = hdr['map_limits']

    min_lat = np.round(max_lat - nl*gridsz)
    max_lon = np.round(min_lon + ns*gridsz)

    # Get filename of matching DNB geo file
    dnb_h5_file_metadata = vtools.ngdc_viirs_get_fname_metadata(dnb_h5_file)
    dnb_h5_dirname = os.path.dirname(dnb_h5_file)
    dnb_gp_id = vtools.ngdc_viirs_get_geo_prefix(dnb_h5_file)
    geo_glob = '_'.join([dnb_gp_id,
                         dnb_h5_file_metadata.spacefraft_id,
                         dnb_h5_file_metadata.data_start_date.text,
                         dnb_h5_file_metadata.data_start_time.text,
                         dnb_h5_file_metadata.data_stop_time.text,
                         dnb_h5_file_metadata.orbit_number.text]) + '*.h5'
    geo_fname = glob.glob(dnb_h5_dirname + geo_glob)
    n_geo = len(geo_fname)
    lat_data_name = 'All_Data/VIIRS-DNB-GEO_All/Latitude_TC'
    lon_data_name = 'All_Data/VIIRS-DNB-GEO_All/Longitude_TC'
    if n_geo == 0:
        dnb_gp_id = 'GDNBO'
        geo_glob = '_'.join([dnb_gp_id,
                             dnb_h5_file_metadata.spacecraft_id,
                             dnb_h5_file_metadata.data_start_date.text,
                             dnb_h5_file_metadata.data_start_time.text,
                             dnb_h5_file_metadata.data_stop_time.text,
                             dnb_h5_file_metadata.orbit_number.text]) + '*.h5'
        geo_fname = glob.glob(dnb_h5_dirname + geo_glob)
        n_geo = len(geo_fname)
        lat_data_name = 'All_Data/VIIRS-DNB-GEO_All/Latitude_TC'
        lon_data_name = 'All_Data/VIIRS-DNB-GEO_All/Longitude_TC'
    if n_geo == 0:
        print('ERROR: NGDC_VIIRS_REPROJECT_CREATE_CFG: ' +
              'Unable to find matching geo file for input ' + dnb_h5_file)
        return

    # take last if more than one match
    geo_fname = geo_fname[-1]

    # Convert DNB lat/lon data to reproj. grid indices.
    # Adjust longitudes if grid crosses -180/180 line.
    latitude = vtools.ngdc_viirs_read_h5(geo_fname, lat_data_name)
    longitude = vtools.ngdc_viirs_read_h5(geo_fname, lon_data_name)
    if (min_lon < 180) and (max_lon > 180):
        # Adjust negative longitudes to match grid
        longitude[longitude < 0] = longitude[longitude < 0] + 360
    xgrid = np.round((longitude - min_lon)/gridsz)
    ygrid = np.round((max_lat - latitude)/gridsz)

    # Read in data from reprojection
    out_file_info = tools.ngdc_get_fnames(out_file_prefix + '.' + out_file_ext)
    data = tools.ngdc_get_image_from_file

    out_data = data[ygrid, xgrid]
    dt = out_data.dtype
    sz_out = out_data.shape

    # TODO - fill out_data with VIIRS no-data value for this data type

    outfile_info = tools.ngdc_get_fnames(out_file, newfile=True)
    if out_file_info['compress']:
        f = gzip.GzipFile(outfile_info.fname)
    else:
        f = open(out_file_info['fname'])
    f.write(out_data)
    f.close()

    tools.ngdc_make_envi_hdr(out_file_info['fname_hdr'], sz_out[1], sz_out[0], data_type=dt)

    # Delete temporary reprojected files
    if not preserve_reproj:
        for i in glob.glob(out_file_prefix+'*'):
            os.remove(i)


def reproject_with_lines_samples(infile, outfile, 
                                 linesfile=None, samplesfile=None,
                                 no_data_val=None, mb_per_tile=None):
    '''
    The REPROJECT_WITH_LINES_SAMPLES routine reprojects an input image using
    "lines" and "samples" files which contain information about how the input
    image is mapped into an output projection.  The "lines" file contains
    an image in output projection space whose values represent input image
    y-coordinate or along track positioning information. The "samples" file
    contains an image in output projection space whose values represent input 
    image x-coordinate or along-scan positioning information.
    @author Kimberly Baugh
    @version $Id$
    @history
      Initial Revision: Octover 16, 2003 <BR>
    @param infile {in} {type=string} {required} A scalar filename containing
      the input iimage to be reprojected.
    @param outfile {in} {type=string} {required} A scalar filename containing
      the name of file to write the output image.  If a file with this name
      already exists, it will be overwritten.  To output a gzip compressed file,
      give an outfile with .gz as the last 3 characters.
    @keyword linesfile {in} {type=string} {required} A scalar filename of the
      file with the image in output projection space containing the line numbers
      (y-values) of the input image.
    @keyword samplesfile {in} {type=string} {required} A scalar filename of the
      file with the image in output projection space containing the sample numbers
      (x-values) of the input image.
    @keyword no_data_val {in} {type=numeric} {optional} {default=0} A numeric
      value of same type as the input image to be sued as the background value
      of the output image. If no_data_val is given as another data type it will
      be recast to match the input image data type before using.
    @keyword mb_per_tile {in} {type=integer} {default=200}  The value in
      megabytes of RAM to be used in processing. The default value is 200.
    @examples
      from x import reproject_with_lines_samples
      reproject_with_lines_samples('F1219902160320.asc.OISlun',
        'F12199902160320.asc.lun', linesfile='F12199902160320.asc.lines',
        samplesfile='F12199902160320.asc.samples', no_data_val=-1)
    @uses
      ngdc_read_envi_hdr <BR>
      ngdc_get_fnames <BR>
      ngdc_make_envi_hdr <BR>
      lut_bytes_per_pixel <BR>
    '''

    print('Processing input file: %s' % infile)

    # Check input parameters
    if infile is None and outfile is None:
        print('ERROR: Called with incorrect number of arguments')
        return None
    if linesfile is None:
        print('ERROR: linesfile keyword must be set.')
        return None
    if samplesfile is None:
        print('ERROR: samplesfile keyword must be set.')
        return None

    if mb_per_tile is None:
        mb_per_tile = 200
    if no_data_val is None:
        no_data_val = 0

    # If output filename has .gz etension assume user wants output file gzip
    # compressed.
    outfile_info = tools.dmsp_get_fnames(outfile, newfile=True)

    # Check input filenames and their ENVI headers for existence, and get 
    # compression states using dmsp_get_fnames
    infile_info = tools.dmsp_get_fnames(infile)
    if infile_info.fname == '' or infile_info.fname_hdr == '':
        print('ERROR: Unable to find input file: %s and/or its ENVI header.' % infile)
        return None

    linesfile_info = tools.dmsp_get_fnames(linesfile)
    if linesfile_info.fname == '' or linesfile_info.fname_hdr == '':
        print('ERROR: Unable to find lines file: %s and/or its ENVI header.' % liensfile)
        return None

    samplesfile_info = tools.dmsp_get_fnames(samplesfile)
    if samplesfile_info.fname == '' or samplesfile_info.fname_hdr == '':
        print('ERROR: Unable to find samples file: %s and/or its ENVI header.' % samplesfile)

    # Check that user input lines file has ".lines" in its filename, and user
    # input smaples file has ".samples" in its filename - give warning if either
    # doesn't.
    if '.lines' not in linesfile_info.fname:
        print('WARNING: User input linesfile does not have .lines extension.')
    if '.samples' not in samplesfile_info.fname:
        print('WARNING: User input samplesfile does not have .samples extension.')

    hdr_info = tools.envi_hdr(linesfile_info.fname_hdr)
    nl = hdr_info['lines']
    ns = hdr_info['samples']
    map_info = hdr_info['map info']
    dt_lines = hdr_info['data type']
    hdr_info = tools.envi_hdr(samplesfile_info.fname_hdr)
    nl_samples = hdr_info['lines']
    ns_samples = hdr_info['samples']
    dt_samples = hdr_info['data type']

    if (nl != nl_samples) or (ns != ns_samples):
        print('ERROR: Input liens and samples files do not have same dimensions.')
        return None

    # Read in entire input file - assumption is that input file will be 
    # small enough to hold in memory in its entirety. If that proves to be
    # a bad assumption, will need to read in section after knowing the range 
    # of lines needed for reprojection.
    hdr_info = tools.envi_hdr(infile_info.fname_hdr)
    ns_in = hdr_info['samples']
    nl_in = hdr_info['lines']
    dt_infile = hdr_info['data type']
    # in_image = np.zeros([nl_in, ns_in], dtype=dt_infile_np)
    if not infile_info.compress:
        dt_numpy = tools.dt_lookup(int(dt_infile), pos=2)
        array = np.fromfile(infile_info.fname, dtype=dt_numpy)
    else:
        # read compressed file into array
        with gzip.GzipFile(infile_info.fname, 'r') as f:
            img = f.read()
        dt_code = tools.dt_lookup(int(dt_infile), pos=3)
        format = str(ns_in * nl_in)+dt_code
        array = np.array(struct.unpack(format, img))
    # hold reshaping for tiling process --david 20191211
    # in_image = array.reshape(ns_in, nl_in)

    # Set up tiling procedure (tiling in output space)
    bytes_per_pixel = (tools.lut_bytes_per_pixel(dt_lines) + 
                       tools.lut_bytes_per_pixel(dt_samples) +
                       tools.lut_bytes_per_pixel(dt_infile))
    bytes_per_mbyte = 10.^6
    nl_per_tile = round((mb_per_tile * bytes_per_mbyte) / (bytes_per_pixel * ns))
    n_tiles = np.ceil(nl / float(nl_per_tile))
    nl_per_last_tile = nl - (n_tiles - 1) * nl_per_tile

    # Allocate arrays
    lines = np.zeros(ns, nl_per_tile, dtype=tools.dt_lookup(int(dt_lines), pos=2))
    samples = np.zeros(ns, nl_per_tile, dtype=tools.dt_lookup(int(dt_samples), pos=2))
    out_image = np.zeros(ns, nl_per_tile, dtype=tools.dt_lookup(int(dt_infile), pos=2))

    # Open input/output files
    if not linesfile_info.compress:
        #dt_numpy = tools.dt_lookup(int(dt_lines), pos=2)
        lines_read = open(linesfile_info.fname, 'rb')
    else:
        lines_read = gzip.GzipFile(linesfile_info.fname, 'r')
    if not smaplesfile_info.compress:    
        samples_read = open(samplesfile_info.fname, 'rb')
    else:
        samples_read = gzip.GzipFile(samplesfile_info.fname, 'r')
    if not outfile_info.compress:
        out_write = open(outfile, 'wb')
    else:
        out_write = gzip.GzipFile(outfile, 'w')
    
    for i in range(0, n_tiles):
        print('Processing tile: %s / %s' % (str(i), str(n_tiles)))
        if i == n_tiles-1:
            # Re-allocate arrays
            lines = np.zeros(ns, nl_per_last_tile, dtype=tools.dt_lookup(int(dt_lines), pos=2))
            samples = np.zeros(ns, nl_per_last_tile, dtype=tools.dt_lookup(int(dt_samples), pos=2))
            out_image = np.zeros(ns, nl_per_last_tile, dtype=tools.dt_lookup(int(dt_infile), pos=2))
            nl_this_tile = nl_per_last_tile
        else:
            nl_this_tile = nl_per_tile

        lines = struct.unpack(lines_format, 
                              lines_read.read(ns * nl_per_tile * tools.dt_lookup(int(dt_lines), pos=4)))
        samples = struct.unpack(samples_format,
                                samples_read.read(ns * nl_per_tile * tools.dt_lookup(int(dt_samples), pos=4)))
        lines = lines.reshape(ns, nl_per_tile)
        sample = samples.reshape(ns, nl_per_tile)

        # Store "no data" pixels
        idx = np.where(samples == 0)
        count_nodata = len(idx)

        # Reconcile lines/samples to 0/0 to origin
        samples_concile = ((samples-1)>0).astype(int)*(samples-1)
        samples[0:samples_concile.shape[0], 0:samples_concile.shape[1]] = samples_concile
        lines_concile = ((lines-1)>0).astype(int)*(lines-1)
        lines[0:lines_concile.shpae[0], 0:lines_concile.shape[1]] = lines_concile

        # Do reprojection
        out_image[0:lines.shape[0],0:lines.shape[1]] = in_image[samples, lines]
        
        # Set no-data back into flag file (where lines/samples eq 0)
        if count_nodata > 0:
            out_image[idx] = no_data_val
        out_write(out_image)

    # Close files
    lines_read.close()
    samples_read.close()
    out_write.close()

    # Make ENVI header for output file
    tools.ngdc_make_envi_hdr(outfile_info.fname_hdr, ns, nl, map_info=map_info,
                             data_type=dt_infile)

